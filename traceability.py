import os
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DEPARTMENTS = {
    "rental-contracts":    "Mietverträge — Fragen zum Mietvertrag, Verlängerungen, Änderungen",
    "terminations-moveout": "Kündigungen & Auszug — Kündigungen, Auszugstermine, Kautionsrückzahlung",
    "tenant-complaints":   "Mieterbeschwerden — Lärm, Nachbarschaftsstreit, allgemeine Beschwerden",
    "energy-heating":      "Energie & Heizung — Heizungsausfälle, Warmwasser, Nebenkostenabrechnung",
    "repairs-maintenance": "Reparaturen & Instandhaltung — defekte Einrichtungen, Gebäudeschäden, allgemeine Reparaturen",
}

class AuthResult(BaseModel):
    verified: bool
    customer_name: str
    reason: str

class RoutingDecision(BaseModel):
    department: str        # einer der fünf Schlüssel oben
    routing_reason: str    # warum diese Abteilung — wichtig für spätere Analysen
    issue_summary: str
    confidence: str        # "niedrig" / "mittel" / "hoch"


# Der LangFuse-Wrapper fängt alle OpenAI-Aufrufe ab und traced sie automatisch
from langfuse.openai import openai
from langfuse import get_client, observe, propagate_attributes

langfuse = get_client()
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

@observe(name="step-greeting")       # Kind-Span: Begrüßung
def greet_and_collect_name(customer_message: str) -> str:
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Du bist eine freundliche Empfangskraft bei der Hausverwaltung. "
                "Begrüße den Mieter herzlich und antworte immer auf Deutsch. "
                "Falls er seinen Namen noch nicht genannt hat, frage danach."
            )},
            {"role": "user", "content": customer_message}
        ]
    )
    return response.choices[0].message.content

@observe(name="step-auth")
def verify_tenant(name: str, address: str) -> AuthResult:
    # Manueller Sub-Span innerhalb der @observe-Funktion
    with langfuse.start_as_current_observation(name="address-format-check") as span:
        is_plausible = len(address.split()) >= 2
        span.update(metadata={"raw_address": address, "passed_format": is_plausible})

    response = openai.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Du simulierst ein Mieter-Verifizierungssystem. "
                "Wenn die Adresse plausibel klingt (Straßenname + Hausnummer + Stadt), "
                "markiere den Mieter als verifiziert. "
                "Schreibe die Begründung (reason) auf Deutsch."
            )},
            {"role": "user", "content": f"Name: {name}\nAdresse: {address}"}
        ],
        response_format=AuthResult
    )
    return response.choices[0].message.parsed

@observe(name="step-routing")
def route_to_department(transcript: str) -> RoutingDecision:
    dept_list = "\n".join(f"- {key}: {desc}" for key, desc in DEPARTMENTS.items())
    response = openai.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Du leitest Mieteranfragen bei der Hausverwaltung an die richtige Abteilung weiter. "
                "Wähle genau einen Abteilungs-Schlüssel aus dieser Liste:\n" + dept_list + "\n"
                "Schreibe routing_reason und issue_summary auf Deutsch. "
                "confidence muss genau einer dieser Werte sein: \"niedrig\", \"mittel\", \"hoch\"."
            )},
            {"role": "user", "content": transcript}
        ],
        response_format=RoutingDecision
    )
    return response.choices[0].message.parsed


@observe(name="tenant-routing-call")
def handle_call():
    session_id = uuid.uuid4().hex[:8]
    with propagate_attributes(session_id=session_id):
        opening = input("Mieter: ").strip()
        greeting = greet_and_collect_name(opening)
        print(f"Agent: {greeting}")

        name = input("\nMieter (Name): ").strip()
        address = input("Mieter (Adresse): ").strip()
        auth = verify_tenant(name, address)

        if not auth.verified:
            with propagate_attributes(tags=["auth-failed"]):
                print("Agent: Es tut mir leid, ich konnte Ihre Angaben nicht verifizieren.")
            return

        print(f"Agent: Vielen Dank, {auth.customer_name}. Wie kann ich Ihnen heute helfen?")
        issue = input("Mieter: ").strip()

        transcript = f"Eröffnung: {opening}\nName: {name}\nAdresse: {address}\nAnliegen: {issue}"
        routing = route_to_department(transcript)

        # Routing-Ergebnis am übergeordneten Trace anhängen (fürs Filtern im Dashboard)
        with propagate_attributes(
            tags=[routing.department, f"confidence-{routing.confidence}"],
            metadata={
                "routing_department": routing.department,
                "routing_reason": routing.routing_reason,
                "confidence": routing.confidence,
                "customer_name": auth.customer_name,
            }
        ):
            print(f"\n✅ Weiterleitung an: {DEPARTMENTS[routing.department]}")
            print(f"   Grund: {routing.routing_reason}")
            print(f"   Konfidenz: {routing.confidence}")
        return routing

if __name__ == "__main__":
    handle_call()
    langfuse.flush()

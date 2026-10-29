"""Hausverwaltungs-Chatbot mit Langfuse-Tracing.

Ablauf eines Anrufs (handle_call, ein Trace pro Session):
1. Begrüßung (step-greeting): Der Mieter schildert sein Anliegen. Aufruf via
   openai.chat.completions.create mit System-Prompt "Empfangskraft" und der
   Mieter-Nachricht als User-Message; Rückgabe ist der freie Antworttext
   (str) aus response.choices[0].message.content.
2. Verifizierung (step-auth): Name und Adresse werden abgefragt; ein
   Sub-Span prüft das Adressformat. Aufruf via
   openai.beta.chat.completions.parse mit response_format=AuthResult —
   das LLM muss also JSON passend zum Pydantic-Modell liefern. Rückgabe
   ist ein AuthResult (verified: bool, customer_name, reason) aus
   response.choices[0].message.parsed. Bei verified=False endet der
   Anruf mit dem Tag "auth-failed".
3. Routing (step-routing): Das Gesprächstranskript (Eröffnung, Name,
   Adresse, Anliegen) geht als User-Message an
   openai.beta.chat.completions.parse mit response_format=RoutingDecision;
   der System-Prompt enthält die fünf Abteilungen. Rückgabe ist eine
   RoutingDecision (department, routing_reason, issue_summary, confidence).
4. Das Routing-Ergebnis wird als Tags/Metadaten an den Trace gehängt und
   dem Mieter ausgegeben.

Alle Aufrufe nutzen das Modell aus der Env-Variable LLM_MODEL
(Default: gpt-4o-mini).

Wichtig: Die LLM-Aufrufe sind zustandslos und bekommen die Ausgaben der
vorherigen LLMs NICHT als Kontext. Jeder Aufruf erhält nur die rohen
Nutzereingaben: die Begrüßungsantwort wird nur ausgegeben, das Routing
sieht weder greeting noch AuthResult.reason — sein Transkript wird in
handle_call aus opening, name, address und issue zusammengebaut.

    opening = input(Anliegen)
            │
            ▼
    ┌─────────────────────┐  LLM-Input: opening
    │ 1. step-greeting    │────────────────────▶ greeting (str)
    └─────────────────────┘                      nur print, wird nicht
            │                                    weitergereicht
    name, address = input(...)
            ▼
    ┌─────────────────────┐  LLM-Input: "Name: {name}\nAdresse: {address}"
    │ 2. step-auth        │────────────────────▶ AuthResult
    └─────────────────────┘                      genutzt: verified (Abbruch?),
            │                                    customer_name (Anrede)
      verified? ──nein──▶ Ende (Tag "auth-failed")
            │ ja
    issue = input(Anliegen konkret)
            ▼
    ┌─────────────────────┐  LLM-Input: transcript =
    │ 3. step-routing     │  opening + name + address + issue
    └─────────────────────┘────────────────────▶ RoutingDecision
            │
            ▼
    4. Tags/Metadaten an Trace, Ausgabe an Mieter

Alle OpenAI-Aufrufe werden über den Langfuse-Wrapper automatisch getraced;
propagate_attributes verknüpft die Schritte über eine gemeinsame session_id.
"""

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

# Ergebnis von Schritt 2 (verify_tenant). Wird dort als response_format an die
# API übergeben — das Modell muss also JSON liefern, das genau hierher passt.
class AuthResult(BaseModel):
    verified: bool         # steuert den Ablauf: bei False bricht handle_call ab (Tag "auth-failed")
    customer_name: str     # nur für die Anrede im Gespräch und als Trace-Metadatum
    reason: str            # Begründung des Modells; wird nirgends ausgewertet, nur getraced

# Ergebnis von Schritt 3 (route_to_department), ebenfalls per response_format erzwungen.
class RoutingDecision(BaseModel):
    department: str        # einer der fünf Schlüssel oben; dient als DEPARTMENTS-Lookup und als Trace-Tag
    routing_reason: str    # warum diese Abteilung — wird ausgegeben und als Metadatum gespeichert
    issue_summary: str     # Kurzfassung des Anliegens; aktuell ungenutzt, nur im Trace sichtbar
    confidence: str        # "niedrig" / "mittel" / "hoch"; wird zum Tag "confidence-<wert>"


# Der LangFuse-Wrapper fängt alle OpenAI-Aufrufe ab und traced sie automatisch
from langfuse.openai import openai
from langfuse import get_client, observe, propagate_attributes

langfuse = get_client()
model = os.getenv("LLM_MODEL", "gpt-4o-mini")

@observe(name="step-greeting")       # Type-Span: Begrüßung
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
        opening = input("Wie kann ich ihnen helfen?: ").strip()
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

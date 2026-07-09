"""Teil 2: Derselbe Support-Bot MIT Langfuse-Tracing.

Gegenüber Teil 1 (support_bot.py) ändern sich nur die mit "NEU" markierten
Stellen - die Klasse selbst bleibt identisch:
1. Der OpenAI-Client kommt aus langfuse.openai (Drop-in-Wrapper):
   Jeder LLM-Aufruf wird automatisch geloggt (Prompt, Tokens, Dauer).
2. Der @observe-Dekorator macht aus jedem beantworte()-Aufruf einen Trace.
3. propagate_attributes ergänzt Session-ID, Nutzer, Tags und Metadaten.
4. flush() sendet am Programmende die restlichen Traces.

Einmaliges Setup:
    1. Kostenloser Account: https://cloud.langfuse.com
    2. Projekt -> Settings -> API Keys -> neue Keys erzeugen
    3. In die .env eintragen:
           LANGFUSE_PUBLIC_KEY=pk-lf-...
           LANGFUSE_SECRET_KEY=sk-lf-...
           LANGFUSE_HOST=https://cloud.langfuse.com
"""

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# NEU (1): statt "from openai import OpenAI" - der Langfuse-Wrapper um das
# OpenAI-SDK. Gleiche Schnittstelle, aber jeder Aufruf wird getract.
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

load_dotenv()
assert os.getenv("LANGFUSE_PUBLIC_KEY"), "Langfuse-Keys fehlen in der .env (Setup: siehe Docstring)"

SYSTEM_PROMPT = (
    "Du bist der Support-Bot eines Software-Unternehmens. "
    "Beantworte die Kundenanfrage freundlich, kurz und auf Deutsch, "
    "ordne sie einer Kategorie zu und entscheide, ob ein Mensch übernehmen muss."
)


class SupportAntwort(BaseModel):
    kategorie: Literal["technik", "abrechnung", "konto", "sonstiges"] = Field(
        description="Die am besten passende Kategorie der Kundenanfrage"
    )
    antwort: str = Field(description="Freundliche, kurze Antwort an den Kunden auf Deutsch")
    weiterleiten_an_mensch: bool = Field(
        description="True, wenn ein menschlicher Mitarbeiter übernehmen sollte"
    )


class SupportBot:
    """OpenAI-gestützter Kundensupport mit strukturierten Antworten."""

    def __init__(self, modell: str) -> None:
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        )
        self.modell = modell

    # NEU (2): Der Dekorator macht aus jedem Aufruf dieser Methode einen
    # Trace: die Anfrage wird als Input, die SupportAntwort als Output
    # erfasst (self wird automatisch ignoriert). Der LLM-Aufruf des
    # Wrapper-Clients hängt sich als Unter-Span darunter - so sieht man in
    # Langfuse die ganze Hierarchie: support-anfrage -> LLM-Generation.
    @observe(name="support-anfrage")
    def beantworte(self, anfrage: str) -> SupportAntwort:
        """Eine einzelne Kundenanfrage kategorisieren und beantworten."""
        response = self.client.beta.chat.completions.parse(
            model=self.modell,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": anfrage},
            ],
            response_format=SupportAntwort,
        )
        return response.choices[0].message.parsed

    def chat(self) -> None:
        """Kommandozeilen-Schleife: hilft, bis der Kunde 'exit' eintippt."""
        print("Support-Bot (Teil 2: mit Langfuse-Tracing) — tippe exit zum Beenden")

        # NEU (3): Diese Angaben gelten für alle Traces im with-Block.
        # Langfuse gruppiert danach (Session) und macht sie filterbar
        # (Nutzer, Tags, Metadaten).
        sitzung = "support-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with propagate_attributes(
            session_id=sitzung,
            user_id="kunde-42",
            tags=["support-bot", "tutorial"],
            metadata={"kanal": "cli", "bot_version": "1.0"},
        ):
            while True:
                anfrage = input("Kunde: ").strip()
                if anfrage.lower() in ["/quit", "/exit"]:
                    break

                ergebnis = self.beantworte(anfrage)
                mensch = "ja" if ergebnis.weiterleiten_an_mensch else "nein"
                print(f"  [Kategorie: {ergebnis.kategorie} | an Mensch weiterleiten: {mensch}]")
                print(f"Bot: {ergebnis.antwort}\n")

        # NEU (4): Langfuse sendet im Hintergrund - flush() stellt sicher,
        # dass beim Programmende alle Traces rausgehen.
        get_client().flush()
        print("Traces gesendet -> in der Langfuse-Weboberfläche unter 'Traces' ansehen.")


if __name__ == "__main__":
    bot = SupportBot(modell=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    bot.chat()

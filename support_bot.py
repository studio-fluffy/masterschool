"""Teil 1: Support-Bot OHNE Tracing.

Ein OpenAI-gestützter Kundensupport-Bot als Klasse, mit:
- strukturierten Antworten (Pydantic-Schema statt Freitext)
- automatischer Kategorisierung jeder Anfrage

Das Problem: KEINE SICHTBARKEIT! Sobald der Bot läuft, weiß niemand:
Welcher Prompt ging raus? Wie viele Tokens kostete der Aufruf? Wie lange
dauerte er? Warum wurde eine Anfrage falsch kategorisiert?
-> Die Lösung ist Tracing, siehe Teil 2: observability.py
"""

import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

SYSTEM_PROMPT = (
    "Du bist der Support-Bot eines Software-Unternehmens. "
    "Beantworte die Kundenanfrage freundlich, kurz und auf Deutsch, "
    "ordne sie einer Kategorie zu und entscheide, ob ein Mensch übernehmen muss."
)


# Strukturierte Antwort: Das Modell MUSS genau dieses Schema füllen.
# So kann der Code zuverlässig weiterarbeiten (z. B. Tickets routen).
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
        print("Support-Bot (Teil 1: ohne Tracing) — tippe exit zum Beenden")
        while True:
            anfrage = input("Kunde: ").strip()
            if anfrage.lower() in ["quit", "exit"]:
                break

            ergebnis = self.beantworte(anfrage)
            mensch = "ja" if ergebnis.weiterleiten_an_mensch else "nein"
            print(f"  [Kategorie: {ergebnis.kategorie} | an Mensch weiterleiten: {mensch}]")
            print(f"Bot: {ergebnis.antwort}\n")


if __name__ == "__main__":
    bot = SupportBot(modell=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    bot.chat()


import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


SYSTEM_PROMPT = """
Du bist ein freundlicher Kundensupport-Bot, der auf Benutzeranfragen reagiert.
Deine Aufgabe ist es, die Benutzeranfrage zu analysieren und eine strukturierte Antwort zu liefern.
Ordne Sie einer Kategorie zu und entscheid, ob ein Mensch übernehmen muss.   
"""

class SupportBotAnswer(BaseModel):
    category: Literal["technik", "abrechnung", "konto", "sonstiges"] = Field(
        ..., description="Die am besten passende Kategorie für die Benutzeranfrage."
    )
    answer: str = Field(description="Freundliche, kurze Antwort auf die Benutzeranfrage.")
    
    route_to_human:bool = Field(
        descritption= "True, wenn ein menschlicher Mitarbeiter die Anfrage übernehmen sollte, andernfalls False.")

class SupportBot:
    """Ai-gestützter Kundensupport mit Strukturierten Antworten."""

    def __init__(self, model: str) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_response(self, user_input: str) -> SupportBotAnswer:
        """Generiert eine Antwort des Support-Bots basierend auf der Benutzeranfrage."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            response_format=SupportBotAnswer,
        )
        return response.choices[0].message.parsed

    def chat(self) -> None:
        """Startet eine interaktive Chat-Sitzung mit dem Support-Bot."""
        print("Willkommen beim Kundensupport-Bot! Geben Sie '/exit' ein, um zu beenden.")
        while True:
            user_input = input("Ihre Anfrage: ").strip()
            if user_input.lower() == "/exit":
                print("Vielen Dank für die Nutzung des Kundensupport-Bots. Auf Wiedersehen!")
                break
            response = self.generate_response(user_input)
            print(f"\nKategorie: {response.category}")
            print(f"Antwort: {response.answer}")
            print(f"An menschlichen Mitarbeiter weiterleiten: {response.route_to_human}\n")


if __name__ == "__main__":
    bot = SupportBot(model=os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    bot.chat()
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


class Erklaerung(BaseModel):
    begriff: str = Field(description="Nur der Fachbegriff, z.B. 'Gradientenabstieg'")
    erklaerung: str = Field(description="2-3 einfache Sätze auf Deutsch, für Anfänger verständlich, keine Formeln")
    beispiel: str = Field(description="Eine konkrete Alltagsanalogie, kein Fachjargon")


system_prompt = """Du bist ein Tutor für maschinelles Lernen. Antworte auf Deutsch.
Fülle die Felder so:
- begriff: nur der Fachbegriff, ggf. mit englischem Original in Klammern
- erklaerung: 2-3 einfache Sätze, für Anfänger verständlich, keine Formeln
- beispiel: ein konkretes Alltagsbeispiel oder eine Analogie, kein Fachjargon"""

response = client.beta.chat.completions.parse(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Was ist Gradientenabstieg?"}
    ],
    response_format=Erklaerung
)

ergebnis = response.choices[0].message.parsed
print("Begriff:", ergebnis.begriff)
print("Erklärung:", ergebnis.erklaerung)
print("Beispiel:", ergebnis.beispiel)

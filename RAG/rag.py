"""RAG-Tutorial: Ein Kommandozeilen-Q&A-Programm über die Geschichte von RAG.

Das Programm baut die klassische RAG-Pipeline in fünf Schritten auf -
ohne Framework, nur mit der OpenAI-API, FAISS und NumPy direkt.
Einzige Ausnahme: das Chunking übernimmt der bewährte Splitter aus dem
kleinen Standalone-Paket langchain-text-splitters.

    1. LADEN      Wissensdokument (rag_geschichte.txt) einlesen
    2. CHUNKING   Dokument in überlappende Textblöcke zerlegen
    3. INDEXIEREN Blöcke einbetten und im FAISS-Vektorspeicher ablegen
    4. RETRIEVAL  Zur Frage die ähnlichsten Blöcke suchen
    5. GENERATION Chat-Modell beantwortet die Frage NUR mit diesen Blöcken

Schritte 1-3 laufen einmal beim Start (Indexierungsphase),
Schritte 4-5 laufen bei jeder Frage neu (Abfragephase).

Start:  python rag.py
Ende:   "exit" eingeben (oder Strg+C)
"""

import os
import sys

try:
    import faiss
    import numpy as np
    from dotenv import load_dotenv
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from openai import OpenAI
except ModuleNotFoundError as fehlendes_paket:
    sys.exit(
        f"Fehlendes Paket: {fehlendes_paket.name}\n"
        f"Dieses Programm läuft gerade mit dem Interpreter:\n"
        f"    {sys.executable}\n"
        f"Die Pakete sind aber vermutlich in einer anderen Umgebung installiert\n"
        f"(hier: miniconda base). Entweder in VS Code unten rechts den richtigen\n"
        f"Interpreter wählen oder die Pakete nachinstallieren mit:\n"
        f"    {sys.executable} -m pip install -r requirements.txt"
    )

# ---------------------------------------------------------------------------
# Schritt 0: Konfiguration
# ---------------------------------------------------------------------------
# Alle Einstellungen kommen aus der .env-Datei. Ohne BASE_URL wird direkt die
# OpenAI-API benutzt; mit BASE_URL jeder OpenAI-kompatible Server (z. B. ein
# lokaler Ollama-Server unter http://localhost:11434/v1).
load_dotenv()

BASE_URL = os.getenv("BASE_URL")  # None => echtes OpenAI
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

WISSENSDATEI = os.path.join(os.path.dirname(__file__), "rag_geschichte.txt")
TOP_K = 3            # Wie viele Textblöcke pro Frage als Kontext verwendet werden.
BLOCK_GROESSE = 800  # Zielgröße eines Blocks in Zeichen
UEBERLAPPUNG = 150   # so viele Zeichen teilen sich zwei Nachbarblöcke


def index_aufbauen(client: OpenAI) -> tuple[faiss.Index, list[str]]:
    """Indexierungsphase: Dokument laden, chunken, einbetten, in FAISS ablegen."""

    # -----------------------------------------------------------------------
    # Schritt 1: Wissensdokument laden
    # -----------------------------------------------------------------------
    # Hier eine einfache Textdatei. In echten Projekten kommen an dieser
    # Stelle Parser für PDF, HTML usw. zum Einsatz.
    with open(WISSENSDATEI, encoding="utf-8") as f:
        text = f.read()

    # -----------------------------------------------------------------------
    # Schritt 2: Chunking - das Dokument in Blöcke zerlegen
    # -----------------------------------------------------------------------
    # Warum? Ein Embedding pro GANZEM Dokument wäre zu unscharf, und das
    # Chat-Modell soll später nur die relevanten Ausschnitte sehen.
    # Der RecursiveCharacterTextSplitter versucht, an sinnvollen Grenzen zu
    # trennen (erst Absätze, dann Sätze, dann Wörter). Die Überlappung sorgt
    # dafür, dass kein Zusammenhang genau an einer Schnittkante verloren geht.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BLOCK_GROESSE,
        chunk_overlap=UEBERLAPPUNG,
    )
    bloecke = splitter.split_text(text)
    print(f"Dokument in {len(bloecke)} Blöcke zerlegt.")

    # -----------------------------------------------------------------------
    # Schritt 3: Embeddings berechnen und im FAISS-Index speichern
    # -----------------------------------------------------------------------
    # Ein einziger API-Aufruf bettet alle Blöcke auf einmal ein. Das
    # Embedding-Modell wandelt jeden Block in einen Vektor um; semantisch
    # ähnliche Texte bekommen ähnliche Vektoren.
    antwort = client.embeddings.create(model=EMBEDDING_MODEL, input=bloecke)
    vektoren = np.array([d.embedding for d in antwort.data], dtype="float32")

    # IndexFlatL2 ist der einfachste FAISS-Index: Er hält alle Vektoren
    # unkomprimiert im Arbeitsspeicher und vergleicht per L2-Distanz.
    # FAISS kennt nur Vektoren und ihre Position (0, 1, 2, ...) - die Texte
    # selbst merken wir uns daneben in der Liste `bloecke`.
    index = faiss.IndexFlatL2(vektoren.shape[1])
    index.add(vektoren)
    print(f"FAISS-Index mit {index.ntotal} Vektoren aufgebaut.\n")
    return index, bloecke


def frage_beantworten(
    index: faiss.Index, bloecke: list[str], client: OpenAI, frage: str
) -> str:
    """Abfragephase: passende Blöcke suchen und daraus eine Antwort erzeugen."""

    # -----------------------------------------------------------------------
    # Schritt 4: Retrieval - die ähnlichsten Blöcke zur Frage finden
    # -----------------------------------------------------------------------
    # Die Frage wird mit demselben Embedding-Modell eingebettet, FAISS liefert
    # die Positionen der TOP_K nächsten Vektoren plus deren Distanzen.
    # Der Score ist eine L2-Distanz: KLEINER bedeutet ÄHNLICHER.
    einbettung = client.embeddings.create(model=EMBEDDING_MODEL, input=[frage])
    frage_vektor = np.array([einbettung.data[0].embedding], dtype="float32")
    distanzen, positionen = index.search(frage_vektor, min(TOP_K, index.ntotal))

    print("  Gefundene Blöcke (Score: kleiner = ähnlicher):")
    treffer = []
    for score, position in zip(distanzen[0], positionen[0]):
        treffer.append(bloecke[position])
        vorschau = bloecke[position].replace("\n", " ")[:80]
        print(f"    [{score:.3f}] {vorschau}...")

    kontext = "\n\n---\n\n".join(treffer)

    # -----------------------------------------------------------------------
    # Schritt 5: Generation - das Chat-Modell antwortet nur aus dem Kontext
    # -----------------------------------------------------------------------
    # Eine Nachrichtenliste und ein einziger API-Aufruf: Die System-Nachricht
    # legt die Spielregeln fest, die User-Nachricht enthält Kontext und Frage
    # (per f-String eingesetzt). Der Antworttext steckt in choices[0].
    antwort = client.chat.completions.create(
        model=CHAT_MODEL,
        # temperature=0: möglichst faktentreue, reproduzierbare Antworten -
        # genau das will man bei Q&A über eine feste Wissensbasis.
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein Tutor für die Geschichte der RAG-Technik. "
                    "Beantworte die Frage AUSSCHLIESSLICH mit dem gelieferten "
                    "Kontext. Steht die Antwort nicht im Kontext, sage das "
                    "offen. Antworte auf Deutsch, kurz und präzise."
                ),
            },
            {"role": "user", "content": f"Kontext:\n{kontext}\n\nFrage: {frage}"},
        ],
    )
    return antwort.choices[0].message.content


def main() -> None:
    print("=" * 60)
    print("RAG-Tutorial: Frag mich zur Geschichte von RAG!")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Fehler: OPENAI_API_KEY fehlt (in der .env-Datei setzen).")

    # Ein Client für alles: Embeddings UND Chat laufen über denselben Server.
    # Den API-Schlüssel liest der Client selbst aus OPENAI_API_KEY.
    client = OpenAI(base_url=BASE_URL)

    index, bloecke = index_aufbauen(client)

    print(f"Chat-Modell: {CHAT_MODEL} | Embedding-Modell: {EMBEDDING_MODEL}")
    print('Beispiel: "Wer hat den Begriff RAG geprägt?" - beenden mit "exit"\n')

    while True:
        try:
            frage = input("Deine Frage> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTschüss!")
            break

        if not frage:
            continue
        if frage.lower() in ("exit", "quit", "q"):
            print("Tschüss!")
            break

        try:
            antwort = frage_beantworten(index, bloecke, client, frage)
        except Exception as fehler:  # z. B. Server nicht erreichbar
            print(f"  Fehler bei der Anfrage: {fehler}\n")
            continue

        print(f"\nAntwort: {antwort}\n")


if __name__ == "__main__":
    main()

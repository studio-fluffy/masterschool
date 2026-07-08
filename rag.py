"""RAG-Tutorial: Ein Kommandozeilen-Q&A-Programm über die Geschichte von RAG.

Das Programm baut die klassische RAG-Pipeline in fünf Schritten auf:

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
    from dotenv import load_dotenv

    # Beim Import erscheint eine Warnung, dass langchain-community abgekündigt
    # wird. Die FAISS-Integration hat aber (Stand Juli 2026) noch kein
    # funktionierendes Nachfolgepaket - die Warnung ist hier also unkritisch.
    from langchain_community.vectorstores import FAISS
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
TOP_K = 3  # Wie viele Textblöcke pro Frage als Kontext verwendet werden.


def index_aufbauen() -> FAISS:
    """Indexierungsphase: Dokument laden, chunken, einbetten, in FAISS ablegen."""

    # -----------------------------------------------------------------------
    # Schritt 1: Wissensdokument laden
    # -----------------------------------------------------------------------
    # Hier eine einfache Textdatei. In echten Projekten übernehmen das die
    # Document-Loader von LangChain (PDF, HTML, Notion, ...).
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
        chunk_size=800,     # Zielgröße eines Blocks in Zeichen
        chunk_overlap=150,  # so viele Zeichen teilen sich zwei Nachbarblöcke
    )
    chunks = splitter.create_documents([text])
    print(f"Dokument in {len(chunks)} Blöcke zerlegt.")

    # -----------------------------------------------------------------------
    # Schritt 3: Embeddings berechnen und im FAISS-Index speichern
    # -----------------------------------------------------------------------
    # Das Embedding-Modell wandelt jeden Block in einen Vektor um; semantisch
    # ähnliche Texte bekommen ähnliche Vektoren. FAISS hält alle Vektoren im
    # Arbeitsspeicher und kann darin extrem schnell nach Nachbarn suchen.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        # OpenAI-fremde Server (z. B. Ollama) erwarten rohen Text statt der
        # vorab berechneten Token-IDs, die der Client sonst schicken würde.
        check_embedding_ctx_length=False,
    )
    vektorspeicher = FAISS.from_documents(chunks, embeddings)
    print(f"FAISS-Index mit {vektorspeicher.index.ntotal} Vektoren aufgebaut.\n")
    return vektorspeicher


def frage_beantworten(vektorspeicher: FAISS, llm: ChatOpenAI, frage: str) -> str:
    """Abfragephase: passende Blöcke suchen und daraus eine Antwort erzeugen."""

    # -----------------------------------------------------------------------
    # Schritt 4: Retrieval - die ähnlichsten Blöcke zur Frage finden
    # -----------------------------------------------------------------------
    # Die Frage wird mit demselben Embedding-Modell eingebettet, FAISS liefert
    # die TOP_K nächsten Blöcke. Der Score ist eine L2-Distanz:
    # KLEINER bedeutet ÄHNLICHER.
    treffer = vektorspeicher.similarity_search_with_score(frage, k=TOP_K)

    print("  Gefundene Blöcke (Score: kleiner = ähnlicher):")
    for dokument, score in treffer:
        vorschau = dokument.page_content.replace("\n", " ")[:80]
        print(f"    [{score:.3f}] {vorschau}...")

    kontext = "\n\n---\n\n".join(dokument.page_content for dokument, _ in treffer)

    # -----------------------------------------------------------------------
    # Schritt 5: Generation - das Chat-Modell antwortet nur aus dem Kontext
    # -----------------------------------------------------------------------
    # Das Prompt-Template hat zwei Platzhalter (kontext, frage). Mit dem
    # |-Operator (LangChain Expression Language) werden Prompt, Modell und
    # Ausgabe-Parser zu einer Kette verbunden: Die Eingaben füllen das
    # Template, das Ergebnis geht ans Modell, dessen Antwort wird zu einem
    # einfachen String.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Du bist ein Tutor für die Geschichte der RAG-Technik. "
            "Beantworte die Frage AUSSCHLIESSLICH mit dem gelieferten Kontext. "
            "Steht die Antwort nicht im Kontext, sage das offen. "
            "Antworte auf Deutsch, kurz und präzise.",
        ),
        ("human", "Kontext:\n{kontext}\n\nFrage: {frage}"),
    ])
    kette = prompt | llm | StrOutputParser()
    return kette.invoke({"kontext": kontext, "frage": frage})


def main() -> None:
    print("=" * 60)
    print("RAG-Tutorial: Frag mich zur Geschichte von RAG!")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Fehler: OPENAI_API_KEY fehlt (in der .env-Datei setzen).")

    vektorspeicher = index_aufbauen()

    # temperature=0: möglichst faktentreue, reproduzierbare Antworten -
    # genau das will man bei Q&A über eine feste Wissensbasis.
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=BASE_URL, temperature=0)

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
            antwort = frage_beantworten(vektorspeicher, llm, frage)
        except Exception as fehler:  # z. B. Server nicht erreichbar
            print(f"  Fehler bei der Anfrage: {fehler}\n")
            continue

        print(f"\nAntwort: {antwort}\n")


if __name__ == "__main__":
    main()

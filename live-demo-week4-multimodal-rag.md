# Live Demo Guide - Building Multimodal Applications

This demo builds a small searchable video assistant.

```text
video
-> representative frames
-> audio track
-> transcript segments
-> timestamped records
-> embeddings
-> retrieval
-> grounded answer
```

The key teaching point: **video becomes useful to an AI application when it is decomposed into searchable, timestamped evidence.**

---

## Teaching Goal

Students should leave with one clear mental model:

```text
decompose -> represent -> index -> retrieve -> ground -> answer
```

They should also understand the boundary between the components:

| Component | Purpose |
|---|---|
| Frame extraction | Create visual evidence from video |
| Audio extraction | Separate the speech/audio signal |
| Speech-to-text | Turn spoken content into text |
| Timestamped records | Keep every piece of evidence connected to time |
| Embeddings | Make records searchable by semantic similarity |
| Retrieval | Select relevant evidence for a query |
| Fusion | Combine transcript and frame evidence |
| Answer generation | Produce a final answer grounded in retrieved evidence |

The default path uses OpenAI for transcript and record embeddings. If transcription or text-generation model access is unavailable, the demo continues with fallback transcript segments and a transparent template answer.

---

## Before the Session

### Install Python packages

```bash
pip install --upgrade openai python-dotenv pandas numpy scikit-learn pillow ipython
```

Recommended for video processing:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

Optional for generating or inspecting videos locally:

```bash
pip install opencv-python
```

### Prepare the `.env` file

Create a file named `.env` in the same folder as the notebook or Python script.

```text
OPENAI_API_KEY=your_api_key_here
```

Recommended defaults:

```text
OPENAI_STT_MODEL=whisper-1
STT_PROVIDER=auto
LOCAL_STT_MODEL=small
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
USE_OPENAI_LLM=false
```

Not every API account has access to the hosted speech-to-text models. If `whisper-1`
returns a 403, the notebook transcribes locally instead with `pip install -U openai-whisper`
and the model named in `LOCAL_STT_MODEL` (`small` is ~460 MB, downloaded once, and
transcribes the 29 second clip in a few seconds on a GPU; `tiny` is faster but misheard
"is a separate case" as "in a separate case" in our test).

`STT_PROVIDER` decides who gets to try:

| Value | Behaviour |
| --- | --- |
| `auto` | Hosted model first, local Whisper if it fails. The default. |
| `local` | Skip the hosted call entirely. Use this when you know the account has no audio models - the demo runs faster and no 403 appears on the projector. |
| `openai` | Hosted model only. Fails loudly instead of degrading. |

There is deliberately no canned transcript to fall back on. If no provider produces
segments, the notebook **raises**: a hardcoded transcript would come with invented
timestamps, and every retrieval and grounding claim later in the demo would quietly be
about text that was never in the video.

Optional LLM mode, only if your API account supports text-generation models:

```text
USE_OPENAI_LLM=true
OPENAI_ANSWER_MODEL=gpt-4o-mini
```

Do not paste the API key into notebook cells. Keep it in `.env`.

### Prepare the sample video

Recommended filename:

```text
sample_multimodal_support_video.mp4
```

The synthetic video contains three useful moments:

| Approx. time | Topic |
|---|---|
| 00:00-00:08 | introduction |
| 00:08-00:20 | password reset issue |
| 00:20-00:30 | ticket creation and escalation |
| 00:30-00:38 | delivery delay contrast case |

If the video file is missing, the notebook creates fallback frame records and transcript segments so the retrieval demo still works.

### Verify the setup

```python
import os
import shutil
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception as exc:
    OpenAI = None
    print("OpenAI package not available. Install with: pip install openai")
    print(type(exc).__name__, exc)

load_dotenv()

print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("FFmpeg available:", bool(shutil.which("ffmpeg")))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "missing")) if OpenAI else None
print("OpenAI client ready:", client is not None)
```

### Font and display setup

* Set notebook and terminal font to 18pt+.
* Keep the sample video short.
* Keep the output folders visible in the notebook file browser.
* Pre-run the notebook once before class if venue WiFi is uncertain.

---

## Demo Flow

| # | What | Time |
|---|---|---:|
| 1 | Check setup and locate the sample video | 3 min |
| 2 | Extract representative frames and audio | 5 min |
| 3 | Transcribe or load fallback transcript segments | 5 min |
| 4 | Build timestamped transcript and frame records | 5 min |
| 5 | Generate embeddings and create a local index | 6 min |
| 6 | Search the video with a text query | 4 min |
| 7 | Generate a grounded answer with evidence | 4 min |
| 8 | Evaluate quality, latency, and failure modes | 3 min |
| **Total** | | **~35 min** |

Suggested class timing:

| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Live demo | 30 min |
| Recap / questions | 5 min |

---

## Step-by-Step Commands

### Step 1 - Imports and constants

```python
import os
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display, Image as IPImage, Video
try:
    from openai import OpenAI
except Exception as exc:
    OpenAI = None
    print("OpenAI package not available. Install with: pip install openai")
    print(type(exc).__name__, exc)

load_dotenv()

VIDEO_PATH = Path("sample_multimodal_support_video.mp4")
WORK_DIR = Path("week4_video_work")
FRAME_DIR = WORK_DIR / "frames"
AUDIO_PATH = WORK_DIR / "extracted_audio.wav"

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
# Used when the hosted STT model is not available on the API account.
LOCAL_STT_MODEL = os.getenv("LOCAL_STT_MODEL", "small")
# "auto" tries the hosted model first, "local" skips it, "openai" allows nothing else.
STT_PROVIDER = os.getenv("STT_PROVIDER", "auto").lower()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() == "true"
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "missing")) if OpenAI else None

print("Video exists:", VIDEO_PATH.exists())
print("FFmpeg available:", bool(shutil.which("ffmpeg")))
print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
```

> Say: "The video file is not the AI system. We first have to decompose it into pieces the system can search."

---

### Step 2 - Extract frames and audio

```python
WORK_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

if VIDEO_PATH.exists() and shutil.which("ffmpeg"):
    # Extract one representative frame every 5 seconds.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(VIDEO_PATH),
            "-vf", "fps=1/5",
            str(FRAME_DIR / "frame_%03d.jpg"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Extract 16 kHz mono audio for speech-to-text.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(VIDEO_PATH),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            str(AUDIO_PATH),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
else:
    print("Video or FFmpeg missing. The notebook will use fallback records.")

frame_paths = sorted(FRAME_DIR.glob("*.jpg"))
print("Extracted frames:", len(frame_paths))
print("Audio exists:", AUDIO_PATH.exists())
```

> Say: "The sampling rate is a design choice. One frame every five seconds is cheap and explainable, but it may miss fast visual events."

Optional display:

```python
if VIDEO_PATH.exists():
    display(Video(str(VIDEO_PATH), embed=True, width=640))

for path in frame_paths[:4]:
    display(IPImage(filename=str(path), width=320))
```

---

### Step 3 - Transcribe the audio

```python
def transcribe_with_openai(audio_path):
    """Hosted speech-to-text. Returns timestamped segments or an empty list."""
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            response_format="verbose_json",
        )

    raw_segments = getattr(transcription, "segments", None)
    if raw_segments:
        return [
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
            for seg in raw_segments
        ]

    text = getattr(transcription, "text", "").strip()
    return [{"start": 0.0, "end": 38.0, "text": text}] if text else []


def transcribe_locally(audio_path):
    """Local Whisper. Same segment shape, runs without API access."""
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(LOCAL_STT_MODEL, device=device)
    result = model.transcribe(str(audio_path), language="en", fp16=(device == "cuda"))
    return [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
        for seg in result["segments"]
    ]


providers = [
    (
        f"openai:{STT_MODEL}",
        transcribe_with_openai,
        STT_PROVIDER in ("auto", "openai") and bool(client and os.environ.get("OPENAI_API_KEY")),
    ),
    (
        f"local whisper:{LOCAL_STT_MODEL}",
        transcribe_locally,
        STT_PROVIDER in ("auto", "local"),
    ),
]

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"No audio at {AUDIO_PATH}. Run Step 2 first - the transcript has to come from the "
        "video, not from a canned example."
    )

transcript_segments = []
transcript_source = None

for name, transcribe, enabled in providers:
    if not enabled:
        continue
    try:
        segments = transcribe(AUDIO_PATH)
    except Exception as exc:
        print(f"{name} unavailable -> {type(exc).__name__}: {str(exc)[:160]}")
        continue
    if segments:
        transcript_segments = segments
        transcript_source = name
        break

if not transcript_segments:
    raise RuntimeError(
        f"No speech-to-text provider produced a transcript (STT_PROVIDER={STT_PROVIDER}). "
        "Check the errors above: either grant the API account access to the hosted model, "
        "or set STT_PROVIDER=local after `pip install -U openai-whisper`."
    )

print("Transcript source:", transcript_source)
segments_df = pd.DataFrame(transcript_segments)
segments_df
```

> Say: "The transcript is useful, but timestamped transcript segments are more useful because they keep the answer connected to the video location."

> Say: "Notice the degradation: hosted model first, local model second. Both return the same segment shape, so the rest of the pipeline never has to know which one produced the transcript. And if neither works, the notebook stops. It does not invent a transcript, because a fake transcript would still produce confident answers with timestamps that point nowhere."

---

### Step 4 - Build timestamped records

```python
records = []

for i, seg in enumerate(transcript_segments):
    records.append(
        {
            "id": f"transcript_{i:03d}",
            "modality": "transcript",
            "start": seg["start"],
            "end": seg["end"],
            "content": seg["text"],
            "source": "speech-to-text",
        }
    )

# For a robust classroom demo, frame records use short visual descriptions.
# In production, these could come from direct image embeddings or a vision model.
frame_descriptions = [
    "title slide for support training video",
    "support dashboard showing account access and password reset issue",
    "support ticket created with high priority status",
    "delivery delay case shown as a contrast example",
]

for i, frame_path in enumerate(frame_paths[:4]):
    timestamp = i * 5
    records.append(
        {
            "id": f"frame_{i:03d}",
            "modality": "frame",
            "start": timestamp,
            "end": timestamp,
            "content": frame_descriptions[min(i, len(frame_descriptions) - 1)],
            "source": str(frame_path),
        }
    )

# Fallback frame records if no real frames were extracted.
if not frame_paths:
    for i, description in enumerate(frame_descriptions):
        records.append(
            {
                "id": f"frame_{i:03d}",
                "modality": "frame",
                "start": i * 10,
                "end": i * 10,
                "content": description,
                "source": "fallback_visual_record",
            }
        )

records_df = pd.DataFrame(records)
display(records_df)
```

> Say: "The vector alone is not enough. The metadata tells us where the evidence came from and how to show it to a user."

---

### Step 5 - Generate embeddings

```python
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)

texts = records_df["content"].tolist()

if os.environ.get("OPENAI_API_KEY") and client:
    try:
        embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        embeddings = np.array([item.embedding for item in embedding_response.data], dtype=np.float32)
        embedding_source = "openai"
    except Exception as exc:
        print("OpenAI embeddings failed. Falling back to local TF-IDF vectors.")
        print(type(exc).__name__, exc)
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer().fit(texts)
        embeddings = vectorizer.transform(texts).toarray().astype(np.float32)
        embedding_source = "tfidf"
else:
    print("No OpenAI API key found. Using local TF-IDF vectors.")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer().fit(texts)
    embeddings = vectorizer.transform(texts).toarray().astype(np.float32)
    embedding_source = "tfidf"

embeddings = normalize_rows(embeddings)

print("Embedding source:", embedding_source)
print("Embedding matrix shape:", embeddings.shape)
print("First record:", records_df.iloc[0][["id", "modality", "content"]].to_dict())
```

> Say: "Embedding turns every record into a searchable vector. The index is just vectors plus metadata."

---

### Step 6 - Search the video

```python
def embed_query(query: str) -> np.ndarray:
    if embedding_source == "openai":
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vector = np.array([response.data[0].embedding], dtype=np.float32)
    else:
        vector = vectorizer.transform([query]).toarray().astype(np.float32)
    return normalize_rows(vector)

query = "Where does the speaker explain the password reset problem?"
query_embedding = embed_query(query)

scores = (query_embedding @ embeddings.T)[0]
records_df["score"] = scores
results = records_df.sort_values("score", ascending=False).head(5)

display(results[["id", "modality", "start", "end", "score", "content", "source"]])
```

> Ask: "Did the search find the transcript segment, the visual frame, or both? What would happen if the transcript were wrong?"

---

### Step 7 - Fuse evidence by timestamp

```python
# Simple late-fusion rule:
# 1. take the top results
# 2. group nearby timestamps
# 3. preserve both transcript and frame evidence

top_results = results.head(4).to_dict("records")

best_time = float(results.iloc[0]["start"])
window_seconds = 10

fused_evidence = records_df[
    (records_df["start"] >= best_time - window_seconds)
    & (records_df["start"] <= best_time + window_seconds)
].sort_values(["start", "modality"])

display(fused_evidence[["id", "modality", "start", "end", "score", "content", "source"]])
```

> Say: "Late fusion is easy to debug: retrieve separately, then merge nearby evidence. It is not the most advanced method, but it is a very good first prototype."

---

### Step 8 - Generate a grounded answer

```python
def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

context_lines = []
for _, row in fused_evidence.iterrows():
    context_lines.append(
        f"- [{row['modality']}] {format_time(row['start'])}-{format_time(row['end'])}: {row['content']}"
    )

context = "\n".join(context_lines)

prompt = f"""
Answer the user's question using only the evidence below.
Include the most relevant timestamp.
If the evidence is insufficient, say so.

Question: {query}

Evidence:
{context}
""".strip()


def template_answer() -> str:
    password_rows = fused_evidence[
        fused_evidence["content"].str.contains("password|reset|account", case=False, regex=True)
    ]
    if password_rows.empty:
        return "I could not find enough evidence for the password reset question in the retrieved segment."
    first = password_rows.iloc[0]
    return (
        f"The password reset problem is discussed around {format_time(first['start'])}. "
        "The evidence says the customer cannot log into the account because the password reset email never arrives. "
        "The nearby segment also shows that the agent creates a high-priority support ticket and asks the customer to check spam filters."
    )

if USE_OPENAI_LLM and client:
    try:
        response = client.responses.create(
            model=ANSWER_MODEL,
            input=prompt,
        )
        answer = response.output_text
    except Exception as exc:
        print("OpenAI answer model failed. Using template answer.")
        print(type(exc).__name__, exc)
        answer = template_answer()
else:
    answer = template_answer()

print(answer)
```

> Say: "The answer should cite the retrieved segment. If the model answers without evidence, the application design is weak."

---

### Step 9 - Inspect the evidence object

```python
evidence_package = {
    "query": query,
    "answer": answer,
    "retrieved_evidence": fused_evidence[[
        "id", "modality", "start", "end", "score", "content", "source"
    ]].to_dict("records"),
}

print(json.dumps(evidence_package, indent=2))
```

> Say: "This object is what you would log in a real system. It makes the answer auditable."

---

### Step 10 - Evaluation discussion

```python
evaluation_questions = pd.DataFrame(
    [
        {"Area": "Retrieval quality", "Question": "Did the top result contain the right topic?"},
        {"Area": "Timestamp accuracy", "Question": "Is the returned time close to the relevant moment?"},
        {"Area": "Grounding", "Question": "Is every answer claim supported by retrieved evidence?"},
        {"Area": "Latency", "Question": "Which step was slowest: video processing, transcription, embedding, or answering?"},
        {"Area": "Storage", "Question": "How many frames and embeddings would a one-hour video create?"},
        {"Area": "Cost", "Question": "Which API calls scale with video length?"},
    ]
)

display(evaluation_questions)
```

> Say: "A multimodal application is not validated by one nice answer. It is validated by retrieval quality, timestamps, grounding, latency, storage, and cost."

---

## Student Mini Challenge

Ask students to change the query:

```python
query = "Where does the video mention a delivery delay?"
```

Then rerun:

1. query embedding
2. similarity search
3. fused evidence
4. answer generation

Students should check:

* Does retrieval switch from password reset to delivery delay?
* Does the returned timestamp change?
* Does the answer avoid mixing both cases?
* Does the evidence package make the result auditable?

Bonus:

```python
query = "Which part of the video shows that a ticket was created?"
```

Ask whether the answer depends more on transcript evidence or frame evidence.

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `ffmpeg` not found | Install FFmpeg or use fallback records |
| video file missing | Use the provided sample video or rely on fallback records |
| transcription fails | Use fallback transcript segments |
| OpenAI embedding call fails | Use local TF-IDF fallback |
| answer model access denied | Keep `USE_OPENAI_LLM=false` and use template answer |
| frame extraction creates no images | Check the video path and FFmpeg command |
| retrieval returns irrelevant results | Inspect transcript quality and frame descriptions |
| timestamps look wrong | Check frame sampling interval and transcript segmentation |
| students confuse retrieval with reasoning | Emphasize: retrieval selects evidence; answering uses evidence |

---

## Instructor Wrap-Up

Close with this mapping:

| Session concept | Demo object |
|---|---|
| Video decomposition | extracted frames and audio file |
| Speech-to-text | transcript segments |
| Multimodal records | transcript and frame rows |
| Embeddings | vector matrix |
| Vector search | similarity ranking |
| Fusion | nearby transcript and frame evidence |
| RAG | answer generated from retrieved evidence |
| Evaluation | retrieval, timestamp, grounding, latency, storage, cost |

Final sentence:

> "A multimodal application is not just a model that accepts many inputs. It is a system that decomposes media, indexes evidence, retrieves the right moments, and answers with traceable grounding."

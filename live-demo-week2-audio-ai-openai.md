# Live Demo Guide - Audio AI with OpenAI APIs

This demo uses a modular audio pipeline:

```text
Audio file
-> preprocessing
-> OpenAI speech-to-text
-> transcript embedding
-> retrieval over past calls
-> OpenAI LLM answer
-> OpenAI text-to-speech
-> spoken output
```

The key teaching point: **audio does not become an answer in one step.** The system first turns speech into text, turns text into embeddings for retrieval, uses an LLM to generate an answer from evidence, and turns the answer back into audio.

---

## Teaching Goal

Students should leave with one clear mental model:

```text
speech -> transcript -> embedding -> retrieval -> grounded answer -> spoken response
```

They should also understand the boundary between the steps:

| Step | Purpose |
|---|---|
| Speech-to-text | Convert speech into a transcript |
| Embeddings | Make transcripts searchable |
| Retrieval | Find similar prior calls or context |
| LLM | Summarize, classify, and suggest action |
| Text-to-speech | Convert the final answer into audio |

---

## Before the Session

### Install Python packages

```bash
pip install --upgrade openai python-dotenv pandas numpy scikit-learn librosa soundfile ipython
```

### Install FFmpeg

FFmpeg is still useful even when the AI models run through OpenAI. It handles audio conversion, resampling, and extracting audio from video.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

### Prepare the `.env` file

Create a file named `.env` in the same folder as the notebook or script:

```env
OPENAI_API_KEY=your_api_key_here
```

Optional model overrides:

```env
OPENAI_ANSWER_MODEL=gpt-5.6-luna
OPENAI_STT_MODEL=gpt-transcribe
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=coral
```

Do not commit `.env` to Git and do not paste the API key into the notebook.

### Verify the setup

```python
import os
import shutil
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("FFmpeg available:", bool(shutil.which("ffmpeg")))

client = OpenAI()
print("OpenAI client ready")
```

Expected:

```text
OpenAI key set: True
FFmpeg available: True
OpenAI client ready
```

### Prepare one short audio file

Bring a short WAV, M4A, or MP3 file named:

```text
sample_support_call.wav
```

Recommended spoken content:

```text
Hi, I cannot log into my account. The password reset email never arrives, and I need access today.
```

Keep it under 15 seconds. A short, clean recording makes the demo more predictable and keeps API cost low.

### Font and audio setup

* Set notebook and terminal font to 18pt+.
* Test speaker volume before class.
* Keep `sample_support_call.wav` in the same folder as the notebook.
* Keep `.env` in the same folder as the notebook.

---

## Demo Flow

| # | What | Time |
|---|---|---:|
| 1 | Load and preprocess a short audio file | 4 min |
| 2 | Transcribe the audio with OpenAI speech-to-text | 5 min |
| 3 | Build a small past-call dataset | 3 min |
| 4 | Generate OpenAI embeddings and retrieve similar calls | 7 min |
| 5 | Send transcript + retrieved context to an OpenAI LLM | 5 min |
| 6 | Generate spoken output with OpenAI text-to-speech | 5 min |
| 7 | Explain failure modes and design tradeoffs | 1 min |
| **Total** | | **~30 min** |

Suggested class timing:

| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Live demo | 30 min |
| Recap / questions | 5 min |

---

## Step-by-Step Commands

### Step 1 - Imports, `.env`, and constants

```python
from pathlib import Path
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Audio, display
from openai import OpenAI

load_dotenv()
client = OpenAI()

INPUT_AUDIO = Path("sample_support_call.wav")
CLEAN_AUDIO = Path("sample_16k_mono.wav")
TTS_OUTPUT = Path("answer_openai.wav")

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-transcribe")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-5.6-luna")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "coral")

FALLBACK_TRANSCRIPT = (
    "Hi, I cannot log into my account. "
    "The password reset email never arrives, and I need access today."
)

print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("FFmpeg available:", bool(shutil.which("ffmpeg")))
print("Input audio exists:", INPUT_AUDIO.exists())
print("Answer model:", ANSWER_MODEL)
```

> Say: "In the local version we used separate local models. In this version, the model calls go through OpenAI: one call for transcription, one for embeddings, one for answer generation, and one for speech generation."

---

### Step 2 - Load and preprocess the audio

Use FFmpeg when available. It gives a clear, production-like audio conversion step.

```python
transcript = None

if INPUT_AUDIO.exists():
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(INPUT_AUDIO),
            "-ar", "16000",
            "-ac", "1",
            str(CLEAN_AUDIO),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print("Saved", CLEAN_AUDIO, "with FFmpeg")
    else:
        audio, sample_rate = librosa.load(INPUT_AUDIO, sr=16000, mono=True)
        sf.write(CLEAN_AUDIO, audio, sample_rate)
        print("Saved", CLEAN_AUDIO, "with librosa/soundfile fallback")

    audio, sample_rate = librosa.load(CLEAN_AUDIO, sr=16000, mono=True)
    duration = len(audio) / sample_rate

    print("Sample rate:", sample_rate)
    print("Duration:", round(duration, 2), "seconds")
    print("Samples:", audio.shape)

    display(Audio(str(CLEAN_AUDIO), autoplay=False))
else:
    transcript = FALLBACK_TRANSCRIPT
    print("Missing sample_support_call.wav. Using fallback transcript:")
    print(transcript)
```

> Say: "OpenAI can accept common audio formats directly, but preprocessing keeps the teaching pipeline explicit and makes the input predictable. FFmpeg is the utility layer that normalizes messy real-world audio before model calls."

---

### Step 3 - Transcribe with OpenAI speech-to-text

```python
if transcript is None:
    with CLEAN_AUDIO.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            prompt=(
                "A short customer support call about account login, "
                "password reset emails, and urgent access."
            ),
        )

    transcript = transcription.text.strip()

print(transcript)
```

Expected transcript should be close to:

```text
Hi, I cannot log into my account. The password reset email never arrives, and I need access today.
```

> Say: "Speech-to-text is not reasoning. It converts spoken language into written text. Everything downstream depends on this transcript being good enough."

---

### Step 4 - Build a small retrieval dataset

```python
past_calls = [
    {
        "id": "call_101",
        "topic": "billing",
        "transcript": "The customer was charged twice for the same invoice and wants a refund.",
    },
    {
        "id": "call_102",
        "topic": "password_reset",
        "transcript": "The user cannot log in because the password reset email is not arriving.",
    },
    {
        "id": "call_103",
        "topic": "delivery_delay",
        "transcript": "The customer asks why their package is delayed and wants a new delivery date.",
    },
    {
        "id": "call_104",
        "topic": "audio_issue",
        "transcript": "The microphone is not detected during video calls after a laptop update.",
    },
]

pd.DataFrame(past_calls)
```

> Say: "These are previous calls. In a production system, each row would link to the original audio file, timestamps, speaker metadata, ticket outcome, and permission metadata."

---

### Step 5 - Generate OpenAI embeddings and retrieve similar calls

```python
def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def embed_texts(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return normalize_rows(vectors)


past_texts = [item["transcript"] for item in past_calls]

past_embeddings = embed_texts(past_texts)
query_embedding = embed_texts([transcript])

scores = cosine_similarity(query_embedding, past_embeddings)[0]

results = pd.DataFrame({
    "id": [item["id"] for item in past_calls],
    "topic": [item["topic"] for item in past_calls],
    "transcript": past_texts,
    "score": scores,
}).sort_values("score", ascending=False)

results
```

Expected behavior: `password_reset` should be the top or near-top result.

> Say: "We are not embedding the raw audio here. We are embedding the transcript. For spoken business content, this is usually the most robust first approach."

---

### Step 6 - Send transcript and retrieved context to an OpenAI LLM

```python
retrieved_context = results.head(2).to_dict(orient="records")

user_prompt = (
    "New call transcript:\n"
    f"{transcript}\n\n"
    "Similar past calls:\n"
    f"{retrieved_context}\n\n"
    "Task:\n"
    "1. Summarize the user issue in one sentence.\n"
    "2. Identify the likely intent.\n"
    "3. Suggest the next best support action.\n"
    "Keep the answer short and clear."
)

response = client.responses.create(
    model=ANSWER_MODEL,
    input=[
        {
            "role": "developer",
            "content": (
                "You are a concise support assistant. "
                "Use only the transcript and retrieved context. "
                "Do not invent account details, ticket IDs, or policies."
            ),
        },
        {"role": "user", "content": user_prompt},
    ],
)

answer = response.output_text.strip()
print(answer)
```

> Say: "The LLM is not listening to audio here. It receives the transcript and the retrieved context. That makes the system easier to inspect and debug."

---

### Step 7 - Convert the generated answer back into speech

Clean the generated answer before sending it to TTS:

```python
def clean_for_tts(text: str) -> str:
    text = re.sub(r"[*_`#>-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


spoken_answer = clean_for_tts(answer)

if len(spoken_answer) > 450:
    spoken_answer = spoken_answer[:450] + "."

print(spoken_answer)
```

Generate the audio file:

```python
with client.audio.speech.with_streaming_response.create(
    model=TTS_MODEL,
    voice=TTS_VOICE,
    input=spoken_answer,
    instructions="Speak clearly and calmly as a customer support assistant.",
    response_format="wav",
) as response:
    response.stream_to_file(TTS_OUTPUT)

print("Saved", TTS_OUTPUT)
display(Audio(str(TTS_OUTPUT), autoplay=False))
```

> Say: "This is text-to-speech. The system is not understanding anything new at this stage. It is only changing the output modality from text to audio."

Ask:

> "The answer now sounds confident. Does that make it more correct?"

Emphasize:

"TTS makes the output feel natural, but it does not validate it. The spoken response inherits all upstream errors from ASR, retrieval, and the LLM."

---

### Step 8 - Show the complete pipeline map

```text
sample_support_call.wav
-> sample_16k_mono.wav
-> OpenAI transcript
-> OpenAI transcript embedding
-> similar past calls
-> OpenAI LLM answer
-> OpenAI TTS audio
-> answer_openai.wav
```

> Ask: "Where could this system fail?"

Good answers:

* the microphone recording is noisy
* ASR mishears a key word, code, name, or number
* retrieval returns the wrong past call
* the LLM overgeneralizes from weak context
* the generated voice makes the answer sound more certain than it should
* the API key, rate limit, network, or account permissions fail during class

---

## Optional Student Mini Challenge

Ask students to change the input content:

```text
My package is delayed and nobody can tell me when it will arrive.
```

Then rerun:

1. transcription or fallback transcript
2. embedding
3. retrieval
4. LLM answer
5. TTS output

Question:

> "Did the top retrieved call change from password reset to delivery delay?"

Bonus:

* Change the TTS instructions to make the voice slower or more empathetic.
* Change the retrieved context from top 2 to top 1 and compare the LLM answer.
* Add one sentence that says: "Please confirm before I take action."

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY` is missing | Check `.env`, run `load_dotenv()`, and restart the notebook kernel |
| `.env` is not loaded | Make sure `.env` is in the same folder as the notebook or pass the path explicitly: `load_dotenv("/path/to/.env")` |
| `No module named dotenv` | Run `pip install python-dotenv` |
| `No module named openai` | Run `pip install --upgrade openai` |
| `sample_support_call.wav` is missing | Use the fallback transcript and continue from retrieval |
| `ffmpeg` not found | Install FFmpeg; on macOS use `brew install ffmpeg`; on Ubuntu use `sudo apt-get install ffmpeg` |
| ASR transcript is poor | Re-record the sample more clearly or use the prepared fallback transcript |
| Embedding call fails | Check API key, account access, model name, and network connection |
| LLM call fails | Set `OPENAI_ANSWER_MODEL` in `.env` to a model available in your OpenAI account |
| TTS call fails | Check `OPENAI_TTS_MODEL`, `OPENAI_TTS_VOICE`, account access, and output path permissions |
| Notebook does not play audio | Open `answer_openai.wav` from the file browser instead |
| API rate limit occurs | Use fewer reruns, shorten audio, reduce class-wide simultaneous calls, or use instructor-only execution |
| Costs are a concern | Keep clips short, avoid repeated TTS generation, and run the demo once from the instructor machine |

---

## Instructor Wrap-Up

Close with this distinction:

| Step | What it does | What it does not do |
|---|---|---|
| Speech-to-text | Converts speech to text | Does not understand intent |
| Transcript embedding | Makes spoken content searchable | Does not reason about the issue |
| Retrieval | Finds similar prior content | Can retrieve irrelevant context |
| LLM | Summarizes and suggests action | Can hallucinate or overstate certainty |
| TTS | Speaks the answer | Does not improve correctness |

Final sentence to students:

> "A voice AI system is not one model. It is a pipeline, and every conversion step needs to be tested separately."

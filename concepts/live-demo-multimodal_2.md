# Live Demo Guide - Audio AI, Speech-to-Text, Retrieval, and Text-to-Speech

This demo uses a modular pipeline:

```text
Audio file
-> preprocessing
-> speech-to-text
-> transcript embedding
-> retrieval over past calls
-> LLM answer
-> text-to-speech
-> spoken output
```

The key teaching point: **Speech-to-text gives us text, embeddings make it searchable, the LLM generates an answer, and text-to-speech turns that answer back into audio.**

---

## Before the Session

### Install Python packages

```bash
pip install openai-whisper sentence-transformers pandas numpy scikit-learn librosa soundfile pyttsx3 ollama
```

### Install system dependencies

Whisper needs FFmpeg. `pyttsx3` also needs a local speech engine on Linux.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install ffmpeg espeak
```

Windows usually works with the built-in SAPI5 speech engine after installing the Python packages.

### Pre-download models before class

Do this before the session so you are not dependent on venue WiFi.

```python
import whisper
from sentence_transformers import SentenceTransformer

asr_model = whisper.load_model("tiny")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("ASR and embedding models loaded")
```

If you use the local LLM step, also pull the small Ollama model before class:

```bash
ollama pull llama3.2:1b
```

### Verify text-to-speech works

Run this before the session. It should create a local WAV file.

```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.save_to_file("Text to speech is ready.", "tts_check.wav")
engine.runAndWait()

print("Saved tts_check.wav")
```

Open or play `tts_check.wav`. If this fails, use the transcript-only fallback for the main demo and show the platform fallback commands in the troubleshooting section.

### Prepare one short audio file

Bring a short WAV file named:

```text
sample_support_call.wav
```

Recommended spoken content:

```text
Hi, I cannot log into my account. The password reset email never arrives, and I need access today.
```

Keep it under 15 seconds. A short, clean recording makes the demo more predictable.

### Font and audio setup

* Set terminal or notebook font to 18pt+.
* Test speaker volume before class.
* Keep `sample_support_call.wav` and `tts_check.wav` in the same folder as the notebook or script.

---

## Demo Flow

| # | What | Time |
|---|---|---:|
| 1 | Load and preprocess a short audio file | 4 min |
| 2 | Transcribe the audio with ASR | 5 min |
| 3 | Build a small past-call dataset | 3 min |
| 4 | Embed transcripts and retrieve similar calls | 7 min |
| 5 | Send transcript + retrieved context to an LLM | 5 min |
| 6 | Speak the generated answer with TTS and save it as audio | 5 min |
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

### Step 1 - Imports and constants

```python
from pathlib import Path

import librosa
import soundfile as sf
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_AUDIO = Path("sample_support_call.wav")
CLEAN_AUDIO = Path("sample_16k_mono.wav")
TTS_OUTPUT = Path("answer.wav")
```

> Say: "We start with a normal audio file. The rest of the pipeline turns that audio into text, then into searchable vectors, then into an answer, and finally back into speech."

---

### Step 2 - Load and preprocess the audio

```python
if not INPUT_AUDIO.exists():
    raise FileNotFoundError(
        "Missing sample_support_call.wav. Use the transcript fallback below if needed."
    )

audio, sample_rate = librosa.load(INPUT_AUDIO, sr=16000, mono=True)
duration = len(audio) / sample_rate

print("Sample rate:", sample_rate)
print("Duration:", round(duration, 2), "seconds")
print("Samples:", audio.shape)

sf.write(CLEAN_AUDIO, audio, sample_rate)
print("Saved", CLEAN_AUDIO)
```

> Say: "ASR models usually expect audio in a predictable format. Here we convert to 16 kHz mono so the model receives clean input."

Fallback if the audio file is missing:

```python
transcript = "Hi, I cannot log into my account. The password reset email never arrives, and I need access today."
print(transcript)
```

If you use the fallback transcript, skip Step 3 and continue with Step 4.

---

### Step 3 - Transcribe with speech-to-text

```python
import whisper

asr_model = whisper.load_model("tiny")
result = asr_model.transcribe(str(CLEAN_AUDIO), fp16=False)

transcript = result["text"].strip()
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

> Say: "These are previous calls. In a production system each row would link to the original audio file, timestamps, speaker metadata, and ticket outcome."

---

### Step 5 - Embed transcripts and retrieve similar calls

```python
embedder = SentenceTransformer("all-MiniLM-L6-v2")

past_texts = [item["transcript"] for item in past_calls]

past_embeddings = embedder.encode(
    past_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

query_embedding = embedder.encode(
    [transcript],
    convert_to_numpy=True,
    normalize_embeddings=True,
)

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

> Say: "We are not embedding the raw audio here. We are embedding the transcript. For spoken business content, this is usually the most robust first approach. Direct audio embeddings are useful for speaker similarity, music, environmental sound, and acoustic events."

---

### Step 6 - Send transcript and retrieved context to an LLM

Primary path with Ollama:

```python
from ollama import chat

retrieved_context = results.head(2).to_dict(orient="records")

prompt = f"""
You are a support assistant.

New call transcript:
{transcript}

Similar past calls:
{retrieved_context}

Task:
1. Summarize the user issue in one sentence.
2. Identify the likely intent.
3. Suggest the next best support action.
Keep the answer short and clear.
"""

response = chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": prompt}],
)

answer = response.message.content.strip()
print(answer)
```

Fallback if Ollama is unavailable:

```python
answer = (
    "The user cannot log in because the password reset email is not arriving. "
    "Likely intent: account access. "
    "Next action: verify the email address and trigger a manual password reset."
)
print(answer)
```

> Say: "The LLM is not listening to audio here. It receives the transcript and the retrieved context. That makes the system easier to inspect and debug."

---

### Step 7 - Convert the generated answer back into speech

This is the explicit Text-to-Speech part of the demo. Do it live, not only conceptually: first speak the answer through the system voice, then optionally save it as an audio file.

```python
import re
import pyttsx3


def clean_for_tts(text: str) -> str:
    """Make LLM output easier for a speech engine to read aloud."""
    text = re.sub(r"[*_`#>-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


spoken_answer = clean_for_tts(answer)

# Keep the live voice output short enough for class.
if len(spoken_answer) > 450:
    spoken_answer = spoken_answer[:450] + "."

print(spoken_answer)
```

Now speak it directly:

```python
engine = pyttsx3.init()
engine.setProperty("rate", 165)    # words per minute; lower = slower
engine.setProperty("volume", 0.9)  # 0.0 to 1.0

engine.say(spoken_answer)
engine.runAndWait()
```

> Say: "This is text-to-speech. The system is not understanding anything new at this stage. It is only changing the output modality from text to audio."

Optional: show that the same response can be saved for later playback.

```python
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.save_to_file(spoken_answer, str(TTS_OUTPUT))
engine.runAndWait()

print("Saved", TTS_OUTPUT)
```

Play it in a notebook:

```python
from IPython.display import Audio, display

display(Audio(str(TTS_OUTPUT), autoplay=False))
```

Or open `answer.wav` from the file browser.

If `pyttsx3` fails in the room, use this platform fallback. It keeps the architecture visible even when the Python TTS engine has driver issues.

```python
import platform
import shutil
import subprocess


def speak_with_system_tts(text: str) -> None:
    system = platform.system()

    if system == "Darwin" and shutil.which("say"):
        subprocess.run(["say", text], check=False)
    elif shutil.which("espeak"):
        subprocess.run(["espeak", text], check=False)
    else:
        print("No system TTS fallback available. Showing text instead:")
        print(text)


speak_with_system_tts(spoken_answer)
```

Ask:

> "The answer now sounds confident. Does that make it more correct?"

Emphasize:

"TTS makes the output feel natural, but it does not validate it. The spoken response inherits all upstream errors from ASR, retrieval, and the LLM."

---

### Step 8 - Show the complete pipeline map

```text
sample_support_call.wav
-> sample_16k_mono.wav
-> transcript
-> transcript embedding
-> similar past calls
-> LLM answer
-> answer.wav
```

> Ask: "Where could this system fail?"

Good answers:

* the microphone recording is noisy
* ASR mishears a key word, code, name, or number
* retrieval returns the wrong past call
* the LLM overgeneralizes from weak context
* TTS makes the answer sound more certain than it should

---

## Optional Student Mini Challenge

Ask students to change the input content:

```text
My package is delayed and nobody can tell me when it will arrive.
```

Then rerun:

1. transcript or ASR
2. embedding
3. retrieval
4. LLM answer
5. TTS output

Question:

> "Did the top retrieved call change from password reset to delivery delay?"

Bonus:

* Change the TTS speed from `170` to `130` or `210`.
* Make the answer shorter before speaking it.
* Add one sentence that says: "Please confirm before I take action."

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `sample_support_call.wav` is missing | Use the transcript fallback and continue from retrieval |
| Whisper model download is slow | Pre-run `whisper.load_model("tiny")` before class |
| `ffmpeg` not found | Install FFmpeg; on macOS use `brew install ffmpeg`; on Ubuntu use `sudo apt-get install ffmpeg` |
| ASR transcript is poor | Use the prepared fallback transcript; explain ASR failure modes |
| `import pyttsx3` fails | Run `pip install pyttsx3` |
| `pyttsx3` fails on Linux | Install `espeak`; use the platform fallback; or show the text answer |
| `answer.wav` is created but silent | Use direct `engine.say(...)`, check speaker output, open the file manually, or reduce text length |
| Notebook does not play audio | Open `answer.wav` from the file browser instead |
| Ollama is not running | Run `ollama serve` in another terminal or use the fallback answer |
| `ollama pull` is too slow | Use the hard-coded fallback answer and keep the LLM step conceptual |

---

## Platform TTS Fallback Commands

Use these only if `pyttsx3` is unstable. First save the answer to a text file:

```python
Path("answer.txt").write_text(spoken_answer, encoding="utf-8")
```

macOS:

```bash
say -o answer.aiff -f answer.txt
```

Ubuntu / Debian:

```bash
espeak -w answer.wav -f answer.txt
```

Windows PowerShell:

```powershell
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SetOutputToWaveFile("answer.wav")
$speak.Speak((Get-Content answer.txt -Raw))
$speak.Dispose()
```

---

## Instructor Wrap-Up

Close with this distinction:

| Step | What it does | What it does not do |
|---|---|---|
| ASR | Converts speech to text | Does not understand intent |
| Transcript embedding | Makes spoken content searchable | Does not reason about the issue |
| Retrieval | Finds similar prior content | Can retrieve irrelevant context |
| LLM | Summarizes and suggests action | Can hallucinate or overstate certainty |
| TTS | Speaks the answer | Does not improve correctness |

Final sentence to students:

> "A voice AI system is not one model. It is a pipeline, and each conversion step needs to be tested separately."

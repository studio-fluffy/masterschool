# Session - SLIDES: Audio AI, Speech-to-Text, and Text-to-Speech

---

### Slide 1 - Title

**Audio AI: Speech-to-Text and Text-to-Speech**
How spoken audio becomes text, embeddings, retrieval context, LLM input, and spoken output

---

### Slide 2 - Why Audio Matters in Multimodal AI

A lot of useful information is spoken, not written.

* Customer calls contain questions, complaints, intent, and emotion
* Meetings contain decisions, tasks, risks, and open questions
* Voice notes are faster to create than structured documents
* Machines, environments, and devices produce useful sounds

Audio AI turns sound into something systems can search, summarize, classify, and answer from.

---

### Slide 3 - The Core Audio Pipeline

Most practical audio AI systems follow the same basic flow:

1. Audio is loaded from a file or microphone.
2. The signal is decoded, resampled, cleaned, and segmented.
3. Speech-to-text converts spoken language into a transcript.
4. The transcript or audio signal is embedded for search or classification.
5. Relevant context is retrieved and passed to an LLM.
6. The LLM creates a text answer.
7. Text-to-speech converts the answer back into spoken output.

Important idea:

**Speech-to-text is only one step.**
A useful audio application usually needs preprocessing, retrieval, reasoning, and output generation too.

---

### Slide 4 - Audio Preprocessing and FFmpeg

Raw audio is not automatically ready for a model.

Common preprocessing steps:

* **Decode** - read MP3, WAV, M4A, or another format
* **Resample** - convert to the sample rate expected by the model, often 16 kHz
* **Convert to mono** - one channel instead of stereo
* **Normalize** - reduce volume differences
* **Segment** - split long recordings into smaller chunks

Where **FFmpeg** fits:

* FFmpeg is the common command-line tool behind many audio and video workflows
* It converts formats, extracts audio from video, resamples audio, and converts stereo to mono
* Libraries such as Whisper or Python audio tooling often depend on it for decoding

Practical takeaway:

In the demo, FFmpeg is used as the audio/video utility layer that makes sure the audio is in a model-friendly format before transcription.

---

### Slide 5 - Speech-to-Text / ASR

**Speech-to-Text** converts spoken language into written text.

Also called:

* ASR - Automatic Speech Recognition
* Transcription
* Audio-to-text

Example idea:

A spoken support request such as “I cannot log into my account” becomes a written transcript that can be searched, summarized, embedded, or sent to an LLM.

ASR is not reasoning. It only produces a transcript.

---

### Slide 6 - Where ASR Fails

Speech recognition is useful, but imperfect.

Common failure modes:

* background noise
* accents and dialects
* unclear pronunciation
* overlapping speakers
* domain-specific terms
* names, codes, product IDs, and numbers
* poor microphones or compression

Critical point:

A small transcription error can change the meaning of the downstream LLM answer.

---

### Slide 7 - Transcript Embeddings vs Direct Audio Embeddings

There are two common ways to represent audio for retrieval.

| Representation | Input | Best for |
|---|---|---|
| **Transcript embedding** | text transcript | semantic search over spoken content |
| **Direct audio embedding** | waveform or spectrogram | speaker similarity, music, sound events, non-speech audio |

Practical distinction:

* Transcript embeddings capture what was said.
* Direct audio embeddings capture acoustic properties such as voice, sound, tone, or background noise.

For most speech-first business apps, transcript embeddings are the simplest robust start.

---

### Slide 8 - Retrieval over Audio Content

Once audio is transcribed, we can search it like documents.

The practical retrieval flow:

1. Past calls are transcribed.
2. The transcripts are embedded.
3. The embeddings are stored in a searchable index.
4. A new call is transcribed and embedded.
5. Similar past calls are retrieved using vector similarity.

This enables:

* finding similar support calls
* retrieving related meeting segments
* searching voice notes
* grouping recurring issues
* passing relevant context to an LLM

Caveat:

If the transcript is wrong, retrieval may be wrong too.

---

### Slide 9 - Using an LLM After Transcription

An LLM can use the transcript and retrieved context to produce a useful answer.

Typical LLM tasks:

* summarize the audio
* classify intent or urgency
* extract action items
* translate the transcript
* draft a response
* answer questions using retrieved context

Recommended pattern:

Give the LLM the new transcript, the retrieved context, and a clear instruction. The model should answer from the evidence, not guess what was said.

---

### Slide 10 - Text-to-Speech / TTS

**Text-to-Speech** converts written text into spoken audio.

A TTS system must model:

* pronunciation
* pauses
* speed
* rhythm
* emphasis
* sentence melody
* voice characteristics

TTS is generation, not understanding.

Important safety point:

Voice output can sound authoritative even when the upstream transcript, retrieval, or LLM answer is wrong.

---

### Slide 11 - System Design Choices

There are two broad approaches.

| Approach | How it works | Strength | Weakness |
|---|---|---|---|
| **Pipeline** | ASR to text, embedding, LLM, then TTS | modular, inspectable, easy to debug | errors propagate step by step |
| **Native multimodal model** | audio goes directly into a multimodal model | fewer conversions, richer context | harder to inspect and control |

For a first production prototype, the pipeline approach is usually easier to validate.

---

### Slide 12 - What We Build in the Demo

The practical demo shows a compact audio-to-answer pipeline.

What happens step by step:

1. Load a short support-call audio file.
2. Convert it into a clean 16 kHz mono audio format.
3. Transcribe the spoken request with an ASR model.
4. Embed the transcript with a text embedding model.
5. Compare it with a small set of previous call transcripts.
6. Retrieve the most similar past calls.
7. Pass the transcript and retrieved context to an LLM.
8. Generate a short support answer.
9. Convert the answer back into spoken audio with TTS.

The goal is not to build a production system. The goal is to see how each component connects.

---

### Slide 13 - Session Plan

What we cover today:

1. **Audio pipeline** - preprocessing, FFmpeg, segmentation, and normalization
2. **Speech-to-text** - convert audio into transcript
3. **Audio representations** - transcript embeddings vs direct audio embeddings
4. **Retrieval** - find similar past audio content through transcripts
5. **LLM step** - summarize and respond using retrieved context
6. **Text-to-speech** - speak the generated answer

Suggested timing:

| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Practical demo | 30 min |
| Recap / questions | 5 min |

---

## Practical Demo Summary: Audio-to-Answer Pipeline

### Demo Goal

Show students how a spoken input becomes usable by an AI application.

By the end, students should understand:

* how audio preprocessing prepares files for ASR
* how speech-to-text creates a transcript
* how transcript embeddings enable retrieval over spoken content
* how retrieved context is passed to an LLM
* how the generated answer can be converted back into speech

---

### Practical Step 1 - Setup and Verification

The instructor verifies that the required tools are available:

* an ASR model for transcription
* an embedding model for transcript retrieval
* an LLM for response generation
* a TTS engine for spoken output
* FFmpeg or equivalent audio tooling for decoding and conversion

Purpose:

Make sure the environment is ready before the live pipeline starts.

---

### Practical Step 2 - Prepare a Short Audio File

The instructor uses a short support-call recording.

Recommended spoken content:

“My password reset email never arrives, and I need access today.”

Purpose:

Use a small, controlled example so students can clearly follow each transformation from audio to transcript to answer.

---

### Practical Step 3 - Load and Preprocess the Audio

The audio file is converted into a clean format suitable for speech recognition.

What happens practically:

* load the audio file
* convert it to mono
* resample it to 16 kHz
* save or pass the cleaned audio file to the ASR model

Purpose:

Show that audio models often expect a specific technical input format. Poor preprocessing can reduce transcription quality.

---

### Practical Step 4 - Transcribe the Audio

The ASR model converts the spoken request into written text.

What students should notice:

* the model does not “understand” the request yet
* it only produces a transcript
* any transcription error will affect later steps

Purpose:

Make the difference between transcription and reasoning explicit.

---

### Practical Step 5 - Build a Small Retrieval Dataset

The instructor prepares a small set of previous support-call transcripts.

Example topics:

* billing problem
* password reset issue
* delivery delay
* microphone issue

Purpose:

Simulate a realistic support archive where the system can look up similar past cases.

---

### Practical Step 6 - Embed and Retrieve Similar Calls

The transcript from the new call and the past-call transcripts are converted into embeddings.

What happens practically:

* generate embeddings for previous transcripts
* generate an embedding for the new transcript
* compare embeddings with similarity search
* rank the most similar past calls

Purpose:

Show how spoken content becomes searchable once it has been transcribed and embedded.

---

### Practical Step 7 - Generate an LLM Answer

The retrieved context and the new transcript are passed to an LLM.

The LLM produces a short answer containing:

* a one-sentence summary
* the likely user intent
* the next best support action

Purpose:

Show that the LLM should answer from transcript and retrieved evidence, not from guesswork.

---

### Practical Step 8 - Convert the Answer Back into Speech

The generated text answer is passed to a TTS engine.

What happens practically:

* clean the answer text
* synthesize spoken output
* optionally save the audio answer as a file
* play the answer back to the class

Purpose:

Close the full audio loop: spoken input becomes transcript, context, LLM answer, and spoken output.

---

### Practical Step 9 - Connect Back to the Theory

The instructor maps the demo back to the conceptual pipeline:

* raw audio
* preprocessing
* speech-to-text
* transcript
* transcript embedding
* retrieval
* LLM answer
* text-to-speech

Key takeaway:

This class demonstrates a pipeline architecture. It is modular and easy to inspect, but every downstream step depends on the quality of the previous step.

---

### Optional Student Challenge

Students change the input request and observe whether retrieval changes.

Example request:

“My package is delayed and nobody can tell me when it will arrive.”

Students should check:

* Does the transcript reflect the spoken input correctly?
* Does retrieval now find the delivery-delay example?
* Does the LLM answer change accordingly?
* Does TTS speak the new answer clearly?

Purpose:

Show that retrieval is driven by semantic similarity, not keyword matching alone.

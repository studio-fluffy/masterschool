# Session - SLIDES: Introduction to Multimodal AI

60-minute version

---

### Slide 0 - Agenda

**What we cover today:**

1. **Data types** – how AI processes text, images, audio, video
2. **Finding matches** – how to search and compare data
3. **Key difference** – between searching and reasoning
4. **Live demo** – match images and text together
5. **Reality check** – where this works, where it fails

---

### Slide 1 - Title

**Introduction to Multimodal AI**
How text, images, audio, and video become searchable and usable by AI systems

---

### Slide 2 - Why Multimodal AI Matters

Most real-world information is not only text.

* A support ticket may include text, screenshots, logs, PDFs, and voice notes
* A video contains images, audio, subtitles, timestamps, and metadata
* A meeting contains speech, slides, chat, and shared screens

**Multimodal AI** means making different data types usable together.

---

### Slide 3 - What Is a Modality?

A **modality** is a type of information the system can process.

| Modality | Raw form | Example task |
|---|---|---|
| Text | words / tokens | summarize a document |
| Image | pixels | find a matching caption |
| Audio | waveform | transcribe speech |
| Video | frames + audio + time | search for a moment |

Core idea: each modality starts in a different raw format.

---

### Slide 4 - From Raw Data to Representations

Models do not directly work with human meaning.
They convert raw input into numerical representations.

```text
Text  -> tokens / token embeddings
Image -> pixels / patches / image embeddings
Audio -> waveform / spectrogram / audio embeddings
Video -> frames / clips / audio / metadata embeddings
```

The model sees numbers. Good representations preserve useful patterns.

---

### Slide 5 - Tokens, Features, and Embeddings

These terms are related, but not identical.

* **Token** - an input unit, such as a word piece or image patch
* **Feature** - a signal extracted from raw data, such as edges, shapes, or frequencies
* **Embedding** - a dense vector that represents content in a numerical space

Important distinction:

```text
A transcript is readable.
An embedding is searchable.
A model answer is generated.
```

Do not treat these as the same thing.

---

### Slide 6 - Encoders and Shared Embedding Spaces

Different modalities usually need different encoders.

* Text encoder -> text vectors
* Vision encoder -> image vectors
* Audio encoder -> audio vectors
* Video pipeline -> frames, clips, audio, transcript, and metadata vectors

A **shared embedding space** lets different modalities become comparable:

```text
Image of a cat          -> vector A
Text: "a cat"           -> vector B
Audio: someone says cat -> vector C
```

If the system is trained well, related vectors are close together.

---

### Slide 7 - Cosine Similarity and Vector Search

Cosine similarity compares the direction of two vectors.

* High score -> vectors are close in embedding space
* Low score -> vectors are less related
* Used for semantic search, recommendations, and retrieval

Basic retrieval loop:

```text
items -> embeddings -> vector store
query -> embedding  -> compare -> top-k results
```

Critical caveat:

**Similarity is not truth.**
A high score means "close in embedding space", not "guaranteed correct".

---

### Slide 8 - Retrieval Is Not Reasoning

These are different operations.

| Concept | Meaning |
|---|---|
| **Transcription** | Convert speech into text |
| **Embedding** | Convert content into a vector |
| **Retrieval** | Find similar or relevant items |
| **Reasoning** | Combine evidence and infer an answer |
| **Generation** | Produce text, audio, image, or action |

Multimodal retrieval finds evidence.
Multimodal reasoning interprets evidence.

---

### Slide 9 - Common Failure Modes

Multimodal systems can look convincing while being wrong.

* A retrieved image is similar, but not the right one
* A caption matches the topic but misses the important detail
* Embeddings ignore small but critical visual differences
* A transcript contains a recognition error
* A retrieval score is treated as proof

Rule of thumb: always inspect examples, not only scores.

---

### Slide 10 

What we cover today:

1. **Core theory** - modalities, encoders, embeddings, shared spaces
2. **Retrieval mechanism** - cosine similarity and vector search
3. **Key distinction** - transcription vs embeddings vs retrieval vs reasoning
4. **Live demo** - match images and text with embeddings
5. **Reflection** - where this works, where it fails

The goal is not to build a full multimodal app yet.
The goal is to understand the core mechanism.



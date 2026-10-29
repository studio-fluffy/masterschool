# Session - SLIDES: Building Multimodal Applications

---

### Slide 1 - Title

**Building Multimodal Applications**
From video, audio, frames, transcripts, and metadata to multimodal retrieval and grounded answers

---

### Slide 2 - The Capstone Idea

Session 4 combines the previous sessions into one application.

We already covered:

* text and image embeddings
* speech-to-text and text-to-speech
* voice agents, memory, and tools

Now we build a searchable multimodal application.

The core question:

**How can an AI system find the relevant moment inside a video and answer from that evidence?**

---

### Slide 3 - Video Is Not One Modality

A video file is a container.

It can contain:

* visual frames
* spoken audio
* environmental sound
* subtitles
* timestamps
* metadata
* scene structure

A useful multimodal application separates these parts before it searches or reasons over them.

```text
video -> frames + audio + transcript + metadata
```

---

### Slide 4 - The Core Application Pipeline

A practical multimodal RAG pipeline often looks like this:

1. Load a video.
2. Extract representative frames.
3. Extract the audio track.
4. Transcribe the spoken audio.
5. Split transcript and frames into timestamped records.
6. Generate embeddings for searchable records.
7. Store embeddings with metadata.
8. Retrieve relevant segments for a user query.
9. Pass the retrieved evidence to an LLM.
10. Generate a grounded answer.

The key idea:

**The LLM should answer from retrieved evidence, not from memory or guesswork.**

---

### Slide 5 - Frame Sampling

Processing every frame is usually wasteful.

At 30 frames per second:

```text
10 minutes x 60 seconds x 30 FPS = 18,000 frames
```

Most applications sample representative frames instead.

Common strategies:

* one frame every few seconds
* one frame per scene change
* keyframes from the video codec
* frames around transcript hits
* frames selected by visual change

The tradeoff:

**More frames can improve recall, but increase cost, storage, and latency.**

---

### Slide 6 - Audio Extraction and Transcription

The audio track makes spoken content searchable.

Typical flow:

```text
video -> audio track -> speech-to-text -> timestamped transcript
```

A plain transcript is useful, but timestamps make it much more valuable.

Instead of saying:

```text
The topic appears somewhere in the video.
```

The system can say:

```text
The relevant explanation starts around 00:14.
```

---

### Slide 7 - Timestamped Records

The index should not store only vectors.

Each searchable item needs metadata.

Example transcript record:

```text
modality: transcript
start_time: 00:10
end_time: 00:18
text: The password reset email never arrives.
```

Example frame record:

```text
modality: frame
timestamp: 00:15
frame_path: frames/frame_003.jpg
visual_summary: support dashboard with password reset issue
```

Metadata connects retrieval results back to the original media.

---

### Slide 8 - Embeddings for Different Modalities

Different parts of the video can be represented differently.

| Record type | Common representation |
|---|---|
| Transcript segment | text embedding |
| Subtitle segment | text embedding |
| Frame | image embedding or caption embedding |
| Audio segment | audio embedding or transcript embedding |
| Metadata | structured filter or text embedding |

A robust first version often starts with transcript embeddings and frame descriptions.

A more advanced version uses direct image and audio embeddings.

---

### Slide 9 - Fusion Strategies

Fusion means combining information from multiple modalities.

| Strategy | Idea | Example |
|---|---|---|
| Early fusion | combine inputs before the model processes them | image and text tokens together |
| Late fusion | search each modality separately, then merge results | transcript search + frame search |
| Embedding-level fusion | combine vectors or scores | weighted ranking across modalities |

For a first prototype, late fusion is usually easiest to debug.

---

### Slide 10 - Multimodal Retrieval

A user query can be matched against different records.

Example query:

```text
Where does the speaker explain the password reset problem?
```

The system may retrieve:

* transcript segment around 00:12
* frame around 00:15
* metadata for the same video section

The result should include both content and location.

Retrieval answers:

```text
What evidence is relevant?
Where is it in the media?
```

---

### Slide 11 - Multimodal RAG

After retrieval, the application passes evidence to an LLM.

The prompt should contain:

* the user question
* retrieved transcript snippets
* relevant frame descriptions
* timestamps
* clear instruction to answer only from evidence

The LLM then produces a grounded answer.

Important distinction:

**Retrieval selects evidence. The LLM formulates the answer.**

---

### Slide 12 - Evaluation

A multimodal application needs more than a nice answer.

Evaluate:

| Area | Question |
|---|---|
| Retrieval quality | Did we retrieve the right segment? |
| Timestamp accuracy | Is the location correct? |
| Answer faithfulness | Is the answer supported by evidence? |
| Latency | Is search fast enough? |
| Storage | How many frames, clips, and vectors are stored? |
| Cost | Which steps are expensive? |

The system is only useful if it finds the right evidence reliably.

---

### Slide 13 - Failure Modes

Common failures:

* sampled frames miss the important visual moment
* speech-to-text mishears a key phrase
* transcript segmentation cuts context in the wrong place
* retrieval finds the right topic but wrong timestamp
* ranking favors text while visual evidence contradicts it
* the LLM answers beyond the retrieved evidence
* cost grows quickly with long videos

Design goal:

**Make errors visible and traceable.**

---

### Slide 14 - What We Build in the Demo

The demo builds a small searchable video assistant.

What happens practically:

1. Load a short support-training video.
2. Extract frames and the audio track.
3. Transcribe or load timestamped transcript segments.
4. Create searchable transcript and frame records.
5. Generate embeddings for the records.
6. Store vectors with timestamps and metadata.
7. Search the video with a text query.
8. Retrieve the best transcript and frame evidence.
9. Generate a grounded answer with timestamps.

The output is not just an answer.

It is an answer plus evidence.

---

### Slide 15 - Session Plan

What we cover today:

1. **Video decomposition** - frames, audio, transcript, metadata
2. **Sampling** - representative frames instead of every frame
3. **Timestamped indexing** - records with modality and location
4. **Embeddings and retrieval** - search over transcript and frame records
5. **Fusion** - merge evidence from multiple modalities
6. **Grounded answer generation** - answer from retrieved evidence
7. **Evaluation** - quality, latency, storage, and cost

Suggested timing:

| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Practical demo | 30 min |
| Recap / questions | 5 min |

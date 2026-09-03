# Session — SLIDES: Local LLMs & Hugging Face

---

### Slide 1 — Title

**Local LLMs & Hugging Face**
Run AI on your own machine — no cloud required

---

### Slide 2 — The Problem

Every time you call a cloud LLM API:

* You pay per token
* Your data is sent to an external server
* You depend on an internet connection
* You are subject to rate limits and pricing changes
* Some data cannot legally leave your infrastructure

---

### Slide 3 — What Can Run Locally Today

Local AI is more capable than most people expect:

* **Text to text** — chat, summarisation, code generation
* **Vision** — send an image, get a description or analysis
* **Multimodal** — combine text and image in one prompt
* **Image generation** — text to image (experimental in Ollama)
* Smaller models are catching up fast with cloud quality

---

### Slide 4 — Ollama

The easiest way to run models locally.

* One-line install on macOS, Linux, and Windows
* Built-in model library — like a package manager for AI
* REST API that works with any OpenAI SDK out of the box
* Run a model in one command: `ollama run gemma3:4b`

**This is what we focus on today.**

---

### Slide 5 — LM Studio

A desktop GUI for running models locally.

* Browse and download models with a visual interface
* No terminal required — good for non-developers
* OpenAI-compatible API on `localhost:1234`
* Built on llama.cpp under the hood

---

### Slide 6 — llama.cpp

The engine that powers most local LLM tools.

* A C++ library — no Python, no heavy dependencies
* Works on CPU, NVIDIA, AMD, and Apple Silicon
* What Ollama and LM Studio are built on
* Use it directly when you need maximum control

---

### Slide 7 — vLLM

Production-grade LLM serving at scale.

* Designed for multi-user, high-throughput deployments
* 2–24x faster than standard HuggingFace Transformers under load
* Requires an NVIDIA or AMD GPU
* Used in production by Meta, Mistral AI, and Stripe

Not for local laptops — this is the tool when you deploy.

---

### Slide 8 — Models You Can Run Today

A rough guide by RAM:

| RAM | Model examples | Size |
|---|---|---|
| 8 GB | llama3.2:1b, gemma3:1b, qwen3:1.7b | < 2 GB |
| 8 GB | gemma3:4b, qwen3:4b | 2–4 GB |
| 16 GB | qwen3:8b, gemma3:12b | 5–8 GB |
| 32 GB | gemma3:27b, qwen3:30b | 17–20 GB |

Rule of thumb: **your RAM in GB ÷ 4 ≈ the model size you can run comfortably**

---

### Slide 9 — What is Hugging Face?

The central hub for the open AI ecosystem.

* **Models** — 400,000+ model checkpoints, any task
* **Datasets** — 100,000+ datasets for training and evaluation
* **Spaces** — live demos anyone can run in a browser
* **Leaderboards** — standardised benchmarks for comparing models

Think of it as GitHub, but for AI models.

---

### Slide 10 — Finding Models for Local Use

How to find a model you can actually run:

1. Go to `huggingface.co/models`
2. Filter by library: **GGUF**
3. Search for the model name you want
4. Look for uploads by **bartowski** — the most active GGUF provider
5. Pick a quantisation level (more on this next)

---

### Slide 11 — What is Quantisation?

Reducing the precision of model weights to make models smaller.

* **FP32** — full precision, 4 bytes per parameter (used in training)
* **FP16** — standard inference, 2 bytes per parameter
* **INT8** — 1 byte per parameter, 50% smaller, minimal quality loss
* **INT4** — 0.5 bytes per parameter, 75% smaller, still very usable

A 7B model at FP16 needs ~14 GB of RAM.
The same model at INT4 needs ~4–5 GB.

---

### Slide 12 — Quantisation Levels in Practice

The GGUF format uses named quantisation levels:

* **Q8_0** — near-lossless quality, largest file
* **Q4_K_M** — the recommended sweet spot ⭐
* **Q4_K_S** — slightly smaller, slightly lower quality
* **Q2_K** — very small, noticeable quality degradation

Start with **Q4_K_M** for almost everything.

---

### Slide 13 — The Open LLM Leaderboard

A public benchmark for comparing open-weight models.

* URL: `huggingface.co/spaces/open-llm-leaderboard`
* Tests models on six standardised benchmarks
* Covers reasoning, maths, science, instruction-following
* Filter by parameter count to find the best model in your size range

Caveat: leaderboard scores measure specific tasks — always test on your own use case.

---

### Slide 14 — Session Plan

What we will build together:

1. **Install Ollama** and run a 1B model in the terminal
2. **Write a Python chatbot** using the Ollama library
3. **Explore Hugging Face** — find a GGUF model and pull it
4. **Check the leaderboard** — compare models by size

---

## Live Demo: Running a Local LLM with Ollama

### Demo Goal

Show students how to install Ollama, run a model interactively in the terminal, then talk to it from Python — all without an API key or internet connection after install.

By the end of the demo, students should understand:

* A model can run entirely on a local machine
* The same chat pattern from OpenAI works locally
* The REST API is a drop-in replacement for the OpenAI API

---

### 1. Install Ollama

Run the installer:

```bash
# macOS and Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download OllamaSetup.exe from ollama.com/download
```

Verify it is running:

```bash
curl http://localhost:11434
```

Expected output: `Ollama is running`

Explain:

"This starts a small server on your machine on port 11434. Every model you run goes through this server."

---

### 2. Pull and Run a 1B Model

```bash
ollama run llama3.2:1b
```

This downloads the model if needed (~1.3 GB) and opens an interactive chat session.

Try these prompts with students:

```
>>> Hello, what are you?
>>> What is the capital of France?
>>> Write a haiku about running AI locally
>>> /bye
```

Explain:

"That prompt never left this machine. No API key. No bill. No internet after the initial download."

---

### 3. Explore Model Management

```bash
# See all installed models
ollama list

# See what is currently loaded in memory
ollama ps

# Show details about a model
ollama show llama3.2:1b
```

Explain:

"Models stay loaded in memory for a few minutes after last use, then unload automatically to free up RAM."

---

### 4. The REST API

```bash
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": false
  }'
```

Then show the OpenAI-compatible endpoint:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'
```

Explain:

"The second endpoint — `/v1/chat/completions` — is the exact same path as the OpenAI API. Any existing OpenAI code works by changing one line."

---

### 5. Python — Basic Chat

```python
from ollama import chat

response = chat(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": "Explain what a neural network is in 2 sentences"}
    ]
)

print(response.message.content)
```

Explain:

"Same structure as the OpenAI SDK — model, messages, role, content. The only difference is we imported from ollama instead of openai."

---

### 6. Python — Multi-Turn Conversation

Build this with students step by step:

```python
from ollama import chat

messages = []

print("Chat with llama3.2:1b — type 'quit' to exit")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        break

    messages.append({"role": "user", "content": user_input})

    response = chat(model="llama3.2:1b", messages=messages)
    reply = response.message.content

    messages.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")
```

Explain:

"The model itself has no memory. We keep the conversation alive by sending the full history every turn. This is how all chat apps work — OpenAI, Claude, everything."

---

### 7. OpenAI SDK Drop-in

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required by the library, ignored by Ollama
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "What is gradient descent?"}]
)

print(response.choices[0].message.content)
```

Explain:

"This is an existing OpenAI code pattern. Two lines changed: `base_url` and `api_key`. The rest is identical."

---

### 8. Pull a Model from Hugging Face

Switch to the browser. Go to `huggingface.co/models`, filter by GGUF, find a bartowski model.

Then back in the terminal:

```bash
# Pull any GGUF model from Hugging Face directly into Ollama
ollama run hf.co/bartowski/Qwen3-4B-GGUF
```

Explain:

"Ollama's built-in library covers the most popular models. This command gives you access to everything else on Hugging Face."

---

### Student Mini Challenge

Ask students to:

* Pull a second model of their choice from the Ollama library
* Ask it the same question you asked `llama3.2:1b`
* Compare the quality and speed

Bonus:

* Use the OpenAI SDK pattern to query the local model
* Try `gemma3:4b` and send it an image file to describe

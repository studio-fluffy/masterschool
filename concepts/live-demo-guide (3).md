# Live Demo Guide — Local LLMs & Hugging Face

Quick reference for the instructor. Full context in `local-llms-huggingface-live-session.md`.

---

## Before the Session

### Install
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: OllamaSetup.exe from ollama.com/download
```

### Pre-pull models (do this at home, not on venue WiFi)
```bash
ollama pull llama3.2:1b      # 1.3 GB — main demo model
ollama pull gemma3:4b        # 3.3 GB — vision demo (optional)
```

### Verify everything works
```bash
curl http://localhost:11434
# → "Ollama is running"

ollama run llama3.2:1b "Say hello in 5 words"
# → should respond immediately

python -c "import ollama; print('ok')"
# → ok
```

### Browser tabs to have open
- `ollama.com/library` — model browser
- `huggingface.co/models` — filter demo
- `huggingface.co/bartowski` — GGUF quantizations
- `huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard`

### Font size
Set terminal font to 18pt+ before class. Students need to read from the back of the room.

---

## Demo Flow

| # | What | Time |
|---|---|---|
| 1 | Install Ollama, show it running | 5 min |
| 2 | Pull and run llama3.2:1b, chat in terminal | 8 min |
| 3 | Explore model management commands | 4 min |
| 4 | REST API with curl | 6 min |
| 5 | Python — basic chat | 3 min |
| 6 | Python — multi-turn conversation | 4 min |
| 7 | Python — OpenAI SDK drop-in | 3 min |
| 8 | Hugging Face browser walkthrough | 5 min |
| 9 | Pull a model from HF into Ollama | 2 min |
| **Total** | | **~40 min** |

---

## Step-by-Step Commands

### Step 1 — Install and verify

```bash
curl -fsSL https://ollama.com/install.sh | sh

curl http://localhost:11434
# "Ollama is running"
```

> Say: "This starts a local server on port 11434. Every model you run goes through this — no cloud, no API key."

---

### Step 2 — Run your first model

```bash
ollama run llama3.2:1b
```

```
>>> Hello, what are you?
>>> What is the capital of France?
>>> Write a haiku about running AI locally
>>> /bye
```

> Say: "That prompt never left this machine. No bill, no internet after the initial download."

---

### Step 3 — Model management

```bash
ollama list           # installed models
ollama ps             # currently loaded in memory
ollama show llama3.2:1b   # architecture, quantization, license
```

> Say: "Models unload from memory automatically after a few minutes of inactivity."

---

### Step 4 — REST API

```bash
# Ollama's native endpoint
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "stream": false
  }'
```

```bash
# OpenAI-compatible endpoint
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

> Say: "The second path — `/v1/chat/completions` — is identical to OpenAI's API. Existing code works by changing one line."

---

### Step 5 — Python basic chat

```python
from ollama import chat

response = chat(
    model="llama3.2:1b",
    messages=[
        {"role": "user", "content": "Explain a neural network in 2 sentences"}
    ]
)

print(response.message.content)
```

> Say: "Same structure as the OpenAI SDK. Model, messages, role, content. Only the import changes."

---

### Step 6 — Python multi-turn conversation

```python
from ollama import chat

messages = []

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

Ask a follow-up question that requires remembering the first answer.

> Say: "The model has no memory. We keep the conversation alive by sending the full history every turn. This is how every chat app works."

---

### Step 7 — OpenAI SDK drop-in

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "What is gradient descent?"}]
)

print(response.choices[0].message.content)
```

> Say: "Two lines changed from standard OpenAI code: base_url and api_key. Everything else is identical."

---

### Step 8 — Hugging Face browser walkthrough

Go to `huggingface.co/models` in the browser.

1. Filter by library: **GGUF**
2. Search: **qwen3**
3. Click `bartowski/Qwen3-8B-GGUF`
4. Open **Files and versions** tab → find `Qwen3-8B-Q4_K_M.gguf`
5. Open `huggingface.co/spaces/open-llm-leaderboard` — sort by Average, filter to 1–10B

> Say: "bartowski uploads quantized versions of nearly every major model within days of release. Q4_K_M is the file you want."

---

### Step 9 — Pull from Hugging Face

```bash
ollama run hf.co/bartowski/Qwen3-4B-GGUF
```

> Say: "Ollama's library covers the popular models. This command unlocks everything else on Hugging Face."

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `ollama: command not found` | Restart terminal; or `export PATH=$PATH:/usr/local/bin` |
| `connection refused` on port 11434 | Run `ollama serve` in a separate terminal |
| Model downloads slowly | Use the pre-pulled model; skip to Step 3 |
| `import ollama` fails | `pip install ollama` |
| HF pull fails | Fall back to `ollama run qwen3:4b` from the built-in library |
| Student machine has < 8 GB RAM | Use `gemma3:270m` (292 MB) instead of llama3.2:1b |

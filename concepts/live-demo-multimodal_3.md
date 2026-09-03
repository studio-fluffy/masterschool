# Live Demo Guide - Talking to Agents

This demo builds a bounded voice agent for a support scenario.

```text
spoken request
-> speech-to-text
-> conversation state
-> transcript embedding
-> semantic memory retrieval
-> tool plan
-> confirmation
-> Python tool call
-> response text
-> text-to-speech
```

The key teaching point: **a voice interface is not automatically an agent.** The agent behavior comes from memory, state, tool planning, permissions, and controlled execution.

---

## Teaching Goal

Students should leave with one clear mental model:

```text
observe -> retrieve -> plan -> confirm -> act -> respond
```

They should also understand the boundary between the components:

| Component | Purpose |
|---|---|
| Speech-to-text | Convert spoken request into transcript |
| Conversation state | Track current and previous turns |
| Embeddings | Make transcripts and memories searchable |
| Retrieval | Select relevant prior context |
| Tool plan | Decide whether an action is needed |
| Confirmation | Prevent unsafe or unintended side effects |
| Tool execution | Run a Python function or external API |
| Text-to-speech | Speak the final result |

The default path works without relying on an OpenAI text-generation model. It uses OpenAI for speech-to-text and embeddings, tries OpenAI text-to-speech when the project has access, and uses a transparent rule-based planner for the tool decision. If OpenAI TTS is unavailable, the demo falls back to local operating-system speech where possible or shows the text response only. If your API account supports text-generation models, you can turn on the optional LLM planner.

After the manual Python demo, the guide maps the same loop to LangGraph. That section is optional and is meant to show the production pattern: explicit state, nodes, conditional routing, and human approval points.

---

## Before the Session

### Install Python packages

```bash
pip install --upgrade openai python-dotenv pandas numpy scikit-learn ipython
```

Optional local TTS fallback:

```bash
pip install pyttsx3
```

Optional for the LangGraph production-pattern section:

```bash
pip install langgraph
```

FFmpeg is recommended for audio conversion.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

### Prepare the `.env` file

Create a file named `.env` in the same folder as the notebook or Python script.

```text
OPENAI_API_KEY=your_api_key_here
```

Recommended defaults:

```text
OPENAI_STT_MODEL=whisper-1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TTS_PROVIDER=auto
OPENAI_TTS_MODELS=gpt-4o-mini-tts,tts-1,tts-1-hd
OPENAI_TTS_VOICE=alloy
USE_OPENAI_LLM=false
```

Optional LLM mode, only if your API account supports text-generation models:

```text
USE_OPENAI_LLM=true
OPENAI_AGENT_MODEL=gpt-4o-mini
```

Do not paste the API key into notebook cells. Keep it in `.env`.

### Prepare one short audio file

Recommended filename:

```text
sample_agent_request.wav
```

Recommended spoken content:

```text
Hi, I cannot log into my account. The password reset email never arrives, and I need access today. Please create a support ticket.
```

If this file is missing, the demo uses a fallback transcript and continues.

### Verify the setup

```python
import os
import shutil
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("FFmpeg available:", bool(shutil.which("ffmpeg")))
print("macOS say available:", bool(shutil.which("say")))
print("espeak available:", bool(shutil.which("espeak")))

client = OpenAI()
print("OpenAI client ready")
```

### Font and audio setup

* Set notebook and terminal font to 18pt+.
* Test speaker volume before class.
* On macOS, local fallback speech uses the built-in `say` command when OpenAI TTS is not available.
* On Linux, local fallback speech uses `espeak` when it is installed.
* Keep `.env` in the same folder as the notebook or script.
* Keep the audio file short, ideally under 15 seconds.

---

## Demo Flow


| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Live demo | 30 min |
| Optional LangGraph mapping or recap | 5 min |

---

## Step-by-Step Commands

### Step 1 - Imports and constants

```python
from pathlib import Path
from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Audio, display
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

INPUT_AUDIO = Path("sample_agent_request.wav")
CLEAN_AUDIO = Path("sample_agent_request_16k_mono.wav")
TTS_OUTPUT = Path("agent_answer.wav")
TTS_AIFF_OUTPUT = Path("agent_answer.aiff")

STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower()
OPENAI_TTS_MODELS = [
    item.strip()
    for item in os.getenv("OPENAI_TTS_MODELS", "gpt-4o-mini-tts,tts-1,tts-1-hd").split(",")
    if item.strip()
]
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

USE_OPENAI_LLM = os.getenv("USE_OPENAI_LLM", "false").lower() in {"1", "true", "yes"}
AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")

FALLBACK_TRANSCRIPT = (
    "Hi, I cannot log into my account. "
    "The password reset email never arrives, and I need access today. "
    "Please create a support ticket."
)

print("OpenAI key set:", bool(os.environ.get("OPENAI_API_KEY")))
print("FFmpeg available:", bool(shutil.which("ffmpeg")))
print("macOS say available:", bool(shutil.which("say")))
print("espeak available:", bool(shutil.which("espeak")))
print("Input audio exists:", INPUT_AUDIO.exists())
print("Speech-to-text model:", STT_MODEL)
print("Embedding model:", EMBEDDING_MODEL)
print("TTS provider:", TTS_PROVIDER)
print("OpenAI TTS candidates:", OPENAI_TTS_MODELS)
print("TTS voice:", TTS_VOICE)
print("Use optional OpenAI LLM planner:", USE_OPENAI_LLM)
if USE_OPENAI_LLM:
    print("Agent model:", AGENT_MODEL)
```

> Say: "Speech-to-text and embeddings use OpenAI APIs. The agent decision is rule-based by default so the class demo does not depend on text-generation model access. The speech output is defensive: it tries OpenAI TTS when available and falls back to local speech or text-only output when access is denied."
---

### Step 2 - Prepare the audio

```python
def prepare_audio(input_path: Path, output_path: Path) -> Path | None:
    """Convert audio to 16 kHz mono WAV when FFmpeg and input audio are available."""
    if not input_path.exists():
        print("No audio file found. The demo will use the fallback transcript.")
        return None

    if not shutil.which("ffmpeg"):
        print("FFmpeg not found. Using the original audio file.")
        return input_path

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    print("Prepared audio:", output_path)
    return output_path

prepared_audio = prepare_audio(INPUT_AUDIO, CLEAN_AUDIO)
```

> Say: "We make the input predictable before transcription. This is a pipeline design detail, not an agent detail, but bad preprocessing can break the whole system."

---

### Step 3 - Transcribe the request

```python
def transcribe_or_fallback(audio_path: Path | None) -> str:
    if audio_path is None:
        return FALLBACK_TRANSCRIPT

    try:
        with audio_path.open("rb") as f:
            result = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
            )
        return result.text.strip()
    except Exception as exc:
        print("Transcription failed. Using fallback transcript.")
        print(type(exc).__name__, exc)
        return FALLBACK_TRANSCRIPT

user_transcript = transcribe_or_fallback(prepared_audio)
print("USER TRANSCRIPT:")
print(user_transcript)
```

> Ask: "At this point, do we have an agent?"  
> Expected answer: "No. We only have a transcript."

---

### Step 4 - Create conversation state

```python
conversation_state = {
    "conversation_id": f"conv_{uuid.uuid4().hex[:8]}",
    "customer_id": "cust_demo_001",
    "messages": [
        {
            "role": "user",
            "content": user_transcript,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    ],
    "pending_action": None,
    "retrieved_memories": [],
    "tool_results": [],
}

conversation_state
```

> Say: "State is what makes follow-up turns possible. If the user later says 'yes', this state tells us what they are confirming."

---

### Step 5 - Build a small semantic memory

```python
memory_items = [
    {
        "id": "mem_001",
        "type": "previous_call",
        "topic": "password_reset",
        "text": "Customer could not log in because the password reset email was not arriving. Support created a high-priority access ticket and advised checking spam and email filters.",
    },
    {
        "id": "mem_002",
        "type": "previous_call",
        "topic": "billing",
        "text": "Customer reported being charged twice for the same subscription renewal. Billing team issued a refund after confirming duplicate payment.",
    },
    {
        "id": "mem_003",
        "type": "knowledge_base",
        "topic": "email_delivery",
        "text": "Password reset emails may be delayed by spam filters, corporate firewalls, expired links, or blocked sender domains.",
    },
    {
        "id": "mem_004",
        "type": "previous_call",
        "topic": "delivery_delay",
        "text": "Customer asked about a delayed package and needed an updated delivery estimate from the shipping provider.",
    },
    {
        "id": "mem_005",
        "type": "knowledge_base",
        "topic": "microphone_issue",
        "text": "If the microphone is not detected, users should check browser permissions, operating system privacy settings, and selected input device.",
    },
]

memory_df = pd.DataFrame(memory_items)
memory_df[["id", "type", "topic", "text"]]
```

> Say: "This is a tiny stand-in for previous calls, tickets, documents, or notes. The agent should retrieve relevant context before acting."

---

### Step 6 - Embed and retrieve relevant memories

```python
def embed_texts(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)

memory_embeddings = embed_texts(memory_df["text"].tolist())
query_embedding = embed_texts([user_transcript])

scores = cosine_similarity(query_embedding, memory_embeddings)[0]
memory_df["score"] = scores

retrieved = memory_df.sort_values("score", ascending=False).head(3).copy()
conversation_state["retrieved_memories"] = retrieved.to_dict(orient="records")

retrieved[["id", "type", "topic", "score", "text"]]
```

> Say: "This is semantic memory. The agent is not only reacting to the latest transcript. It is grounding the next step in similar prior context."

---

### Step 7 - Define Python tools

```python
support_tickets = []

help_articles = {
    "password_reset": (
        "Ask the customer to check spam, verify the email address, wait a few minutes, "
        "and retry the reset link. If access is urgent, create a support ticket."
    ),
    "billing": "Check invoices, payment provider status, and refund eligibility.",
    "delivery_delay": "Check shipment status and carrier tracking events.",
}


def search_help_article(topic: str) -> dict:
    return {
        "tool": "search_help_article",
        "topic": topic,
        "article": help_articles.get(topic, "No matching article found."),
    }


def create_support_ticket(customer_id: str, issue_type: str, priority: str, summary: str) -> dict:
    ticket_id = f"TCK-{len(support_tickets) + 1:04d}"
    ticket = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "issue_type": issue_type,
        "priority": priority,
        "summary": summary,
        "status": "created",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    support_tickets.append(ticket)
    return ticket


tools = {
    "search_help_article": search_help_article,
    "create_support_ticket": create_support_ticket,
}

print("Available tools:", list(tools.keys()))
```

> Say: "Tools are normal functions. The agent layer decides which function to call and with which arguments. The application code actually executes it."

---

### Step 8 - Build a tool plan

The default planner is intentionally simple and transparent.

```python
def rule_based_tool_plan(transcript: str, customer_id: str, retrieved_memories: list[dict]) -> dict:
    text = transcript.lower()

    password_issue = (
        "password" in text
        or "reset" in text
        or "log in" in text
        or "login" in text
        or "account" in text
    )

    asks_for_ticket = "ticket" in text or "support" in text or "access today" in text

    if password_issue and asks_for_ticket:
        return {
            "tool_name": "create_support_ticket",
            "arguments": {
                "customer_id": customer_id,
                "issue_type": "password_reset",
                "priority": "high",
                "summary": "Customer cannot log in because the password reset email is not arriving and needs access today.",
            },
            "requires_confirmation": True,
            "reason": "The request indicates an urgent password reset issue and asks for support action.",
        }

    if password_issue:
        return {
            "tool_name": "search_help_article",
            "arguments": {"topic": "password_reset"},
            "requires_confirmation": False,
            "reason": "The request is about a password reset issue, but no state-changing action is required yet.",
        }

    return {
        "tool_name": "search_help_article",
        "arguments": {"topic": retrieved_memories[0]["topic"] if retrieved_memories else "password_reset"},
        "requires_confirmation": False,
        "reason": "Use the nearest retrieved topic to look up guidance.",
    }


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def llm_tool_plan(transcript: str, customer_id: str, retrieved_memories: list[dict]) -> dict:
    retrieved_text = "\n".join(
        f"- {item['id']} / {item['topic']}: {item['text']}"
        for item in retrieved_memories
    )

    prompt = f"""
You are a bounded support agent planner.

Available tools:
1. search_help_article(topic: str) - read-only
2. create_support_ticket(customer_id: str, issue_type: str, priority: str, summary: str) - state-changing, requires confirmation

User transcript:
{transcript}

Customer ID:
{customer_id}

Retrieved memories:
{retrieved_text}

Return only JSON with this schema:
{{
  "tool_name": "search_help_article" or "create_support_ticket",
  "arguments": {{...}},
  "requires_confirmation": true or false,
  "reason": "short reason"
}}
""".strip()

    response = client.responses.create(
        model=AGENT_MODEL,
        input=prompt,
    )
    return extract_json_object(response.output_text)


def build_tool_plan(state: dict) -> dict:
    if USE_OPENAI_LLM:
        try:
            return llm_tool_plan(
                transcript=state["messages"][-1]["content"],
                customer_id=state["customer_id"],
                retrieved_memories=state["retrieved_memories"],
            )
        except Exception as exc:
            print("Optional LLM planner failed. Falling back to rule-based planner.")
            print(type(exc).__name__, exc)

    return rule_based_tool_plan(
        transcript=state["messages"][-1]["content"],
        customer_id=state["customer_id"],
        retrieved_memories=state["retrieved_memories"],
    )


tool_plan = build_tool_plan(conversation_state)
conversation_state["pending_action"] = tool_plan
print(json.dumps(tool_plan, indent=2))
```

> Say: "This is the planning step. It is not the same as executing the tool. Planning can be done by an LLM, rules, or a hybrid. Execution should still be controlled by application logic."

---

### Step 9 - Confirm before state-changing tool calls

```python
def execute_tool_plan(plan: dict) -> dict:
    tool_name = plan["tool_name"]
    arguments = plan.get("arguments", {})

    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")

    return tools[tool_name](**arguments)


print("Proposed action:")
print(json.dumps(tool_plan, indent=2))

if tool_plan.get("requires_confirmation"):
    print("\nThis action changes system state and requires confirmation.")
    confirmation = input("Type 'yes' to execute the tool call: ").strip().lower()
else:
    confirmation = "yes"

if confirmation == "yes":
    tool_result = execute_tool_plan(tool_plan)
    conversation_state["tool_results"].append(tool_result)
    conversation_state["pending_action"] = None
    print("Tool executed:")
    print(json.dumps(tool_result, indent=2))
else:
    tool_result = {
        "status": "cancelled",
        "reason": "User did not confirm the action.",
    }
    print("Tool execution cancelled.")
```

> Say: "This is the most important safety pattern in the demo. A transcript error should not automatically create, send, delete, or update anything."

---

### Step 10 - Generate the final response

```python
def template_agent_response(transcript: str, retrieved_memories: list[dict], tool_plan: dict, tool_result: dict) -> str:
    top_memory = retrieved_memories[0] if retrieved_memories else None

    if tool_result.get("status") == "created" and "ticket_id" in tool_result:
        return (
            f"I created support ticket {tool_result['ticket_id']} for the password reset issue. "
            f"It is marked as {tool_result['priority']} priority. "
            "The customer should check spam or email filters while support investigates the missing reset email."
        )

    if tool_result.get("tool") == "search_help_article":
        return (
            "I found the relevant help guidance. "
            f"Suggested next step: {tool_result['article']}"
        )

    if tool_result.get("status") == "cancelled":
        return "I did not execute the action because it was not confirmed. No ticket was created."

    if top_memory:
        return (
            f"The request appears related to {top_memory['topic']}. "
            "I found relevant context, but no action was completed."
        )

    return "I could not complete an action, but the request has been captured."


def llm_agent_response(transcript: str, retrieved_memories: list[dict], tool_plan: dict, tool_result: dict) -> str:
    retrieved_text = "\n".join(
        f"- {item['id']} / {item['topic']}: {item['text']}"
        for item in retrieved_memories
    )

    prompt = f"""
You are a support voice agent. Write a concise spoken response.

User transcript:
{transcript}

Retrieved memory:
{retrieved_text}

Tool plan:
{json.dumps(tool_plan, indent=2)}

Tool result:
{json.dumps(tool_result, indent=2)}

Rules:
- Be brief.
- Mention completed actions exactly.
- Do not claim anything that is not supported by the tool result.
- If no action was confirmed, say that no action was taken.
""".strip()

    response = client.responses.create(
        model=AGENT_MODEL,
        input=prompt,
    )
    return response.output_text.strip()


def build_final_response(state: dict, tool_plan: dict, tool_result: dict) -> str:
    if USE_OPENAI_LLM:
        try:
            return llm_agent_response(
                transcript=state["messages"][-1]["content"],
                retrieved_memories=state["retrieved_memories"],
                tool_plan=tool_plan,
                tool_result=tool_result,
            )
        except Exception as exc:
            print("Optional LLM response failed. Falling back to template response.")
            print(type(exc).__name__, exc)

    return template_agent_response(
        transcript=state["messages"][-1]["content"],
        retrieved_memories=state["retrieved_memories"],
        tool_plan=tool_plan,
        tool_result=tool_result,
    )


final_answer = build_final_response(conversation_state, tool_plan, tool_result)
conversation_state["messages"].append(
    {
        "role": "assistant",
        "content": final_answer,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
)

print("FINAL ANSWER:")
print(final_answer)
```

> Say: "The final answer should report what actually happened. It should not pretend that a tool succeeded if the tool was cancelled or failed."

---

### Step 11 - Convert the response to speech

This step is intentionally defensive. Some API projects do not have access to every TTS model. If OpenAI TTS returns a permission or model-access error, the demo falls back to local speech on macOS or Linux. If no speech engine is available, it prints the text response and continues.

```python
def clean_for_tts(text: str) -> str:
    text = re.sub(r"[*_`#>-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def synthesize_with_openai_tts(text: str, output_path: Path) -> Path:
    last_error = None
    for model_name in OPENAI_TTS_MODELS:
        try:
            print(f"Trying OpenAI TTS model: {model_name}")
            speech_kwargs = {
                "model": model_name,
                "voice": TTS_VOICE,
                "input": text,
                "response_format": "wav",
            }
            with client.audio.speech.with_streaming_response.create(**speech_kwargs) as response:
                response.stream_to_file(output_path)
            print("OpenAI TTS succeeded with:", model_name)
            return output_path
        except Exception as exc:
            last_error = exc
            print(f"OpenAI TTS failed for {model_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"No configured OpenAI TTS model worked. Last error: {last_error}")


def synthesize_with_local_tts(text: str, output_path: Path) -> Path | None:
    # macOS fallback: built-in 'say'. It usually outputs AIFF, then FFmpeg can convert to WAV.
    if shutil.which("say"):
        print("Using local macOS 'say' fallback.")
        subprocess.run(["say", "-o", str(TTS_AIFF_OUTPUT), text], check=True)
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(TTS_AIFF_OUTPUT), "-ar", "16000", "-ac", "1", str(output_path)],
                check=True,
                capture_output=True,
            )
            return output_path
        return TTS_AIFF_OUTPUT

    # Linux fallback if espeak is installed.
    if shutil.which("espeak"):
        print("Using local espeak fallback.")
        subprocess.run(["espeak", "-w", str(output_path), text], check=True)
        return output_path

    # Optional Python fallback if pyttsx3 is installed.
    try:
        import pyttsx3
        print("Using local pyttsx3 fallback.")
        engine = pyttsx3.init()
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        return output_path if output_path.exists() else None
    except Exception as exc:
        print("Local pyttsx3 fallback unavailable:", type(exc).__name__, exc)

    return None


def speak_or_print(text: str) -> Path | None:
    if TTS_PROVIDER not in {"auto", "openai", "local", "none"}:
        print(f"Unknown TTS_PROVIDER={TTS_PROVIDER!r}. Using auto.")

    provider = TTS_PROVIDER if TTS_PROVIDER in {"auto", "openai", "local", "none"} else "auto"

    if provider in {"auto", "openai"}:
        try:
            return synthesize_with_openai_tts(text, TTS_OUTPUT)
        except Exception as exc:
            print("OpenAI TTS unavailable for this project or configuration.")
            print(type(exc).__name__, exc)
            if provider == "openai":
                print("TTS_PROVIDER=openai, so local fallback is disabled.")
                return None

    if provider in {"auto", "local"}:
        try:
            return synthesize_with_local_tts(text, TTS_OUTPUT)
        except Exception as exc:
            print("Local TTS fallback failed.")
            print(type(exc).__name__, exc)
            return None

    return None


spoken_answer = clean_for_tts(final_answer)
if len(spoken_answer) > 450:
    spoken_answer = spoken_answer[:450].rstrip() + "."

output_audio = speak_or_print(spoken_answer)

if output_audio and output_audio.exists():
    print("Saved spoken answer to:", output_audio)
    display(Audio(str(output_audio), autoplay=False))
else:
    print("No speech output available. Showing text response only.")
    print(spoken_answer)
```

> Say: "This is an engineering point, not a model-quality point. Voice output is optional in the system architecture. If the current API project cannot access a TTS model, the agent can still complete the action and return text."

---

### Step 12 - Inspect the final state

```python
print(json.dumps(conversation_state, indent=2))
```

> Say: "A real application would log state carefully. It helps debugging, auditability, user corrections, and escalation."

---

## Optional Production Pattern - Map the Agent Loop to LangGraph

Use this after the manual Python demo if students are ready for the production pattern.

The manual demo already has the structure of a graph:

```text
transcribe
-> retrieve_memory
-> plan_action
-> confirm_action?
-> execute_tool
-> respond
-> speak
```

LangGraph lets you represent that structure directly as state, nodes, and edges.

### Conceptual mapping

| Manual demo | LangGraph version |
|---|---|
| `conversation_state` | shared graph state |
| `transcribe_or_fallback(...)` | `transcribe` node |
| embedding retrieval block | `retrieve_memory` node |
| `build_tool_plan(...)` | `plan_action` node |
| `if requires_confirmation` | conditional edge |
| `input("yes")` | human approval node, or interrupt in production |
| `execute_tool_plan(...)` | `execute_tool` node |
| `build_final_response(...)` | `respond` node |

### Minimal graph sketch

This sketch reuses the functions and variables from the manual demo. It is not required for the main run.

```python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END


class AgentGraphState(TypedDict, total=False):
    transcript: str
    retrieved_memories: list[dict]
    tool_plan: dict
    confirmed: bool
    tool_result: dict
    final_answer: str


def node_transcribe(state: AgentGraphState) -> dict:
    return {"transcript": user_transcript}


def node_retrieve_memory(state: AgentGraphState) -> dict:
    # Reuse the retrieved rows from the earlier embedding step.
    return {"retrieved_memories": retrieved.to_dict(orient="records")}


def node_plan_action(state: AgentGraphState) -> dict:
    plan = rule_based_tool_plan(
        transcript=state["transcript"],
        customer_id=conversation_state["customer_id"],
        retrieved_memories=state["retrieved_memories"],
    )
    return {"tool_plan": plan}


def route_after_plan(state: AgentGraphState) -> Literal["confirm_action", "execute_tool", "respond"]:
    if state["tool_plan"].get("requires_confirmation"):
        return "confirm_action"
    if state["tool_plan"].get("tool_name"):
        return "execute_tool"
    return "respond"


def node_confirm_action(state: AgentGraphState) -> dict:
    print("Proposed action:")
    print(json.dumps(state["tool_plan"], indent=2))
    confirmation = input("Type 'yes' to execute the tool call: ").strip().lower()
    return {"confirmed": confirmation == "yes"}


def route_after_confirmation(state: AgentGraphState) -> Literal["execute_tool", "respond"]:
    return "execute_tool" if state.get("confirmed") else "respond"


def node_execute_tool(state: AgentGraphState) -> dict:
    result = execute_tool_plan(state["tool_plan"])
    return {"tool_result": result}


def node_respond(state: AgentGraphState) -> dict:
    tool_result = state.get(
        "tool_result",
        {"status": "cancelled", "reason": "User did not confirm the action."},
    )
    answer = template_agent_response(
        transcript=state["transcript"],
        retrieved_memories=state.get("retrieved_memories", []),
        tool_plan=state["tool_plan"],
        tool_result=tool_result,
    )
    return {"tool_result": tool_result, "final_answer": answer}


builder = StateGraph(AgentGraphState)
builder.add_node("transcribe", node_transcribe)
builder.add_node("retrieve_memory", node_retrieve_memory)
builder.add_node("plan_action", node_plan_action)
builder.add_node("confirm_action", node_confirm_action)
builder.add_node("execute_tool", node_execute_tool)
builder.add_node("respond", node_respond)

builder.add_edge(START, "transcribe")
builder.add_edge("transcribe", "retrieve_memory")
builder.add_edge("retrieve_memory", "plan_action")
builder.add_conditional_edges("plan_action", route_after_plan)
builder.add_conditional_edges("confirm_action", route_after_confirmation)
builder.add_edge("execute_tool", "respond")
builder.add_edge("respond", END)

graph = builder.compile()
result_state = graph.invoke({})

print(result_state["final_answer"])
```

### Instructor explanation

Say:

> "The first demo showed the mechanics explicitly. LangGraph is the next architectural step: state is formalized, each operation becomes a node, and routing becomes explicit. This is useful once the agent has more branches, more tools, or human approval steps."

Important distinction:

```text
Manual Python demo: best for learning the core ideas.
LangGraph version: better when the workflow needs to become larger, inspectable, and production-oriented.
```

For a real approval workflow, the confirmation node can be replaced with a LangGraph interrupt so the graph pauses, keeps its state, waits for external approval, and then resumes.

---

## Student Mini Challenge

Ask students to change the fallback transcript or record a second audio file.

Suggested second request:

```text
My package is delayed and I need to know when it will arrive.
```

Students should check:

* Does transcription capture the request correctly?
* Which memory gets retrieved?
* Does the tool plan change?
* Does the agent ask for confirmation only when needed?
* Does the spoken answer reflect the actual tool result?

Bonus:

* Add a new tool called `check_order_status(order_id)`.
* Modify the planner to call it when the request contains an order number.
* Require confirmation only for actions that change data.
* Add a new LangGraph node for `check_order_status` and route delivery requests to it.

---

## If Things Go Wrong

| Problem | Fix |
|---|---|
| `.env` not loaded | Check that `.env` is in the same folder and contains `OPENAI_API_KEY` |
| Audio file missing | Use the fallback transcript; the demo still works |
| FFmpeg not found | Use the original WAV file or install FFmpeg before class |
| Transcription fails | Use fallback transcript and continue to state, retrieval, and tool calling |
| Embedding call fails | Explain retrieval conceptually and use the top memory manually |
| Optional LLM planner fails | Keep `USE_OPENAI_LLM=false`; the rule-based planner is the default path |
| `langgraph` import fails | Run `pip install --upgrade langgraph`, or skip the optional LangGraph section |
| Local TTS does not play | On macOS check that `say` is available; otherwise keep text-only output |
| LangGraph graph is confusing for students | Treat it as an architecture map, not the main demo path |
| OpenAI TTS returns 403 or model_not_found | Use `TTS_PROVIDER=auto` or `TTS_PROVIDER=local`; on macOS the demo can fall back to `say` |
| Students execute tool without confirmation | Reset `support_tickets = []` and repeat Step 9 slowly |
| Wrong tool is selected | Show that planning is not execution; improve the planner or escalate |

---

## Wrap-Up Explanation

Use this mapping at the end:

| Demo step | Agent concept |
|---|---|
| Transcribe audio | Observe |
| Store conversation state | Maintain context |
| Embed transcript | Create searchable representation |
| Retrieve memories | Ground decision in context |
| Build tool plan | Decide next action |
| Ask for confirmation | Apply safety gate |
| Execute Python function | Act |
| Generate response | Explain result |
| Speak or print response | Return output through the best available channel |
| Map to LangGraph | Structure the workflow as state, nodes, and edges |

Final takeaway:

```text
A useful agent is not just a model.
It is a controlled system around the model.
```

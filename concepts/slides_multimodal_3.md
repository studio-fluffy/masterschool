# Session - SLIDES: Talking to Agents

---

### Slide 1 - Title

**Talking to Agents**
Voice interaction, memory, tool calling, controlled execution, and LangGraph as a production pattern

---

### Slide 2 - Why Voice Agents Matter

People do not always interact with software by typing.

They may speak while driving, working, cooking, inspecting machines, or supporting customers.

Voice agents can help when a system needs to:

* understand spoken requests
* remember the current conversation
* retrieve relevant prior context
* call tools or APIs
* return spoken answers

The key point:

**A voice interface is not automatically an agent.**

---

### Slide 3 - Chatbot vs Voice Assistant vs Agent

These terms are often mixed together, but they mean different things.

| System | What it does |
|---|---|
| **Chatbot** | Responds to text messages |
| **Voice assistant** | Adds speech-to-text and text-to-speech |
| **Tool-using assistant** | Can call functions, APIs, or databases |
| **Agent** | Maintains state, retrieves context, plans actions, uses tools, and handles permissions |

A system becomes agentic when it can move from conversation to controlled action.

---

### Slide 4 - The Core Voice Agent Pipeline

A practical voice agent often follows this flow:

1. Spoken request enters the system.
2. Speech-to-text creates a transcript.
3. Conversation state records the current turn.
4. Embeddings retrieve relevant memories or documents.
5. A planner decides whether a tool call is needed.
6. The system asks for confirmation when required.
7. A tool or API is executed.
8. The result is turned into a response.
9. Text-to-speech speaks the answer.

The agent is the whole controlled system, not just the model.

---

### Slide 5 - Conversation State

A model does not automatically remember previous turns.

Conversation state stores information such as:

* conversation ID
* user or customer ID
* previous messages
* pending actions
* retrieved context
* tool results
* confirmation status

State makes follow-up turns possible.

Example:

If the user says “yes”, state tells the system what they are confirming.

---

### Slide 6 - Memory and Retrieval

Agents often need context beyond the latest message.

Memory can include:

* previous calls
* support tickets
* meeting notes
* help articles
* user preferences
* documents
* screenshots or images

Embeddings make this memory searchable.

The system can retrieve related information before deciding what to do next.

---

### Slide 7 - Transcript Embeddings for Semantic Memory

For voice agents, a common starting point is transcript-based memory.

The flow is:

```text
spoken request -> transcript -> embedding -> semantic search
```

This allows the agent to find similar prior conversations even when the wording is different.

Example:

* “I cannot access my account”
* “The password reset email never arrives”
* “I am locked out and need access today”

These may be semantically close even if the exact words differ.

---

### Slide 8 - Tool Calling

Tools are normal functions or APIs that the agent can use.

Examples:

* search a help article
* create a support ticket
* check an order status
* update a CRM record
* send a message
* schedule a meeting

Important distinction:

**Planning a tool call is not the same as executing it.**

The model or planner may propose an action, but the application should control execution.

---

### Slide 9 - Permissions and Confirmation

Tool calls can have side effects.

Some actions only read information:

* search help article
* retrieve previous ticket
* check order status

Other actions change system state:

* create a ticket
* send an email
* cancel an order
* update customer data

State-changing actions should usually require confirmation.

Example:

“Do you want me to create a high-priority support ticket for this password reset issue?”

---

### Slide 10 - Multimodal Agent Context

A voice agent does not have to rely only on speech.

It can combine:

* spoken input
* transcript history
* screenshots
* uploaded documents
* images
* previous tickets
* database records
* retrieved help articles

The important design question is:

**Which context should be retrieved and passed to the model before action?**

Too little context causes weak decisions. Too much context increases cost, latency, and confusion.

---

### Slide 11 - Error Handling and Escalation

Voice agents need explicit failure handling.

Common failure points:

* speech-to-text mishears the user
* retrieval finds the wrong memory
* the planner selects the wrong tool
* required parameters are missing
* the tool call fails
* text-to-speech model access is unavailable
* the user does not confirm the action

Robust systems should:

* ask clarification questions
* confirm risky actions
* log tool results
* avoid pretending that failed actions succeeded
* fall back to text when speech output is unavailable
* escalate when confidence is low

---

### Slide 12 - LangGraph as a Production Pattern

The demo builds the agent loop manually in Python first.

That is the right starting point for learning because every step is visible:

* state is a dictionary
* memory is a small dataset
* tools are Python functions
* routing is explicit conditional logic
* confirmation is a visible safety gate

LangGraph becomes useful when the same logic needs to become:

* stateful across many turns
* interruptible for human approval
* inspectable during execution
* resumable after failures
* easier to route across multiple branches

The key idea:

**Our manual agent loop is already a graph. LangGraph formalizes it.**

---

### Slide 13 - From Manual Loop to LangGraph

LangGraph maps the same agent design into state, nodes, and edges.

| Manual demo concept | LangGraph concept |
|---|---|
| `conversation_state` dictionary | Graph state |
| Python function | Node |
| `if/else` routing | Conditional edge |
| confirmation prompt | Human-in-the-loop / interrupt |
| tool execution function | Tool node or custom node |
| final answer builder | Response node |
| logged state | Checkpoint / trace |

The flow stays the same:

```text
observe -> retrieve -> plan -> confirm -> act -> respond
```

The benefit is not more autonomy.

The benefit is more control over a stateful agent workflow.

---

### Slide 14 - What We Build in the Demo

The practical demo builds a bounded support voice agent.

What happens step by step:

1. Load or simulate a spoken support request.
2. Transcribe the request with speech-to-text.
3. Store the transcript in conversation state.
4. Retrieve relevant memories using transcript embeddings.
5. Build a tool plan.
6. Ask for confirmation before creating a ticket.
7. Execute a Python tool if confirmed.
8. Generate a short final response.
9. Convert the response into spoken audio when TTS access is available.
10. Fall back to text if speech output is unavailable.
11. Map the same loop to LangGraph as a production pattern.

The goal is not full autonomy.

The goal is to show how controlled agent behavior emerges from state, memory, tool planning, confirmation, execution, fallback handling, and explicit workflow design.

---

### Slide 15 - Session Plan

What we cover today:

1. **Agent concepts** - chatbot, voice assistant, tool-using assistant, agent
2. **Voice pipeline** - speech input, transcript, response, spoken output
3. **State and memory** - how the system keeps context
4. **Retrieval** - finding relevant prior calls or documents
5. **Tool calling** - planning and executing Python functions
6. **Safety gates** - confirmation, permissions, and escalation
7. **Output fallback** - what happens when speech generation is unavailable
8. **LangGraph mapping** - turning the manual loop into a graph-shaped workflow
9. **Demo** - bounded voice agent for a support scenario

Suggested timing:

| Segment | Time |
|---|---:|
| Theory slides | 25 min |
| Practical demo | 30 min |
| LangGraph mapping or recap | 5 min |

---


### Key Takeaway

A useful agent is not just an LLM.

It is a controlled system that combines:

```text
speech-to-text
+ state
+ memory retrieval
+ planning
+ confirmation
+ tool execution
+ grounded response
+ optional text-to-speech
+ explicit workflow orchestration
```

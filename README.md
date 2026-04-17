# Human-in-the-Loop (HITL) Chatbot

A LangGraph-powered chatbot that implements a **Human-in-the-Loop** approval mechanism — every user question must be approved by a human operator before the AI responds.

---

## Overview

This project demonstrates how to build a safe, supervised AI chatbot using LangGraph's `interrupt` feature. Before the LLM generates a response, the workflow pauses and waits for a human to approve or reject the query. This is useful in scenarios where AI responses need oversight — such as customer service, regulated industries, or sensitive domains.

---

## How It Works

```
User sends message
        ↓
   [chat_node starts]
        ↓
INTERRUPT — Human approval required
        ↓
Human approves (yes) or rejects (no)
        ↓
Approved → LLM generates response
Rejected → Returns "Not approved."
```

---

## Project Structure

```
hitl_chatbot/
│
├── hitl_chatbot.ipynb   # Main notebook with full implementation
├── .env                 # API keys (not committed to git)
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https:
cd hitl-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install langgraph langchain-nvidia-ai-endpoints langchain-core python-dotenv
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

> Get your free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com)

---

## 🚀 Usage

Open and run `hitl_chatbot.ipynb` cell by cell.

### Step 1 — Send a message

```python
config = {"configurable": {"thread_id": "1234"}}

initial_input = {
    "messages": [("user", "Explain what is a computer in very simple terms.")]
}

result = app.invoke(initial_input, config=config)
```

### Step 2 — Human reviews the interrupt

```python
message = result['__interrupt__'][0].value
user_input = input(f"Approve this question? (y/n): ")
```

### Step 3 — Resume with decision

```python
from langgraph.types import Command

final_result = app.invoke(
    Command(resume={"approved": user_input}),
    config=config
)
```

---

##  Key Concepts

**`interrupt()`** — Pauses graph execution and surfaces data to the human operator. Requires a checkpointer to save state.

**`Command(resume=...)`** — Resumes the graph from the interrupt point with the human's decision.

**`MemorySaver`** — Stores graph state in memory so execution can be paused and resumed seamlessly.

---
## Author  
**Vishwas**  
**LinkedIn:** https://www.linkedin.com/in/vishwas-kori/

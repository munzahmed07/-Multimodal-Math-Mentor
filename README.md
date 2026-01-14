# 🧮 Reliable Multimodal Math Mentor

> An end-to-end AI tutor for JEE-style math problems, featuring Multimodal RAG, Multi-Agent Orchestration, and Self-Learning Memory.

🔗 **Live Demo:** [Click Here to Open App](https://share.streamlit.io/munzahmed07/-Multimodal-Math-Mentor/main/app.py)

---

## 🎯 Objective

Build a reliable AI system that can:

1. **See, Hear, and Read:** Accept inputs via Text, Image (OCR), and Audio (Speech-to-Text).
2. **Think:** Use a Multi-Agent Graph (Parser → Solver → Verifier) to plan and validate solutions.
3. **Learn:** Cache verified solutions to `memory.json` for instant recall on repeated queries.
4. **Interact:** Allow Human-in-the-Loop (HITL) correction of OCR and solution verification.

---

## 🚀 Key Features

### 1. Multimodal Input

* **📸 Image:** Uses **GPT-4o Vision** to extract math problems from photos/screenshots. Includes an edit step for user verification.
* **🎤 Audio:** Uses **OpenAI Whisper** to transcribe spoken math questions into text.
* **✍️ Text:** Standard text input for direct queries.

### 2. Multi-Agent Architecture (LangGraph)

The system uses a directed graph of agents to ensure accuracy:

* **🤖 Parser Agent:** Cleans input and structures the problem.
* **🧠 Solver Agent:** Retrieves formulas from the **RAG Knowledge Base (ChromaDB)** and generates a solution.
* **⚖️ Verifier Agent:** Checks the steps for logical or calculation errors.
* **🎓 Explainer Agent:** Formats the final output into a clear, student-friendly tutorial.

### 3. RAG Pipeline

* **Knowledge Base:** A curated set of math formula documents stored in `knowledge_base/`.
* **Vector Store:** **ChromaDB** with OpenAI Embeddings for semantic retrieval.
* **Lazy Loading:** Optimized for cloud deployment to prevent startup crashes.

### 4. Memory & Self-Learning

* **Caching:** Correctly solved problems (validated by user feedback ✅) are saved to `memory.json`.
* **Instant Recall:** Before triggering expensive agents, the system checks memory for similar past questions.

---

## 🛠️ Architecture

```mermaid
graph TD
    User[User Input (Text/Image/Audio)] --> MemoryCheck{In Memory?}
    MemoryCheck -- Yes --> Instant[Return Cached Answer]
    MemoryCheck -- No --> Parser[Parser Agent]
    
    subgraph "LangGraph Workflow"
    Parser --> Solver[Solver Agent + RAG]
    Solver --> Verifier[Verifier Agent]
    Verifier -- Rejected --> Solver
    Verifier -- Approved --> Explainer[Explainer Agent]
    end
    
    Explainer --> UI[Streamlit UI]
    UI -- User Feedback ✅ --> Save[Save to Memory.json]
```

---

## 💻 Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangGraph, LangChain
* **Models:** GPT-4o (Reasoning & Vision), Whisper (Audio)
* **Vector DB:** ChromaDB
* **Deployment:** Streamlit Cloud

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/munzahmed07/-Multimodal-Math-Mentor.git
cd -Multimodal-Math-Mentor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory and add your OpenAI key:

```env
OPENAI_API_KEY=sk-proj-your-key-here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
📂 math-mentor
├── 📂 src
│   ├── agents.py       # LangGraph Agent logic (Parser, Solver, Verifier)
│   ├── rag.py          # RAG pipeline & Vector DB initialization
│   └── utils.py        # Helper functions for OCR & Audio transcription
├── 📂 knowledge_base   # Text files containing math formulas
├── 📂 chroma_db        # Vector Database (Generated automatically)
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── memory.json         # Self-learning storage
```

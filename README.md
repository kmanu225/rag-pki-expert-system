# 🔐 PKI Expert System with RAG

## 🎯 Goals

Develop an expert system for **Public Key Infrastructure (PKI)** using a **text-based knowledge base** and **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware responses.

---

## ✨ Features

* **🔍 Retrieval-Augmented Generation (RAG)**
  Leverages vector search via **ChromaDB** and text generation using open models from **Hugging Face** or **OpenRouter**.

* **📚 Automated Knowledge Processing**
  Automatically chunks and indexes documents from the `knowledge/` folder for efficient and fast retrieval.

* **💬 Interactive Dialog Interface**
  Supports both **command-line** and **Gradio-based** user interfaces for interactive expert consultation.

* **🧩 Modular Architecture**
  Components like the LLM, vector store, and prompt logic are modular and easily replaceable.

---

## 🗂️ Project Structure

```plaintext
.
├── Dockerfile                     # Docker container setup
├── LICENSE                        # License file
├── README.md                      # Project documentation
├── db/
│   ├── chroma_db/                 # ChromaDB local vector database files
│   ├── docker-compose.yml         # Docker configuration for ChromaDB
│   └── otel-collector-config.yaml # OpenTelemetry collector config (if used for observability)
├── docs/
│   └── architecture.png           # System architecture diagram
├── requirements.txt               # Python dependencies
├── src/
│   ├── __pycache__/               # Python cache files (can be ignored)
│   ├── app.py                     # Main application entry point
│   ├── embedding_storage.py       # Embedding storage & retrieval logic
│   ├── media_to_text.py           # (Optional) Convert media (audio/video) to text for indexing
│   ├── rag.py                     # Core RAG implementation
│   └── knowledge/                 # Directory for source documents (.txt, .pdf, etc.)
└── test/
    ├── demo.ipynb                 # Jupyter notebook for testing & exploration
    └── uploads/                   # Folder for uploaded files (used in demos/tests)

```

---

## 🧱 System Architecture

![System Architecture](docs/architecture.png)

---

## ⚙️ Prerequisites

* Python **3.10.12+**
* [OpenRouter](https://openrouter.ai/) API key
* [Hugging Face](https://huggingface.co/) API token
* Docker

---

## 🚀 Getting Started

### 1. Update Your System

```bash
sudo apt update
```

### 2. Build and Run the Docker Vector Database

```bash
cd db
docker-compose up --build
```

### 3. Set Environment Variables

```bash
export OPENROUTER_API_KEY=your_openrouter_api_key
export HF_TOKEN=your_huggingface_token
```

### 4. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the app

```bash
python3 src/app.py
```


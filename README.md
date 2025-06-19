# 🔍 ai-llm-rag-system

This repository is dedicated to experimenting with and testing various **RAG (Retrieval-Augmented Generation)** architectures using modern **LLMs** and vector databases. It provides a modular framework for building, evaluating, and interacting with systems that combine local knowledge sources and generative models.

---

## 📦 Features

- 🔗 **Retrieval-Augmented Generation (RAG)**: Integrates vector search (via ChromaDB) with text generation (Hugging Face Transformers).
- 🧠 **LLM Integration**: Utilizes models like `google/gemma-2-2b-it` for high-quality generation.
- 🗂️ **Long-term Knowledge Handling**: Automatically chunks and indexes documents from a local folder.
- 💬 **Interactive Dialog Interface**: Terminal-based assistant that retrieves relevant context and generates informed answers.
- 🛠️ **Modular Design**: Easily swap models, vector DBs, or prompt templates.

---

## 🧰 Tech Stack

- 🤗 **LLM Transformers** — Text generation pipeline.
- 🧠 **ChromaDB** — Lightweight in-memory vector database for similarity search.
- 🧩 **LangChain** — Used for efficient and intelligent text chunking.
- 🐍 **Python** — Main development language.
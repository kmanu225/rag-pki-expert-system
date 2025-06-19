# Goals

Build a PKI expert system using text based knowledge for RAG context.


# Features

- 🔗 **Retrieval-Augmented Generation (RAG)**: Integrates vector search (via ChromaDB) with text generation (Transformers available on Hugging Face for free).
- 🧠 **LLM Integration**: Utilizes models like `google/gemma-2-2b-it` for high-quality generation.
- 🗂️ **Long-term Knowledge Handling**: Automatically chunks and indexes documents from a local folder.
- 💬 **Interactive Dialog Interface**: Terminal-based assistant that retrieves relevant context and generates informed answers.
- 🛠️ **Modular Design**: Easily swap models, vector DBs, or prompt templates.

# Architecture

```perl
.
│
├── knowledge/                  # Folder with source documents (.txt)
├── llm_loader.py               # LLM loading logic
├── embedder.py                 # Text chunking functions
├── vector_store.py             # ChromaDB integration
├── query.py           # Interactive assistant logic
├── app.py                      # Example entry point
└── README.md
```
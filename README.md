# Goals

Build a PKI expert system using text based knowledge for RAG context.


# Features

- 🔗 **Retrieval-Augmented Generation (RAG)**: Integrates vector search (via ChromaDB) with text generation (Transformers available on Hugging Face for free).
- 🧠 **LLM Integration**: Utilizes models like `google/gemma-2-2b-it` for high-quality generation.
- 🗂️ **Long-term Knowledge Handling**: Automatically chunks and indexes documents from a local folder.
- 💬 **Interactive Dialog Interface**: Terminal-based assistant that retrieves relevant context and generates informed answers.
- 🛠️ **Modular Design**: Easily swap models, vector DBs, or prompt templates.

# Project Structure

```perl
.
│
├── knowledge/                 # Folder containing source documents (e.g., .txt files)
├── llm_loader.py              # Logic for loading the language model (LLM)
├── vector_loader.py           # Text chunking and ChromaDB integration
├── test.py                    # Script or module for testing
├── gr_gui.py                  # Gradio-based graphical user interface
├── dialogue_mngr.py           # Functions for question-answer interaction
├── query.py                   # Logic for handling interactive assistant queries
├── rag.py                     # RAG (Retrieval-Augmented Generation) core implementation
├── main.py                    # Entry point of the application
└── README.md                  # Project documentation

```
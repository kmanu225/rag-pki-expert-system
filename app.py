from llm_loader import load_llm
from embedder import load_and_split_texts
from vector_db import store_passages_in_chromadb
from dialog_manager import interactive_dialog

hf_token = "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"
model = "google/gemma-2-2b-it"
llm_pipeline = load_llm(model_name=model, hf_token=hf_token)

knowledge_folder = "knowledge"
passages = load_and_split_texts(knowledge_folder)
collection = store_passages_in_chromadb(passages)

interactive_dialog(collection, llm_pipeline)
from llm_loader import load_llm
from vector_loader import load_and_split_texts, store_passages_in_chromadb


hf_token = "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"
model = "google/gemma-2-2b-it"
llm_pipeline = load_llm(model_name=model, hf_token=hf_token)

knowledge_folder = "ai-system/knowledge"
passages = load_and_split_texts(knowledge_folder)
collection = store_passages_in_chromadb(passages)
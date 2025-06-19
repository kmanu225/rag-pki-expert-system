from llm_loader import load_llm
from vector_loader import load_and_split_texts, store_passages_in_chromadb


model = "google/gemma-2-2b-it"
llm_pipeline = load_llm(model_name=model)

knowledge_folder = "src/knowledge"
passages = load_and_split_texts(knowledge_folder)
collection = store_passages_in_chromadb(passages)

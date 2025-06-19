from llm_loader import load_llm
from vector_loader import load_and_split_texts, store_passages_in_chromadb
from dialogue_mngr import query


if __name__ == "__main__":

    hf_token = "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"
    model = "google/gemma-2-2b-it"
    llm_pipeline = load_llm(model_name=model, hf_token=hf_token)

    # Example prompt
    response = llm_pipeline("What is the capital of France?", max_new_tokens=20)
    print(response)

    knowledge_folder = "knowledge"
    passages = load_and_split_texts(knowledge_folder)
    collection = store_passages_in_chromadb(passages)
    print(f"Loaded {len(passages)} passages.")

    user_question = (
        "Generate a questionnaire for my job interview on PKI and certificate."
    )
    response = query(user_question, collection, llm_pipeline)
    print(response)

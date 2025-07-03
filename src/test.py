from app import query_local_hgf_llm, query_openrouter_llm
from time import time


if __name__ == "__main__":

    user_question = "How PKI helps protect certificates ?"
    print(f"User question: {user_question}")

    print("Querying OpenRouter LLM...")
    start_time = time()
    print(query_openrouter_llm(user_question, None))
    print(f"OpenRouter LLM response time: {time() - start_time:.2f} seconds")

    print("Querying local Hugging Face LLM...")
    start_time = time()
    print(query_local_hgf_llm(user_question, None))
    print(f"Local Hugging Face LLM response time: {time() - start_time:.2f} seconds")

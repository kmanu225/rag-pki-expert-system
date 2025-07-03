from llm_loader import hg_face
from embedding_storage import load_and_split_texts, chromadb_storage


llm_pipeline = hg_face(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct")

knowledge_base = "src/knowledge"
passages = load_and_split_texts(knowledge_base)
collection = chromadb_storage(passages)


def context_retrieval(user_question: str, top_k: int = 3, max_new_tokens: int = 256):
    """
    Retrieves relevant context documents for a user's question from the ChromaDB collection.

    Args:
        user_question (str): The user's input question.
        top_k (int, optional): Number of top relevant documents to retrieve. Defaults to 3.
        max_new_tokens (int, optional): (Unused in this function; can be removed or documented if used elsewhere.)

    Returns:
        str: A formatted string containing the top retrieved context documents.
    """
    print(f"Retrieving context for question: {user_question}")
    results = collection.query(query_texts=[user_question], n_results=top_k)
    documents = results["documents"][0]
    context = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    return context


def generate_response(user_question, context, max_new_tokens: int = 256):
    """
    Generates a response to the user's question based on the provided context using a text-generation pipeline.

    Args:
        user_question (str): The user's input question.
        context (str): The context information retrieved for the question.
        max_new_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 256.

    Returns:
        str: The assistant's generated response, or a fallback message if the answer is not in the context.
    """
    prompt_template = """You are a helpful and knowledgeable assistant who is expert in cybersecurity. Please answer the user's question based on the context provided below.

If the answer is not present in the context, respond with:
"I could not find the answer based on the context you provided."

Answer in a clear, concise, and informative manner.

User Question:
{}

Context:
{}"""

    prompt = prompt_template.format(user_question, context)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted prompt: {formatted_prompt}")

    outputs = llm_pipeline(
        formatted_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1
    )
    response = outputs[0]["generated_text"][len(formatted_prompt) :].strip()
    return response

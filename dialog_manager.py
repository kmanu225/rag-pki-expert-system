
def dialog_manager(
    user_question: str, collection, pipeline, top_k: int = 3, max_new_tokens: int = 256
):
    """
    Handles a user query by retrieving relevant context from a ChromaDB collection
    and generating a response using a text-generation pipeline.

    Args:
        user_question (str): The user's input question.
        collection: The ChromaDB collection for retrieving relevant documents.
        pipeline: Hugging Face text-generation pipeline.
        top_k (int): Number of top documents to retrieve from ChromaDB.
        max_new_tokens (int): Max tokens to generate in the response.

    Returns:
        str: The assistant's generated response.
    """
    # Prompt template
    prompt_template = """You are a helpful and knowledgeable assistant. Please answer the user's question based on the context provided below.

If the answer is not present in the context, respond with:
"I could not find the answer based on the context you provided."

Answer in a clear, concise, and informative manner.

User Question:
{}

Context:
{}
"""

    # Retrieve relevant context
    results = collection.query(query_texts=[user_question], n_results=top_k)
    documents = results["documents"][0]

    # Format context
    context = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

    # Build the prompt
    prompt = prompt_template.format(user_question, context)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    outputs = pipeline(
        formatted_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1
    )

    # Extract and return the generated text
    response = outputs[0]["generated_text"][len(formatted_prompt) :].strip()
    return response


def interactive_dialog(collection, pipeline, top_k: int = 3, max_new_tokens: int = 256):
    """
    Launches an interactive loop where a user can ask questions,
    and the assistant answers using retrieved context + LLM generation.

    Args:
        collection: ChromaDB collection for context retrieval.
        pipeline: Hugging Face text-generation pipeline.
        top_k (int): Number of top documents to retrieve.
        max_new_tokens (int): Number of tokens to generate per response.
    """

    prompt_template = """You are a helpful and knowledgeable assistant. Please answer the user's question based on the context provided below.

If the answer is not present in the context, respond with:
"I could not find the answer based on the context you provided."

Answer in a clear, concise, and informative manner.

User Question:
{}

Context:
{}
"""

    print("💬 Interactive RAG Assistant — Ask me anything (type 'exit' to quit)\n")

    while True:
        user_question = input("🧑 You:  ")
        if user_question.strip().lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # Retrieve relevant documents
        results = collection.query(query_texts=[user_question], n_results=top_k)
        documents = results["documents"][0]

        # Format context
        context = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])

        # Construct prompt
        full_prompt = prompt_template.format(user_question, context)
        messages = [{"role": "user", "content": full_prompt}]
        formatted_prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        outputs = pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
        )

        assistant_reply = outputs[0]["generated_text"][len(formatted_prompt) :].strip()
        print(f"🤖 Assistant: {assistant_reply}\n")


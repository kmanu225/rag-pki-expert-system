from embedding_storage import get_remote_contextual_knowledge
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
import requests
import torch
import json
import os

collection = get_remote_contextual_knowledge()
prompt_template = """You are a helpful and knowledgeable assistant who is expert in cybersecurity. Please answer the user's question based on the context provided below. Answer in a clear, concise, and informative manner. If the answer is not present in the context, respond with: "I could not find the answer based on the context you provided."
    
User Question:
{}
    
Context:
{}"""


def context_retrieval(user_question: str, top_k: int = 3):
    """
    Retrieves relevant context documents for a user's question from the ChromaDB collection.

    Args:
        user_question (str): The user's input question.
        top_k (int, optional): Number of top relevant documents to retrieve. Defaults to 3.
        max_new_tokens (int, optional): (Unused in this function; can be removed or documented if used elsewhere.)

    Returns:
        str: A formatted string containing the top retrieved context documents.
    """
    results = collection.query(query_texts=[user_question], n_results=top_k)
    documents = results["documents"][0]
    context = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
    return context


def hgf_llm(model_name: str, torch_dtype=torch.bfloat16):
    """
    Loads a text-generation pipeline using a specified Hugging Face model and token.

    Args:
        model_name (str): The model identifier from Hugging Face (e.g., "google/gemma-2-2b-it").
        hf_token (str): Hugging Face access token.
        torch_dtype: Desired PyTorch data type for the model (default: torch.bfloat16).

    Returns:
        pipeline: A Hugging Face text-generation pipeline.
    """
    # Log in to Hugging Face
    login(token=os.getenv("HF_TOKEN"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model and create pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        model_kwargs={
            "torch_dtype": torch_dtype,
        },
    )

    return text_pipeline


def hgf_generator(user_question, context, max_new_tokens: int = 256):
    """
    Generates a response to the user's question based on the provided context using a text-generation pipeline.

    Args:
        user_question (str): The user's input question.
        context (str): The context information retrieved for the question.
        max_new_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 256.

    Returns:
        str: The assistant's generated response, or a fallback message if the answer is not in the context.
    """

    HGF_MODEL = "google/gemma-2b-it"
    pipeline = hgf_llm(model_name=HGF_MODEL)

    prompt = prompt_template.format(user_question, context)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipeline(
        formatted_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1
    )
    response = outputs[0]["generated_text"][len(formatted_prompt) :].strip()
    return response


def openrouter_generator(user_question, context):
    """
    Generates a response to the user's question based on the provided context using OpenRouter API.

    Args:
        user_question (str): The user's input question.
        context (str): The context information retrieved for the question.
        max_new_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 256.

    Returns:
        str: The assistant's generated response.
    """

    prompt = prompt_template.format(user_question, context)
    api_key = os.getenv("OPENROUTER_API_KEY")
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
        },
        data=json.dumps(
            {
                "model": "openrouter/cypher-alpha:free",  # Optional
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )
    if response.status_code != 200:
        raise Exception(
            f"OpenRouter API request failed: {response.status_code} {response.reason}"
        )
    return response.json()["choices"][0]["message"]["content"].strip()



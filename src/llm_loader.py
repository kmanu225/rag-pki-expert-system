from transformers import AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import os


def hg_face(model_name: str, torch_dtype=torch.bfloat16):
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

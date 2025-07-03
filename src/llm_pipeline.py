from transformers import AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import os
import requests
import json


class OpenRouterPipeline:
    """
    Simple text-generation pipeline for OpenRouter models.
    Mimics Hugging Face's pipeline interface.
    """

    def __init__(self, model_name, api_key=None):
        """
        Args:
            model_name (str): The OpenRouter model identifier (e.g., "openai/gpt-4o").
            api_key (str, optional): OpenRouter API key. If not provided, uses the OPENROUTER_API_KEY environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided via argument or OPENROUTER_API_KEY env variable."
            )
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __call__(self, prompt, max_new_tokens=256, temperature=0.1):
        """
        Generates text using the OpenRouter API.

        Args:
            prompt (str or list): User prompt as a string or list of messages.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated response text.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be a string or a list of messages.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        response = requests.post(
            self.endpoint, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]



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

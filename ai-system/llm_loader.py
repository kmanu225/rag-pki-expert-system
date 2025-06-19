from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import login
import torch

def load_llm(model_name: str, hf_token: str, use_4bit=True):
    """
    Loads a quantized and device-optimized text-generation pipeline using Hugging Face Transformers.

    Args:
        model_name (str): The Hugging Face model ID (e.g., "google/gemma-2-2b-it").
        hf_token (str): Hugging Face access token.
        use_4bit (bool): Whether to load the model in 4-bit precision using bitsandbytes.

    Returns:
        pipeline: Hugging Face text-generation pipeline.
    """
    # Log in to Hugging Face
    login(token=hf_token)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # BitsAndBytes configuration for memory efficiency
    quant_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    # Load model with appropriate quantization and device map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # lets accelerate pick the best device(s)
        torch_dtype=torch.float16 if use_4bit else torch.bfloat16,
        quantization_config=quant_config,
    )

    # Build text-generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    return text_pipeline

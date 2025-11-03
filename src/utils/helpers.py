# src/utils/helpers.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
import yaml


def load_config(config_path: str) -> dict:
    """Loads and returns the content of a YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_model_and_tokenizer(model_name: str, quant_config: dict, torch_dtype: str = "bfloat16"):
    """
    Loads the quantized base model and its corresponding tokenizer.

    Args:
        model_name (str): The name of the model on the Hugging Face Hub.
        quant_config (dict): A dictionary containing the BitsAndBytes configuration.
        torch_dtype (str): The computation data type ('bfloat16' or 'float16').

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading base model: {model_name}...")

    compute_dtype = getattr(torch, torch_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', False),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print("-> Successfully loaded model and tokenizer.")
    return model, tokenizer


def get_vietnamese_medical_qa_dataset(test_size: float, seed: int) -> DatasetDict:
    """
    Loads and splits the 'hungnm/vietnamese-medical-qa' dataset.

    Returns:
        DatasetDict: Contains 'train' and 'validation' splits.
    """
    print("Loading dataset 'hungnm/vietnamese-medical-qa'...")
    dataset = load_dataset("hungnm/vietnamese-medical-qa", split="train")
    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)

    print(
        f"-> Successfully split dataset: {len(train_test_split['train'])} train, {len(train_test_split['test'])} validation.")
    return train_test_split

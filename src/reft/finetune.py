import os
import torch
import yaml
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    peft_config = config['peft']
    quant_config = config['quantization']

    print("--- Loading and processing DPO dataset ---")
    dataset = load_dataset("json", data_files=data_config['paths']['reft_dpo_dataset'], split="train")
    train_test_split = dataset.train_test_split(test_size=0.1, seed=data_config['seed'])
    train_data = train_test_split['train']
    eval_data = train_test_split['test']
    print(f"-> Dataset split: {len(train_data)} train samples, {len(eval_data)} eval samples.")

    print("--- Configuring and loading model ---")
    compute_dtype = getattr(torch, model_config['torch_dtype'])

    bnb_config = BitsAndBytesConfig(**quant_config,
                                    bnb_4bit_compute_dtype=compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("--- Initializing DPOTrainer ---")
    dpo_args = DPOConfig(**training_config)
    lora_config = LoraConfig(**peft_config)

    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    print("--- Starting Fine-tuning with DPO ---")
    trainer.train()

    output_path = os.path.join("src/models", model_config['new_adapter_name'])
    print(f"--- Saving LoRA adapter at: {output_path} ---")
    os.makedirs(output_path, exist_ok=True)
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/reft_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

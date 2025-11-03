# src/tvaft/finetune.py

import os
import torch
import yaml
import argparse
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trainer import TVAFTTrainer  # Import custom trainer


def preprocess_for_tvaft_trainer(batch, tokenizer, max_length):
    """
    Preprocessing function for TVAFTTrainer: tokenizes, creates labels, and pads/truncates saliency weights.
    """
    model_inputs = tokenizer(
        batch["text_for_training"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    labels = torch.tensor(model_inputs["input_ids"]).clone()
    processed_saliency_weights = []

    for i in range(len(batch["text_for_training"])):
        prompt_tokens = tokenizer.encode(batch["prompt_for_masking"][i], add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        labels[i, :prompt_len] = -100

        saliency = [0.0] * max_length
        original_weights = batch["saliency_weights"][i]

        completion_len = len(original_weights)
        end_idx = prompt_len + completion_len

        if end_idx <= max_length:
            saliency[prompt_len: end_idx] = original_weights
        else:
            saliency[prompt_len:] = original_weights[:max_length - prompt_len]

        processed_saliency_weights.append(saliency)

    model_inputs["labels"] = labels
    model_inputs["saliency_weights"] = torch.tensor(processed_saliency_weights, dtype=torch.float)
    model_inputs["is_correct"] = torch.tensor(batch["is_correct"], dtype=torch.bool)

    return model_inputs


def main(config_path: str):
    print("--- STARTING STEP 5: FINE-TUNING WITH TVAFT ---")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    peft_config = config['peft']
    quant_config = config['quantization']

    # Configure Quantization and Model
    print("--- Configuring and loading model ---")
    compute_dtype = getattr(torch, model_config['torch_dtype'])
    bnb_config = BitsAndBytesConfig(**quant_config, bnb_4bit_compute_dtype=compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT/TVAFT typically uses right padding
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure PEFT (LoRA)
    lora_config = LoraConfig(**peft_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load the final TVAFT dataset
    print("Loading the processed TVAFT dataset...")
    dataset = load_dataset("json", data_files=data_config['paths']['tvaft_final_dataset'], split="train")
    train_test_split = dataset.train_test_split(test_size=0.1, seed=data_config['seed'])

    # Apply the preprocessing function
    print("Preprocessing dataset for TVAFT Trainer...")
    tokenized_train_dataset = train_test_split['train'].map(
        lambda x: preprocess_for_tvaft_trainer(x, tokenizer, data_config['max_seq_length']),
        batched=True,
        remove_columns=train_test_split['train'].column_names
    )
    tokenized_validation_dataset = train_test_split['test'].map(
        lambda x: preprocess_for_tvaft_trainer(x, tokenizer, data_config['max_seq_length']),
        batched=True,
        remove_columns=train_test_split['test'].column_names
    )

    # Initialize TVAFTTrainer
    training_args = TrainingArguments(**training_config)
    trainer = TVAFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    print("--- Starting training with TVAFT Trainer ---")
    trainer.train()

    # Save the model
    output_model_path = os.path.join("src/models", model_config['new_model_name'])
    print(f"--- Saving model and tokenizer at: {output_model_path} ---")
    os.makedirs(output_model_path, exist_ok=True)
    trainer.model.save_pretrained(output_model_path)
    trainer.tokenizer.save_pretrained(output_model_path)
    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5 of TVAFT Pipeline: Fine-tuning with a Custom Trainer.")
    parser.add_argument("--config", type=str, default="src/configs/tvaft_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

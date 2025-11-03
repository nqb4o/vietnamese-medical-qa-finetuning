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
from trl import SFTTrainer


def preprocess_function(examples, prompt_template):
    """Preprocessing function, creates a 'text' column from 'question' and 'answer'."""
    prompt = prompt_template.format(instruction=examples['question'])
    examples["text"] = f"{prompt} {examples['answer']}"
    return examples


def main(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    peft_config = config['peft']
    quant_config = config['quantization']

    print("--- Loading and processing dataset ---")
    dataset = load_dataset(data_config['dataset_name'], split="train")
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_splits = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    prompt_template = data_config['prompt_template']
    train_dataset = dataset_splits['train'].map(
        lambda x: preprocess_function(x, prompt_template),
        remove_columns=list(dataset_splits['train'].features)
    )
    validation_dataset = dataset_splits['validation'].map(
        lambda x: preprocess_function(x, prompt_template),
        remove_columns=list(dataset_splits['validation'].features)
    )

    print("--- Configuring and loading model ---")
    compute_dtype = getattr(torch, model_config['torch_dtype'])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(**peft_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(**training_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=lora_config,
        dataset_text_field=data_config['text_field'],
        max_seq_length=data_config['max_seq_length'],
        tokenizer=tokenizer,
    )

    print("--- Starting Supervised Fine-tuning ---")
    trainer.train()

    output_model_path = os.path.join("src/models", model_config['new_model_name'])
    print(f"--- Saving model and tokenizer at: {output_model_path} ---")
    os.makedirs(output_model_path, exist_ok=True)
    trainer.model.save_pretrained(output_model_path)
    trainer.tokenizer.save_pretrained(output_model_path)
    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

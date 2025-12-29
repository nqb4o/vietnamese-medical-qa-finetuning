# src/tvaft/01_generate_responses.py

import os
import torch
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm
import argparse
import yaml


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")

    inputs_on_device = input_ids["input_ids"].to(model.device)
    attention_mask_on_device = input_ids["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs_on_device,
            attention_mask=attention_mask_on_device,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    try:
        # Split the answer after the prompt tag
        answer_part = full_response.split("### Trả lời:")[1].strip()
    except IndexError:
        answer_part = ""  # Handle cases with incorrect formatting
    return answer_part


def main(config_path: str):
    print("--- STARTING STEP 1: GENERATE RESPONSES FROM BASE MODEL ---")

    # 1. Load configuration file
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model_config = config['model']
    data_config = config['data']
    quant_config = config['quantization']

    # 2. Configure and load Model & Tokenizer
    print(f"Loading base model: {model_config['base_model_name']}...")
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

    tokenizer = AutoTokenizer.from_pretrained(model_config['base_model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. Load and prepare the dataset
    print(f"Loading dataset: {data_config['dataset_name']}...")
    dataset = load_dataset(data_config['dataset_name'], split="train")
    train_test_split = dataset.train_test_split(test_size=data_config['test_size'], seed=data_config['seed'])
    train_subset = train_test_split['train'].select(range(data_config['max_samples']))
    print(f"Processing {len(train_subset)} samples.")

    # 4. Generate responses
    responses = []
    for sample in tqdm(train_subset, desc="Generating responses from model"):
        prompt = data_config['prompt_template'].format(instruction=sample['question'])
        response = generate_response(model, tokenizer, prompt)
        responses.append(response)

    # 5. Save results to a CSV file
    output_path = data_config['paths']['tvaft_01_model_responses']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({'model_response': responses})
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n Done! Saved {len(responses)} responses to file: {output_path}")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="Step 1 of TVAFT Pipeline: Generate responses from the base model.")
    parser.add_argument("--config", type=str, default="src/configs/base_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

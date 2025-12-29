import pandas as pd
import json
import yaml
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import argparse


def create_dpo_dataset(config):
    """
    Loads, processes, and creates a dataset file for DPO from CSV files.
    """
    print("--- STARTING DPO DATASET CREATION PROCESS ---")
    data_config = config["data"]
    max_samples = data_config['max_samples']

    try:
        print(f"Step 1/4: Loading the original dataset '{data_config['dataset_name']}'...")
        full_dataset = load_dataset(data_config['dataset_name'], split="train")
        train_test_split = full_dataset.train_test_split(test_size=0.1, seed=data_config['seed'])
        train_subset = train_test_split['train'].select(range(max_samples))
        questions_list = [item['question'] for item in train_subset]
        print(f"-> Successfully retrieved {len(questions_list)} questions.")

        print(f"Step 2/4: Reading 'chosen' file from '{data_config['paths']['judgments_path']}'...")
        judgments_df = pd.read_csv(data_config['paths']['judgments_path'])
        chosen_responses_list = judgments_df['judgment'].values.tolist()
        print(f"-> Successfully read {len(chosen_responses_list)} 'chosen' responses.")

        print(f"Step 3/4: Reading 'rejected' file from '{data_config['paths']['model_responses_path']}'...")
        model_responses_df = pd.read_csv(data_config['paths']['model_responses_path'])
        rejected_responses_list = model_responses_df['model_response'].values.tolist()
        print(f"-> Successfully read {len(rejected_responses_list)} 'rejected' responses.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during data loading: {e}")
        return

    print("\nStep 4/4: Validating and processing data...")
    if not (len(questions_list) == len(chosen_responses_list) == len(rejected_responses_list) == max_samples):
        print("\n[ERROR] The number of rows in the input files do not match!")
        return
    print("-> Validation successful.")

    dpo_data = []
    skipped_count = 0
    prompt_template = config['data']['prompt_template']

    for question, chosen, rejected in tqdm(zip(questions_list, chosen_responses_list, rejected_responses_list),
                                           total=max_samples, desc="Processing data pairs"):
        prompt = prompt_template.format(instruction=str(question).strip())
        chosen_response = str(chosen).strip()
        rejected_response = str(rejected).strip()

        if question and chosen_response and rejected_response:
            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            })
        else:
            skipped_count += 1

    output_path = data_config['paths']['reft_dpo_dataset']
    print(f"\nSaving results to file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n--- COMPLETE ---")
    print(f"Successfully created file '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/reft_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    create_dpo_dataset(config)

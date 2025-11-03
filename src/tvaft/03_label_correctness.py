# src/tvaft/03_label_correctness.py

import torch
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import argparse
import yaml


def main(config_path: str):
    print("--- STARTING STEP 3: ASSIGN 'IS_CORRECT' LABELS USING SENTENCE-BERT ---")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    data_config = config['data']

    # Semantic similarity threshold
    SIMILARITY_THRESHOLD = 0.7

    # Load Sentence Transformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    bert_model_name = 'keepitreal/vietnamese-sbert'
    print(f"Loading Sentence Transformer model: {bert_model_name}")
    bert_model = SentenceTransformer(bert_model_name, device=device)

    # Load necessary data
    print("Loading original data and model responses...")
    dataset = load_dataset(data_config['dataset_name'], split="train")
    train_test_split = dataset.train_test_split(test_size=data_config['test_size'], seed=data_config['seed'])
    train_subset = train_test_split['train'].select(range(data_config['max_samples']))
    standard_answers = [sample['answer'] for sample in train_subset]

    model_responses_df = pd.read_csv(data_config['paths']['tvaft_01_model_responses'])
    model_responses = model_responses_df['model_response'].values.tolist()

    assert len(standard_answers) == len(model_responses), "Data mismatch!"

    # Calculate similarity and assign labels
    print("Calculating similarity and assigning labels...")

    standard_embeddings = bert_model.encode(standard_answers, convert_to_tensor=True, show_progress_bar=True,
                                            device=device)
    model_embeddings = bert_model.encode(model_responses, convert_to_tensor=True, show_progress_bar=True, device=device)

    # Calculate cosine similarity for each pair
    cosine_scores = util.pytorch_cos_sim(standard_embeddings, model_embeddings)
    is_correct_labels = [cosine_scores[i][i].item() >= SIMILARITY_THRESHOLD for i in range(len(model_responses))]

    # Save results
    output_path = data_config['paths']['tvaft_03_bert_labels']
    df = pd.DataFrame({
        'model_response': model_responses,
        'standard_answer': standard_answers,
        'is_correct': is_correct_labels
    })
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    correct_count = sum(is_correct_labels)
    total_count = len(is_correct_labels)
    print(f"\n--- Labeling Statistics ---")
    print(f"Total samples: {total_count}")
    print(f"Number of 'Correct' (True) labels: {correct_count} ({correct_count / total_count:.2%})")
    print(f"Number of 'Incorrect' (False) labels: {total_count - correct_count}")
    print(f"\n✅ Done! Saved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3 of TVAFT Pipeline: Assign is_correct labels using Sentence-BERT.")
    parser.add_argument("--config", type=str, default="src/configs/base_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

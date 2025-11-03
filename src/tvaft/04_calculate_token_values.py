# src/tvaft/04_calculate_token_values.py

import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import json
import warnings
import yaml
import argparse

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_token_probabilities(model, tokenizer, context_prompt, target_token_ids, epsilon):
    """
    Calculates the probability of each token in target_token_ids given the context_prompt.

    Args:
        model: The Hugging Face causal language model.
        tokenizer: The tokenizer for the model.
        context_prompt (str): The preceding text context.
        target_token_ids (list[int]): The list of token IDs to calculate probabilities for.
        epsilon (float): A small constant to avoid division by zero.

    Returns:
        list[float]: A list of probabilities corresponding to each target token.
    """
    model.eval()
    context_inputs = tokenizer(context_prompt, return_tensors="pt", add_special_tokens=True)

    if not isinstance(target_token_ids, torch.Tensor):
        target_token_ids = torch.tensor(target_token_ids, dtype=torch.long)

    full_input_ids = torch.cat([context_inputs.input_ids[0].cpu(), target_token_ids]).unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids)
        logits = outputs.logits

    context_len = context_inputs.input_ids.shape[1]
    # Extract logits corresponding to the position of each target token
    relevant_logits = logits[0, context_len - 1: context_len + len(target_token_ids) - 1, :]
    probs = torch.softmax(relevant_logits, dim=-1)

    target_token_ids_tensor = torch.tensor(target_token_ids, dtype=torch.long, device=model.device).unsqueeze(1)
    # Gather the probabilities of the specific target tokens
    token_probs = probs.gather(dim=1, index=target_token_ids_tensor).squeeze(1).tolist()
    return token_probs


def smooth_scale_weights(weights, min_val, max_val, scale=1.0):
    """
    Applies a tanh function to smoothly scale weights into a defined range.

    Args:
        weights (list[float]): The input saliency weights.
        min_val (float): The minimum value of the target range.
        max_val (float): The maximum value of the target range.
        scale (float): The scaling factor for the tanh function.

    Returns:
        list[float]: The list of scaled and smoothed weights.
    """
    mean_val = (max_val + min_val) / 2
    amplitude = (max_val - min_val) / 2

    weights_np = np.array(weights, dtype=np.float32)

    return (mean_val + amplitude * np.tanh((weights_np - 1.0) / scale)).tolist()


def main(config_path: str):
    """
    Main function to execute the token value calculation pipeline.
    """
    print("--- STARTING STEP 4: CALCULATE TOKEN VALUES (SALIENCY WEIGHTS) ---")

    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    model_cfg = config['model']
    data_cfg = config['data']
    tvaft_cfg = config['tvaft_params']
    EPSILON = tvaft_cfg['epsilon']

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model and tokenizer from '{model_cfg['name']}'...")

    # Determine compute dtype based on hardware support and config
    compute_dtype = getattr(torch, model_cfg['torch_dtype'])
    if model_cfg['torch_dtype'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        print("Warning: bfloat16 is not supported on this device. Falling back to float16.")
        compute_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg['name'],
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Model and tokenizer loaded successfully.")

    # --- 3. Prepare Input Data ---
    print("Preparing input data...")
    try:
        dataset = load_dataset(data_cfg['dataset_name'], split="train")
        subset = dataset.select(range(data_cfg['max_samples']))

        judgments_df = pd.read_csv(data_cfg['paths']['judgments'])
        bert_labels_df = pd.read_csv(data_cfg['paths']['bert_labels'])
        bert_is_correct_labels = bert_labels_df['is_correct'].values.tolist()

        # Ensure data sources are consistent
        assert len(judgments_df) >= data_cfg['max_samples'], "Not enough judgments found."
        assert len(bert_is_correct_labels) >= data_cfg['max_samples'], "Not enough BERT labels found."

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please check paths in config. Path: {e.filename}")
        return

    # Create the initial list of samples for processing
    initial_samples_list = [
        {'prompt_for_masking': data_cfg['prompt_template'].format(instruction=s['question']),
         'completion': s['answer'],
         'question': s['question']}
        for s in subset
    ]
    print(f"Prepared {len(initial_samples_list)} samples for processing.")

    # Pre-encode punctuation tokens to exclude them during weight calculation
    puncs = ".,!?;:()[]{}<>*@#$%^&'\\\""
    puncs_enc = {tokenizer.encode(p, add_special_tokens=False)[0] for p in puncs}

    # --- 4. Main Loop for Saliency Weight Calculation ---
    tvaft_training_data = []

    print("Starting saliency weight calculation for each token...")
    try:
        for i, sample in enumerate(tqdm(initial_samples_list, desc="Creating TVAFT Data")):
            x_base_prompt = sample["prompt_for_masking"]
            y_standard = sample["completion"]
            y_standard_token_ids = tokenizer.encode(y_standard, add_special_tokens=False)

            if not y_standard_token_ids:
                continue

            is_correct = bert_is_correct_labels[i]
            saliency_weights = []

            # --- Logic for CORRECT samples (is_correct = True) ---
            if is_correct:
                context_probe = x_base_prompt + y_standard + tvaft_cfg['probe_context_separator'] + x_base_prompt
                probs_standard = get_token_probabilities(model, tokenizer, context_probe, y_standard_token_ids, EPSILON)

                for t_idx, p_std_t in enumerate(probs_standard):
                    ratio = (p_std_t - tvaft_cfg['p_correct_threshold']) / max(1.0 - tvaft_cfg['p_correct_threshold'],
                                                                               EPSILON)
                    bonus = 1.5 / (1.0 + np.exp(-5.0 * ratio))
                    penalty_factor = (p_std_t / max(tvaft_cfg['p_correct_threshold'], EPSILON)) ** 0.5

                    if p_std_t > tvaft_cfg['p_correct_threshold'] and y_standard_token_ids[t_idx] not in puncs_enc:
                        s_yt = 1.0 + bonus
                    else:
                        s_yt = penalty_factor

                    saliency_weights.append(max(0.1, min(s_yt, 10.0)))  # Local clipping

            # --- Logic for INCORRECT samples (is_correct = False) ---
            else:
                judgment_text = judgments_df.iloc[i]['judgment']

                probs_base = get_token_probabilities(model, tokenizer, x_base_prompt, y_standard_token_ids, EPSILON)

                context_standard = x_base_prompt + y_standard + tvaft_cfg['probe_context_separator'] + x_base_prompt
                probs_standard = get_token_probabilities(model, tokenizer, context_standard, y_standard_token_ids,
                                                         EPSILON)

                context_judge = x_base_prompt + str(judgment_text) + tvaft_cfg[
                    'probe_context_separator'] + x_base_prompt
                probs_judge = get_token_probabilities(model, tokenizer, context_judge, y_standard_token_ids, EPSILON)

                min_len = len(y_standard_token_ids)

                for t_idx in range(min_len):
                    p_base_t = probs_base[t_idx] if t_idx < len(probs_base) else EPSILON
                    p_std_t = probs_standard[t_idx] if t_idx < len(probs_standard) else EPSILON
                    p_judge_t = probs_judge[t_idx] if t_idx < len(probs_judge) else EPSILON

                    r1 = p_judge_t / max(p_base_t, EPSILON)
                    r2 = p_judge_t / max(p_std_t, EPSILON)
                    r_max = max(r1, r2)

                    s_yt_raw = 2.0 / (1.0 + np.exp(-(r_max - tvaft_cfg['r_saliency_threshold'])))

                    if p_judge_t > tvaft_cfg['p_incorrect_threshold'] and y_standard_token_ids[t_idx] not in puncs_enc:
                        s_yt = s_yt_raw
                    else:
                        s_yt = 0.5

                    saliency_weights.append(max(0.0, min(s_yt, 1.5)))  # Local clipping

            # --- 5. Normalization and Scaling ---
            if saliency_weights:
                weights_np = np.array(saliency_weights, dtype=np.float32)
                mean_weight = weights_np.mean()
                if mean_weight > 0:
                    weights_np = weights_np / mean_weight  # Mean normalization

                # Apply smooth scaling
                saliency_weights = smooth_scale_weights(weights_np.tolist(), **tvaft_cfg['scaling'])

            # Store the final processed data for this sample
            tvaft_training_data.append({
                "text_for_training": x_base_prompt + y_standard,
                "prompt_for_masking": x_base_prompt,
                "saliency_weights": saliency_weights,
                "is_correct": bool(is_correct),  # Ensure JSON serializable boolean
                "question": sample["question"],
                "completion": y_standard
            })

    finally:
        # --- 6. Cleanup and Save Results ---
        del model
        torch.cuda.empty_cache()
        print("Cleaned up model and CUDA cache.")

        output_path = data_cfg['paths']['output']
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tvaft_training_data, f, ensure_ascii=False, indent=4)

        print(f"\n✅ Done! Saved {len(tvaft_training_data)} samples to file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4 of TVAFT Pipeline: Calculate token values.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)

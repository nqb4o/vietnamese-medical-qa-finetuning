# src/utils/metrics.py

import numpy as np
import evaluate
from pyvi import ViTokenizer

def compute_token_metrics_vi(predictions, references):
    """
    Calculates token-level Precision, Recall, and F1-score for Vietnamese.
    """
    f1_scores, precision_scores, recall_scores = [], [], []
    for pred, ref in zip(predictions, references):
        # Tokenize using pyvi
        pred_tokenized = ViTokenizer.tokenize(pred)
        ref_tokenized = ViTokenizer.tokenize(ref)

        pred_tokens = set(pred_tokenized.split())
        ref_tokens = set(ref_tokenized.split())

        # Handle special cases
        if not pred_tokens and not ref_tokens:
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            f1_scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        common_tokens = pred_tokens.intersection(ref_tokens)

        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "Token-level Precision": np.mean(precision_scores) * 100,
        "Token-level Recall": np.mean(recall_scores) * 100,
        "Token-level F1": np.mean(f1_scores) * 100
    }

def compute_all_metrics(predictions, references):
    """
    Calculates a full suite of metrics including BLEU, ROUGE, BERTScore, and token-level metrics.
    """
    results = {}

    # 1. BLEU-4
    print("Calculating BLEU-4...")
    bleu_metric = evaluate.load('sacrebleu')
    references_for_bleu = [[ref] for ref in references]
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_for_bleu)
    results['BLEU-4'] = bleu_results['score']

    # 2. ROUGE (Recall) - As per the paper
    print("Calculating ROUGE...")
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    results['ROUGE-1 (Recall)'] = rouge_results['rouge1'] * 100
    results['ROUGE-2 (Recall)'] = rouge_results['rouge2'] * 100
    results['ROUGE-L (Recall)'] = rouge_results['rougeL'] * 100

    # 3. BERTScore (F1)
    print("Calculating BERTScore...")
    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="vi")
    results['BERTScore (F1)'] = np.mean(bertscore_results['f1']) * 100

    # 4. Token-level Metrics for Vietnamese
    print("Calculating Token-level metrics...")
    token_metrics = compute_token_metrics_vi(predictions, references)
    results.update(token_metrics)

    return results

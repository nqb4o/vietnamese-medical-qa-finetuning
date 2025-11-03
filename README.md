# Fine-tuning LLMs for Vietnamese Medical Question Answering (TVAFT)

This project is an open-source implementation and comparison of different fine-tuning methods for large language models (LLMs) on a Vietnamese medical question-answering dataset. The core focus of the project is the implementation of the **Token Value-Aware Fine-Tuning (TVAFT)** method described in the accompanying research paper.

The main objective is to improve the quality of language models in specialized and low-resource domains by focusing on tokens that carry significant information.

## Implemented Methods
- **Supervised Fine-Tuning (SFT)**: The standard fine-tuning method.
- **Reinforced Fine-Tuning (ReFT)**: Implemented via DPO (Direct Preference Optimization), a technique for learning from preferences.
- **Token Value-Aware Fine-Tuning (TVAFT)**: The proposed method in the paper, which uses saliency weights for each token to guide the learning process.

## Directory Structure
```
.
├── data/                 # Contains raw and processed data
├── notebooks/            # Contains notebooks for exploration and evaluation
├── src/                  # Contains all source code
│   ├── configs/          # YAML configuration files for each experiment
│   ├── sft/              # Source code for SFT
│   ├── reft/             # Source code for ReFT/DPO
│   ├── tvaft/            # Source code for the TVAFT pipeline
│   └── utils/            # Shared utility functions
├── .env.example          # Example file for environment variables
├── requirements.txt      # Required Python libraries
└── README.md             # This guide file
```

## Installation Guide

**Requirements:**
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url.git
    cd vietnamese-medical-qa-finetuning
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install PyTorch:**
    Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the version that matches your system (especially the CUDA version). For example:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Install the remaining libraries:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    Copy the `.env.example` file to a new file named `.env` and enter your API Key.
    ```bash
    cp .env.example .env
    # Open the .env file and add your GEMINI_API_KEY
    ```

## Execution Flow

All scripts are run from the project's root directory.

### 1. Fine-tuning with SFT
```bash
python src/sft/finetune.py --config src/configs/sft_config.yaml
```

### 2. Fine-tuning with ReFT (DPO)
This is a 2-step process:
**Step 1: Prepare the DPO dataset**
*(Note: This step requires the `judgments_0_4166.csv` and `base_model_responses_0_4166.csv` files to be available in the `data/processed/tvaft_intermediate/` directory, or you must adjust the paths in the `reft_config.yaml` file)*
```bash
python src/reft/prepare_data.py --config src/configs/reft_config.yaml
```
**Step 2: Run DPO fine-tuning**
```bash
python src/reft/finetune.py --config src/configs/reft_config.yaml
```

### 3. Fine-tuning with TVAFT
This is a pipeline consisting of several sequential steps. Please run the scripts in the following order:

**Step 1: Generate responses from the base model**
```bash
python src/tvaft/01_generate_responses.py --config src/configs/base_config.yaml
```

**Step 2: Get "judgements" from the Gemini API**
```bash
python src/tvaft/02_get_judgements.py --config src/configs/base_config.yaml
```

**Step 3: Label `is_correct` using Sentence-BERT**
```bash
python src/tvaft/03_label_correctness.py --config src/configs/base_config.yaml
```

**Step 4: Calculate Saliency values for each token**
```bash
python src/tvaft/04_calculate_token_values.py --config src/configs/base_config.yaml
```

**Step 5: Run TVAFT fine-tuning**
```bash
python src/tvaft/finetune.py --config src/configs/tvaft_config.yaml
```

## Model Evaluation
After the models have been trained, open and run the `notebooks/02_evaluation.ipynb` notebook to generate responses on the validation set and calculate metrics (BLEU, ROUGE, BERTScore, etc.) to compare performance.

## Citation
This project is based on the method proposed in the paper:

> [Paper Title: Token Value-Aware Fine-Tuning: A Token-Level Importance Learning Framework for Low-Resource Vietnamese Medical Question Answering](link)
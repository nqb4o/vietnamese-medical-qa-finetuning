# src/tvaft/02_get_judgements.py

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm.auto import tqdm
import time
import json
import argparse
import yaml
from datasets import load_dataset

# Load environment variables from .env file (containing GEMINI_API_KEY)
load_dotenv()


def get_judgement(model, question, standard_answer, model_response):
    """Sends a request to Gemini and returns the parsed judgement."""
    prompt_template = """
        Bạn là một chuyên gia sửa lỗi và cải thiện câu trả lời.

        Cho mỗi câu hỏi, bạn được cung cấp:
        1. Câu hỏi của người dùng.
        2. Câu trả lời đúng chuẩn mực (Y_standard).
        3. Câu trả lời sai do Model tạo ra (Y_model).
        
        Nhiệm vụ của bạn là tạo ra một phiên bản **đã được sửa chữa hoàn toàn và cải thiện**
        của Y_model (gọi là 'judgment'), tuân thủ các yêu cầu sau:
        - Trả lời trực tiếp và chính xác cho câu hỏi.
        - Sửa tất cả lỗi có trong Y_model.
        - Tận dụng thông tin từ Y_standard để đảm bảo tính đúng đắn và đầy đủ.
        - Không cần giải thích quá trình sửa lỗi, chỉ cung cấp câu trả lời cuối cùng.
        - Trả về JSON hợp lệ. Không chứa ký tự escape không hợp lệ. Dùng \\ thay vì \.
        
        Hãy cung cấp kết quả cho toàn bộ batch dưới dạng một mảng JSON (JSON array).
        Mỗi phần tử trong mảng phải có cấu trúc:
        {'id': <ID_câu_hỏi>, 'judgment': <câu_trả_lời_đã_sửa>}.
        """
    prompt = prompt_template.format(
        question=question,
        standard_answer=standard_answer,
        model_response=model_response
    )

    for _ in range(3):
        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(cleaned_response)
            return result['judgment']
        except Exception as e:
            print(f"API or JSON parsing error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    return "Processing Error"


def main(config_path: str):
    print("--- STARTING STEP 2: GET JUDGEMENTS FROM GEMINI API ---")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    data_config = config['data']

    # Configure Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY in your .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Load necessary data
    print("Loading necessary data...")
    dataset = load_dataset(data_config['dataset_name'], split="train")
    train_test_split = dataset.train_test_split(test_size=data_config['test_size'], seed=data_config['seed'])
    train_subset = train_test_split['train'].select(range(data_config['max_samples']))

    model_responses_df = pd.read_csv(data_config['paths']['tvaft_01_model_responses'])

    assert len(train_subset) == len(model_responses_df), "Sample counts do not match!"

    # Create judgements
    judgements = []

    for i in tqdm(range(len(train_subset)), desc="Generating Judgements"):
        sample = train_subset[i]
        model_response = model_responses_df.iloc[i]['model_response']

        judgement = get_judgement(
            model,
            sample['question'],
            sample['answer'],
            model_response
        )
        judgements.append(judgement)

        time.sleep(1.1)

    # Save results
    output_path = data_config['paths']['tvaft_02_judgements']
    df = pd.DataFrame({'judgment': judgements})
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ Done! Saved {len(judgements)} judgements to file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 of TVAFT Pipeline: Get judgements from Gemini.")
    parser.add_argument("--config", type=str, default="src/configs/base_config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    main(args.config)

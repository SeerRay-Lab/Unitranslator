import json
import os
from tqdm import tqdm

def convert_to_json_format_robust(
    en_anno_path: str,
    de_anno_path: str,
    en_image_dir: str,
    de_image_dir: str,
    output_json_path: str
):
    print("Reading annotation files...")
    try:
        with open(en_anno_path, 'r', encoding='utf-8') as f:
            en_sentences = [line.strip() for line in f]
        with open(de_anno_path, 'r', encoding='utf-8') as f:
            de_sentences = [line.strip() for line in f]
    except FileNotFoundError as e:
        print(f"Error: Annotation file not found - {e}")
        return

    if len(en_sentences) != len(de_sentences):
        print(
            f"Warning: Sentence count mismatch! "
            f"English: {len(en_sentences)}, German: {len(de_sentences)}"
        )
        min_len = min(len(en_sentences), len(de_sentences))
        en_sentences = en_sentences[:min_len]
        de_sentences = de_sentences[:min_len]

    print(f"Found {len(en_sentences)} parallel sentences. Verifying images...")

    all_data = []
    skipped_count = 0

    for i in tqdm(range(len(en_sentences)), desc="Processing sentences"):
        en_image_path = os.path.join(en_image_dir, f"{i}.jpg")
        de_image_path = os.path.join(de_image_dir, f"{i}.jpg")

        if not (os.path.exists(en_image_path) and os.path.exists(de_image_path)):
            skipped_count += 1
            continue

        en_text = en_sentences[i]
        de_text = de_sentences[i]

        # -------- 1. German -> English --------
        all_data.append({
            "image": [de_image_path, en_image_path],
            "conversations": [
                {"from": "human", "value": "<image>\nTranslate all German texts into English."},
                {"from": "gpt", "value": "<gen_image>"},
                {"from": "ocr", "value": de_text},
                {"from": "translation", "value": en_text}
            ]
        })

        # -------- 2. English -> German --------
        all_data.append({
            "image": [en_image_path, de_image_path],
            "conversations": [
                {"from": "human", "value": "<image>\nTranslate all English texts into German."},
                {"from": "gpt", "value": "<gen_image>"},
                {"from": "ocr", "value": en_text},
                {"from": "translation", "value": de_text}
            ]
        })

        # -------- 3. English -> English (identity) --------
        all_data.append({
            "image": [en_image_path, en_image_path],
            "conversations": [
                {"from": "human", "value": "<image>\nTranslate all English texts into English."},
                {"from": "gpt", "value": "<gen_image>"},
                {"from": "ocr", "value": en_text},
                {"from": "translation", "value": en_text}
            ]
        })

        # -------- 4. German -> German (identity) --------
        all_data.append({
            "image": [de_image_path, de_image_path],
            "conversations": [
                {"from": "human", "value": "<image>\nTranslate all German texts into German."},
                {"from": "gpt", "value": "<gen_image>"},
                {"from": "ocr", "value": de_text},
                {"from": "translation", "value": de_text}
            ]
        })

    print("\n--- Processing Summary ---")
    print(f"Total annotation pairs: {len(en_sentences)}")
    print(f"Valid image pairs: {len(all_data) // 4}")
    print(f"Skipped pairs (missing images): {skipped_count}")
    print(f"Total generated samples: {len(all_data)}")

    print(f"\nSaving to {output_json_path}...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print("✅ Conversion successful!")
    except IOError as e:
        print(f"Error: Failed to write file - {e}")

if __name__ == '__main__':
    # --- 请在这里修改为你的实际路径 ---
    EN_ANNO_PATH = '/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en/train.en'
    DE_ANNO_PATH = '/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en/train.de'
    EN_IMAGE_DIR = '/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/train_en'
    DE_IMAGE_DIR = '/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/train_de'
    
    # 输出的JSON文件名
    OUTPUT_JSON_PATH = 'translation_dataset_validated_de_en_self_train.json'
    # -----------------------------------

    convert_to_json_format_robust(
        en_anno_path=EN_ANNO_PATH,
        de_anno_path=DE_ANNO_PATH,
        en_image_dir=EN_IMAGE_DIR,
        de_image_dir=DE_IMAGE_DIR,
        output_json_path=OUTPUT_JSON_PATH
    )

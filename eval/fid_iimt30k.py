#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
import unicodedata
import re
import torch
from tqdm import tqdm

# --- Import evaluation libraries ---
import easyocr
import sacrebleu
from comet import download_model, load_from_checkpoint
from pytorch_fid.fid_score import calculate_fid_given_paths

# ---------------------------------
# 辅助函数
# ---------------------------------

def remove_punctuation(text: str) -> str:
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# def get_ocr_texts(image_dir: Path, ocr_reader, lang: str) -> dict:
#     print(f"\n[INFO] Running EasyOCR on images in: {image_dir}")
#     ocr_results = {}
#     # 确保图像按名称排序，这对于与 GT 文件行号匹配至关重要
#     image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])

#     for img_path in tqdm(image_paths, desc="EasyOCR Processing"):
#         try:
#             result = ocr_reader.readtext(str(img_path), detail=0, paragraph=True)
#             full_text = " ".join(result).lower()
#             cleaned_text = remove_punctuation(full_text)
#             ocr_results[img_path.stem] = cleaned_text
#         except Exception as e:
#             print(f"[WARNING] Could not process OCR for {img_path}: {e}")
#             ocr_results[img_path.stem] = ""

#     return ocr_results

def natural_sort_key(path):
    """生成自然排序的键，用于 sorted 函数的 key 参数"""
    # 将文件名中的数字部分转换为整数，其他部分保持字符串
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', path.stem)]

def get_ocr_texts(image_dir: Path, ocr_reader, lang: str) -> dict:
    print(f"\n[INFO] Running EasyOCR on images in: {image_dir}")
    ocr_results = {}
    # 使用自然排序获取图像路径列表，确保按数字顺序排列
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')],
        key=natural_sort_key
    )

    for img_path in tqdm(image_paths, desc="EasyOCR Processing"):
        try:
            result = ocr_reader.readtext(str(img_path), detail=0, paragraph=True)
            full_text = " ".join(result).lower()
            cleaned_text = remove_punctuation(full_text)
            ocr_results[img_path.stem] = cleaned_text
        except Exception as e:
            print(f"[WARNING] Could not process OCR for {img_path}: {e}")
            ocr_results[img_path.stem] = ""

    return ocr_results

def load_ground_truth_texts_from_file(gt_text_file: Path, sorted_image_stems: list) -> dict:
    """
    从单个 .txt 文件加载真实（GT）文本。
    假设 .txt 文件中的行顺序与排序后的图像文件顺序一一对应。
    """
    print(f"\n[INFO] Loading ground truth texts from single file: {gt_text_file}")
    gt_texts = {}
    try:
        with open(gt_text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"[ERROR] Could not read GT file {gt_text_file}: {e}")
        return {}

    if len(lines) != len(sorted_image_stems):
        print(f"[WARNING] Mismatch! Found {len(sorted_image_stems)} images but {len(lines)} lines in GT file.")
        print("[WARNING] Metrics will be calculated for the smaller of the two counts.")
        min_len = min(len(lines), len(sorted_image_stems))
        lines = lines[:min_len]
        sorted_image_stems = sorted_image_stems[:min_len]

    for i, stem in enumerate(tqdm(sorted_image_stems, desc="Mapping GT Texts")):
        raw_text = lines[i].lower()
        cleaned_text = remove_punctuation(raw_text)
        gt_texts[stem] = cleaned_text
            
    return gt_texts

def calculate_bleu_comet(ocr_results: dict, gt_texts: dict):
    print("\n[INFO] Calculating BLEU and COMET scores...")
    hypotheses, references = [], []

    # 确保 OCR 结果和 GT 文本按相同的顺序配对
    for stem in sorted(ocr_results.keys()):
        if stem in gt_texts and gt_texts[stem]:
            hypotheses.append(ocr_results[stem])
            references.append(gt_texts[stem])

    if not hypotheses:
        print("[ERROR] No valid pairs of OCR text and Ground Truth text found.")
        return

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"✅ Corpus BLEU Score: {bleu.score:.2f}")

    print("\n[INFO] Loading COMET model...")
    try:
        # model_path = download_model("Unbabel/wmt22-comet-da")
        
        model = load_from_checkpoint("/mnt/vlm-ks3/ljh/code/GPT-Image-Edit/wmt22-comet-da/checkpoints/model.ckpt")
    except Exception as e:
        print(f"[ERROR] Could not load COMET model: {e}"); return

    comet_data = [{"src": "", "mt": mt, "ref": ref} for mt, ref in zip(hypotheses, references)]
    print("[INFO] Calculating COMET scores...")
    model_output = model.predict(comet_data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
    print(f"✅ COMET Score: {model_output.system_score:.4f}")

def calculate_fid(test_images_dir: str, ref_images_dir: str, batch_size: int, device: str):
    print("\n[INFO] Calculating FID score...")
    try:
        fid_value = calculate_fid_given_paths([ref_images_dir, test_images_dir], batch_size, device, 2048, int(os.cpu_count() / 2))
        print(f"✅ FID Score: {fid_value:.2f}")
    except Exception as e:
        print(f"[ERROR] FID calculation failed: {e}")

# ---------------------------------
# 主函数
# ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Generation Metrics using EasyOCR (BLEU, COMET, FID)")
    parser.add_argument("--test_images_dir", type=str, default="results/outputs_iimt30k_test_4k_de2en_cut", help="Path to the directory of generated/test images.")
    parser.add_argument("--ref_images_dir", type=str, default="/mnt/vlm-ks3/ljh/data/yztian/IIMT30k/IIMT30k/Arial/test_flickr/en/image", help="Path to the directory of reference/ground-truth images (for FID).")
    parser.add_argument("--gt_text_file", type=str, default="/mnt/vlm-ks3/ljh/data/yztian/IIMT30k/IIMT30k/Arial/test_flickr/en/subtitle.txt", help="Path to the directory of ground-truth text files (.txt).")
    parser.add_argument("--lang", type=str, default='en', help="Language for EasyOCR (e.g., 'en', 'ch_sim', 'de').")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for FID calculation.")
    
    args = parser.parse_args()

    test_images_dir, ref_images_dir, gt_text_file = Path(args.test_images_dir), Path(args.ref_images_dir), Path(args.gt_text_file)
    
    if not test_images_dir.is_dir(): print(f"[ERROR] Directory not found: {test_images_dir}"); return
    if not ref_images_dir.is_dir(): print(f"[ERROR] Directory not found: {ref_images_dir}"); return
    if not gt_text_file.is_file(): print(f"[ERROR] Ground truth file not found: {gt_text_file}"); return

    # --- 步骤 1: 初始化 OCR 并处理测试图像 ---
    print("[INFO] Initializing EasyOCR reader...")
    reader = easyocr.Reader([args.lang], gpu=torch.cuda.is_available())
    ocr_results = get_ocr_texts(test_images_dir, reader, args.lang)
    
    # --- 步骤 2: 加载 GT 文本 ---
    # 确保 OCR 结果的键是排序的，以便和文件行号对应
    image_stems = list(ocr_results.keys())
    gt_texts = load_ground_truth_texts_from_file(gt_text_file, image_stems)

    # --- 步骤 3: 计算 BLEU 和 COMET ---
    calculate_bleu_comet(ocr_results, gt_texts)

    # --- 步骤 4: 计算 FID ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculate_fid(str(test_images_dir), str(ref_images_dir), args.batch_size, device)

    print("\n[INFO] Evaluation finished.")


if __name__ == "__main__":
    main()


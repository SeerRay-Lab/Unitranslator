import easyocr
import os
from tqdm import tqdm
import argparse
import sacrebleu
import numpy as np
from shapely.geometry import Polygon
import unicodedata
import re

def calculate_iou(box1, box2):
    """
    计算两个四边形的IoU值
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection_area = poly1.intersection(poly2).area
    area1 = poly1.area
    area2 = poly2.area
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area
    return iou

def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    """
    对两个图片中的文本框进行匹配
    """
    matches = []
    for i, box1 in enumerate(boxes1):
        max_iou = 0
        max_j = -1
        for j, box2 in enumerate(boxes2):
            iou = calculate_iou(box1, box2)
            if iou > max_iou:
                max_iou = iou
                max_j = j
        if max_iou >= iou_threshold:
            matches.append((i, max_j))
        else:
            matches.append((i, -1))
    return matches

def remove_punctuation(text: str) -> str:
    """
    去掉 Unicode 标点符号，并把多余空白压缩为单空格
    """
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_dir", type=str, default="results/outputs_textloss_3b_de_en/images_6k", help="generated img dir.")
    parser.add_argument("--ref_dir", type=str, default="/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/test_en", help="reference img dir.")
    parser.add_argument("--lang", type=str, default='en', help="Source language.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold.")
    args = parser.parse_args()

    reader = easyocr.Reader([args.lang])

    # --- 代码修改开始 ---

    # 1. 读取所有推理结果文件，并存入一个字典（key: 文件名, value: 完整路径）
    generate_result_folder = args.generate_dir
    generate_files_map = {
        file: os.path.join(generate_result_folder, file)
        for file in os.listdir(generate_result_folder) if 'jpg' in file
    }

    # 2. 读取所有GT文件
    ref_result_dir = args.ref_dir
    ref_files = [file for file in os.listdir(ref_result_dir) if 'jpg' in file]

    generate_result = []
    ref_result = []
    bleu_total = 0

    # 3. 遍历GT文件，在推理结果中查找同名文件进行匹配
    print(f"Found {len(ref_files)} reference files. Starting matching and OCR process...")
    
    for ref_filename in tqdm(ref_files):
        # 在推理结果字典中查找匹配项
        if ref_filename in generate_files_map:
            ref_file_path = os.path.join(ref_result_dir, ref_filename)
            generate_file_path = generate_files_map[ref_filename]

            generate_ocr_result = reader.readtext(generate_file_path, paragraph=True)
            ref_ocr_result = reader.readtext(ref_file_path, paragraph=True)
            generate_ocr_boxes = [np.array(item[0]) / 1024. for item in generate_ocr_result]
            ref_ocr_boxes = [np.array(item[0]) / 512. for item in ref_ocr_result]

            matches = match_boxes(ref_ocr_boxes, generate_ocr_boxes, iou_threshold=args.iou_threshold)
            generate_ocr_result = [generate_ocr_result[item[1]][1].lower() if item[1] != -1 else '' for item in matches]
            ref_ocr_result = [ref_ocr_result[item[0]][1].lower() for item in matches]

            generate_text = remove_punctuation(' '.join(generate_ocr_result))
            ref_text = remove_punctuation(' '.join(ref_ocr_result))
            generate_result.append(generate_text)
            ref_result.append(ref_text)
        else:
            # 如果GT文件在推理结果中没有对应的文件，可以选择打印一条警告
            print(f"Warning: No matching generated file found for reference file '{ref_filename}'")
            
    # --- 代码修改结束 ---

    if not generate_result:
        print("Error: No matching files were processed. Cannot calculate BLEU score.")
    else:
        # calculate bleu
        bleu = sacrebleu.corpus_bleu(generate_result, [ref_result])
        # bleu_v2 = bleu_total / len(generate_result) # 您原有的bleu_v2计算逻辑依赖于循环内部的累加，如果您需要它，需要取消注释并修改循环
        
        print("-" * 30)
        print(f"Total matched files processed: {len(generate_result)}")
        print(f"IoU threshold: {args.iou_threshold}")
        print(f"Structure sacrebleu: {bleu.score}")
        # print("structure sacrebleu v2: {}".format(bleu_v2))


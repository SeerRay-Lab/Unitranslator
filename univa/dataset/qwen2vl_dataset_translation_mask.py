from typing import Any, Callable, Optional, List

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
from PIL import Image
from univa.utils.prompter import Prompter
import numpy as np
from einops import rearrange
import random
# from qwen_vl_utils.vision_process import fetch_image, fetch_video
from qwen_vl_utils.vision_process import to_rgb, smart_resize, fetch_video
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.get_mask import get_weight_mask
from univa.utils.get_ocr import get_ocr_result
from fractions import Fraction
from torchvision.transforms import functional
from torchvision import transforms
from io import BytesIO
import base64
import requests
import torch
from PIL import Image
from torchvision import io, transforms
from typing import Optional
import re


def scale_and_pad_image(
    image: Image.Image,
    target_size: tuple[int, int],
    background_color: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    将图像按比例缩放以适应目标尺寸，然后用纯色背景填充剩余空间。

    Args:
        image: 输入的 PIL.Image 对象。
        target_size: 最终图像的 (宽度, 高度) 元组，例如 (256, 256)。
        background_color: 用于填充背景的 (R, G, B) 颜色。

    Returns:
        经过处理后，尺寸为 target_size 的 PIL.Image 对象。
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_width, original_height = image.size
    target_width, target_height = target_size

    # 计算缩放比例，以完全容纳图像
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 使用高质量的LANCZOS滤波器进行缩放
    try:
        # Pillow >= 9.1.0
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except AttributeError:
        # 兼容旧版 Pillow
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # 创建一个具有目标尺寸和背景色的新画布
    padded_image = Image.new("RGB", target_size, background_color)

    # 计算粘贴位置，使其居中
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # 将缩放后的图像粘贴到画布中央
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

def letterbox_pil(
    img: Image.Image,
    target_size=(256, 256),
    fill=(255, 255, 255),  # 白边；想黑边用 (0,0,0)
    resample=Image.Resampling.LANCZOS,
):
    """
    等比缩放 + padding 到 target_size.
    返回: new_img (PIL), scale, (pad_left, pad_top)
    """
    tw, th = target_size
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # 处理 RGBA：先按 fill 颜色铺底再合成，避免透明区域变黑
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, fill + (255,))
        img = Image.alpha_composite(bg, img).convert("RGB")

    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError(f"Invalid image size: {img.size}")

    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), resample=resample)

    canvas = Image.new("RGB", (tw, th), fill)
    pad_left = (tw - new_w) // 2
    pad_top  = (th - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    return canvas, scale, (pad_left, pad_top)

def find_subsequence_positions(seq: List[int], subseq: List[int]) -> List[int]:
    """
    返回 subseq 在 seq 中所有起始索引（如果存在）。简单暴力查找。
    seq, subseq 都是 int list（token ids）。
    """
    if len(subseq) == 0 or len(seq) < len(subseq):
        return []
    starts = []
    n, m = len(seq), len(subseq)
    # 暴力匹配（n*m），对 prompt 长度正常是可以接受的
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            starts.append(i)
    return starts

def make_mask_for_subtoken(seq_ids: List[int], sub_ids: List[int]) -> torch.BoolTensor:
    """
    返回与 seq_ids 长度相同的布尔 mask，sub_ids 的位置被标为 True（包含整个子序列）。
    """
    mask = torch.zeros(len(seq_ids), dtype=torch.bool)
    starts = find_subsequence_positions(seq_ids, sub_ids)
    m = len(sub_ids)
    for s in starts:
        mask[s:s+m] = True
    return mask

def convert_tag_block(text, src_role, tgt_tag):
    """
    Convert:
        <|im_start|>ocr\n...<|im_end|>
    into:
        <ocr>...</ocr>
    """
    prefix = f"<|im_start|>{src_role}\n"
    if text.startswith(prefix):
        text = text[len(prefix):]

    text = text.replace("<|im_end|>", "").strip()
    return f"<{tgt_tag}>{text}</{tgt_tag}>"

class Step1XTokenizer:
    """
    Enhanced Step1X tokenization for better quote protection and image token handling.
    
    核心功能：
    1. 文本分段：识别引号内容和普通文本
    2. 字面量保护：对引号内容进行字符级保护
    3. 分段tokenize：避免截断，保护图像token
    """
    
    def __init__(self, tokenizer, image_token: str):
        self.tokenizer = tokenizer
        self.image_token = image_token
        # 使用一个在词汇表中存在但极少在普通文本中出现的token作为临时占位符
        self.placeholder_token = "<|endoftext|>"
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        self.is_checkpoint_tokenizer = self._detect_checkpoint_tokenizer()
        self.failure_count = 0
        self.max_failures = 10
        
    def _detect_checkpoint_tokenizer(self) -> bool:
        """检测是否为checkpoint加载的tokenizer"""
        try:
            if hasattr(self.tokenizer, 'name_or_path'):
                path = str(self.tokenizer.name_or_path)
                return 'checkpoint' in path.lower()  # 只检测checkpoint，不硬编码UniWorld
            return False
        except:
            return False
    
    def _normalize_quotes(self, text: str) -> str:
        """标准化引号类型"""
        # 修复：正确处理中文/弯引号
        text = text.replace('"', '"').replace('"', '"')  # 左右弯双引号
        text = text.replace(''', "'").replace(''', "'")  # 左右弯单引号
        return text
    
    def _extract_literal_segments(self, text: str) -> List[tuple]:
        """
        Step1X核心：提取字面量段落
        
        返回: List[(segment_text, is_literal, quote_type)]
        """
        text = self._normalize_quotes(text)
        segments = []
        current_segment = ""
        in_literal = False
        quote_char = None
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # 处理引号字符 (", ', `)
            if char in ['"', "'", '`'] and (quote_char is None or char == quote_char):
                if current_segment:
                    segments.append((current_segment, in_literal, quote_char))
                    current_segment = ""
                
                if not in_literal:
                    # 开始字面量
                    in_literal = True
                    quote_char = char
                    current_segment = char
                else:
                    # 结束字面量
                    current_segment += char
                    segments.append((current_segment, True, quote_char))
                    current_segment = ""
                    in_literal = False
                    quote_char = None
            else:
                current_segment += char
            
            i += 1
        
        # 添加剩余段落
        if current_segment:
            segments.append((current_segment, in_literal, quote_char))
        
        return segments
    
    def _protect_literal_content(self, text: str, quote_type: str) -> str:
        """
        Step1X字面量保护：给引号内每个字符加空格
        
        例如: "step1x" → " s t e p 1 x "
        """
        if len(text) <= 2:  # 只有引号
            return text
        
        # 提取引号内内容
        if text.startswith(quote_type) and text.endswith(quote_type):
            inner_text = text[1:-1]
        else:
            inner_text = text
        
        # 优化：单侧空格即可，避免过多空格
        protected = quote_type
        for i, char in enumerate(inner_text):
            if char.isspace():
                protected += char
            else:
                if i > 0:  # 非第一个字符前加空格
                    protected += " "
                protected += char
        protected += quote_type
        
        return protected
    

    
    def tokenize_with_protection(self, text: str, **kwargs) -> dict:
        """
        Step1X主函数：带保护的tokenization
        
        核心策略：
        1. 检测checkpoint tokenizer → 直接回退
        2. 无引号内容 → 标准tokenization  
        3. 有引号内容 → Step1X分段处理
        4. 失败 → 计数并回退
        """
        
        # 策略1: Checkpoint tokenizer直接回退
        if self.is_checkpoint_tokenizer:
            return self.tokenizer(text, **kwargs)
        
        # 策略2: 失败次数过多，自动禁用
        if self.failure_count >= self.max_failures:
            return self.tokenizer(text, **kwargs)
        
        # 策略3: 无引号内容，使用标准tokenization
        if '"' not in text and "'" not in text and '`' not in text:
            return self.tokenizer(text, **kwargs)

        try:
            # 修复：只传text，避免参数冲突
            return self._step1x_process(text, **kwargs)
        except Exception as e:
            self.failure_count += 1
            print(f"Warning: Step1X failed ({self.failure_count}/{self.max_failures}): {e}")
            if self.failure_count >= self.max_failures:
                print("Warning: Step1X disabled due to repeated failures")
            
            # 回退到标准tokenization
            return self.tokenizer(text, **kwargs)
    
    def _step1x_process(self, text: str, **kwargs) -> dict:
        """
        Step1X核心处理逻辑
        """
        # Step 0: 保护 image_token
        has_image_token = self.image_token in text
        if has_image_token:
            text = text.replace(self.image_token, self.placeholder_token)

        # Step 1: 文本分段
        segments = self._extract_literal_segments(text)
        
        # Step 2: 分段tokenize
        token_segments = []
        
        for segment_text, is_literal, quote_type in segments:
            if not segment_text.strip():
                continue

            # Step 2.1: 字面量保护
            if is_literal and quote_type:
                segment_text = self._protect_literal_content(segment_text, quote_type)
            
            # Step 2.2: 分段tokenize（修复：内部硬编码参数，避免冲突）
            try:
                # 使用 self.tokenizer 而不是直接调用 __call__
                segment_result = self.tokenizer(
                    text=segment_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                    truncation=False,
                )
                
                if segment_result.input_ids.shape[1] > 0:
                    token_segments.append(segment_result.input_ids)
                    
            except Exception as seg_e:
                print(f"Warning: Segment tokenization failed: {seg_e}")
                # 段落级回退：对这个段落使用标准tokenization
                try:
                    fallback_result = self.tokenizer(
                        segment_text,
                        add_special_tokens=False,
                        return_tensors="pt",
                        truncation=False,
                    )
                    if fallback_result.input_ids.shape[1] > 0:
                        token_segments.append(fallback_result.input_ids)
                except:
                    # 段落完全失败，跳过
                    continue
        
        # Step 3: 合并所有段落
        if not token_segments:
            # 如果没有有效的token，返回一个空的tokenization结果
             return self.tokenizer(
                "", return_tensors="pt", add_special_tokens=False
            )

        # 拼接所有token段落
        combined_tokens = torch.cat(token_segments, dim=1)

        # Step 4: 恢复 image_token
        if has_image_token:
            image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            combined_tokens[combined_tokens == self.placeholder_token_id] = image_token_id
        
        # 返回标准格式
        # 使用 **kwargs 来传递额外的参数给最终的 BatchEncoding
        final_result = self.tokenizer.prepare_for_model(
            combined_tokens.squeeze(0).tolist(), 
            add_special_tokens=False, 
            return_tensors='pt', 
            **kwargs
        )

        return final_result

def get_aspect_ratio(img):
    width, height = img.size
    return Fraction(width, height).limit_denominator()

def has_same_aspect_ratio(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    ratio1 = get_aspect_ratio(img1)
    ratio2 = get_aspect_ratio(img2)
    return ratio1 == ratio2

def has_same_resolution(img1, img2):
    if not isinstance(img1, Image.Image):
        img1 = Image.open(img1).convert('RGB')
    if not isinstance(img2, Image.Image):
        img2 = Image.open(img2).convert('RGB')
    return img1.size == img2.size

class Qwen2VLDataset_mask(Dataset):
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        transform: Callable, 
        tokenizer: PreTrainedTokenizer,
        prompter: Prompter,
        image_processor: Callable,
        processor: Callable = None,
        min_pixels: int = 384*384, 
        max_pixels: int = 384*384, 
        image_token_length: int = 729,
        only_generated_task: bool = False,
        drop_prompt_rate: float = 0.0,
        joint_ref_feature: bool = False,
        anyres: bool = False, 
        mask_weight_type: str = 'log', 
        siglip_processor: Callable = None,
        ocr_enhancer: bool = False, 
        random_data: bool = False, 
        maxnum_per_data: int = -1, 
        notry: bool = False, 
        use_step1x_preprocessing: bool = True,  # 临时禁用Step1X避免训练问题
    ):
        assert dataset_type == 'qwen2vl' or dataset_type == 'qwen2p5vl' or dataset_type == 'qwen2p5vl_trans' or dataset_type == 'qwen2p5vl_mask' or dataset_type == 'qwen2p5vl_trans_cond' or dataset_type == 'qwen2p5vl_tf_mask'
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self._load_data(maxnum_per_data)
        
        self.transform = transform
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompter = prompter
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token = SPACIAL_TOKEN[dataset_type]['image_token']
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = processor.image_processor
        # self.factor = 4 if joint_ref_feature else 1
        self.factor = 2

        self.only_generated_task = only_generated_task  # For denoiser training
        self.drop_prompt_rate = drop_prompt_rate
        if self.drop_prompt_rate > 0:
            assert self.only_generated_task, (
                "Only generated task is supported when drop_prompt_rate > 0"
            )
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.ocr_enhancer = ocr_enhancer
        self.random_data = random_data
        self.notry = notry
        self.use_step1x_preprocessing = use_step1x_preprocessing

        # Initialize Step1X tokenizer if enabled
        if self.use_step1x_preprocessing:
            self.step1x_tokenizer = Step1XTokenizer(self.tokenizer, image_token=self.image_token)
        else:
            self.step1x_tokenizer = None

        # Add image token if not exists.
        assert self.image_token in self.tokenizer.get_vocab()
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_begin_token
        )
        assert isinstance(self.image_begin_token_id, int), (
            f"tokenizer miss image begin token `{self.image_begin_token}`"
        )
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_end_token
        )
        assert isinstance(self.image_end_token_id, int), (
            f"tokenizer miss image end token `{self.image_end_token}`"
        )

    def _load_data(self, maxnum_per_data=-1):
        """
        加载数据，现在会同时处理 `mask` 字段。
        """
        for dataset_line in self.datasets:
            # 假设你的 data.txt 格式是: image_root,json_file,need_weight
            # 我们需要一种方式来关联 mask 根目录，这里我们约定它与json_file在同一级
            # 例如: .../data/train.json -> .../mask/train/
            image_root, json_file, need_weight = dataset_line.split(",")
            
            with open(json_file, "r") as f:
                json_data = json.load(f)

            if maxnum_per_data > 0 and maxnum_per_data < len(json_data):
                json_data = random.sample(json_data, maxnum_per_data)

            dataset_data = []
            print(f"Loading data from {json_file} and verifying paths...")
            for line in tqdm(json_data):
                # a. 处理和验证 image 路径
                if "image" not in line: line["image"] = []
                if isinstance(line["image"], str): line["image"] = [line["image"]]
                
                abs_image_paths = [os.path.join(image_root, p) for p in line["image"]]
                
                # b. ✨ 处理和验证 mask 路径 ✨
                abs_mask_paths = []
                if "mask" in line and line["mask"]:
                    # 假设 mask 路径是相对于某个根目录的，你需要在这里构建绝对路径
                    # 如果 mask 路径已经是绝对的，则不需要 os.path.join
                    # 为了灵活性，我们先假设mask路径是完整的
                    abs_mask_paths = line["mask"]

                # c. ✨ 验证所有文件是否存在 ✨
                all_paths = abs_image_paths + abs_mask_paths
                if not all(os.path.exists(p) for p in all_paths):
                    # print(f"[WARN] Skipping sample due to missing file. Image: {abs_image_paths}, Mask: {abs_mask_paths}")
                    continue

                # d. 将验证过的绝对路径存入
                line["image"] = abs_image_paths
                line["mask"] = abs_mask_paths
                line["need_weight"] = need_weight
                dataset_data.append(line)
                
            print(f"Loaded {len(dataset_data)} valid data entries from {json_file}.")
            self.data.extend(dataset_data)

    def __len__(self):
        return len(self.data)

    def _get_random_data(self, ):
        
        prompt = self.prompter(
            [
                {"from": "system", "value": "You are a helpful assistant."},
                {
                    "from": "user",
                    "value": f"test an image {self.image_token}",
                },
            ]
        )
        input_ids = self.tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", truncation=False,
        ).input_ids
        labels = input_ids

        width, height = 448, 448
        random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(random_data, 'RGB')

        image_slice = [image]
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=image,
            vae_image_transform=self.transform, 
            drop_prompt=False, 
            prompt=prompt, 
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight='true',
            )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        ref_pixel_values = image_dict['ref_pixel_values']
        pil_pixel_values = image_dict['pil_pixel_values']
        siglip_pixel_values = image_dict['siglip_pixel_values']
        weights = image_dict['weights']

        input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths, 
            )
        
        generated_image = torch.randn(3, 512, 512)
        
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": prompt,
            "ref_pixel_values": ref_pixel_values,
            "pil_pixel_values": pil_pixel_values, 
            "siglip_pixel_values": siglip_pixel_values, 
            "weights": weights, 
            "generated_image": generated_image, 
        }
        return return_data


    def getitem(self, data):
        # 1. 你的原始代码：从JSON数据构建conversations列表
        conversations = []
        last_human_prompt = ""
        for item in data["conversations"]:
            role = ""
            if item["from"] == "human":
                role = self.prompter.user_role
                last_human_prompt = item["value"] # 保存最后的用户输入
            elif item["from"] == "gpt":
                role = self.prompter.assistant_role
            elif item['from'] == "translation":
                role = self.prompter.translation_role
            elif item['from'] == "ocr":
                continue
            else:
                raise ValueError(f"Unknown role: {item['from']}")
            
            value = item["value"]
            # if role == self.prompter.translation_role:
            #      value = f"<translation>{value}</translation>"
            conversations.append({"from": role, "value": value})
        
        # 2. 从conversations获取包含所有特殊token的prompt_list
        # prompt_list = self.prompter.get_train_prompt(conversations)
        prompt_list = self.prompter.get_train_prompt(conversations)

        # 3. === 【最终版】使用 "差分" 逻辑精确定位并屏蔽 Labels ===

        # 3.1 找到最后一个答案块在 prompt_list 中的索引
        last_answer_item_index = -1
        for i in range(len(prompt_list) - 1, -1, -1):
            if prompt_list[i].get("is_labels", False):
                last_answer_item_index = i
                break

        if last_answer_item_index == -1:
            raise ValueError("No turn with 'is_labels: True' found in the prompt_list.")

        # 3.2 获取并拼接【答案之前】的所有文本块
        prompt_list_before_answer = prompt_list[:last_answer_item_index]
        text_before_answer = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list_before_answer])

        # 3.3 对 "答案之前" 的文本进行tokenize，得到精确的起始位置长度
        tokens_before_answer = self.tokenizer(text_before_answer, return_tensors="pt", truncation=False)
        answer_start_index = tokens_before_answer.input_ids.shape[1]

        # 3.4 对【完整对话】进行tokenize，得到最终的 input_ids
        full_text = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list])
        input_ids = self.tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
        
        # 3.5 === 核心 "一刀切" 逻辑 ===
        labels = input_ids.clone()
        labels[0, :answer_start_index] = -100
        
        # 8. 图像处理 (你的原始逻辑)
        has_generated_image = self.generated_image_token in full_text
        image_slice = data["image"]
        if has_generated_image:
            image_slice = data["image"][:-1]
        
        image_dict = self._load_image(
            image_slice, self.max_pixels, self.min_pixels, 
            processor=self.processor, image_token=self.image_token, 
            factor=self.factor, 
            last_image=data["image"][-1] if has_generated_image else None,
            vae_image_transform=self.transform, 
            drop_prompt=False, 
            prompt=last_human_prompt.replace('<image>', '').replace('\n', ''),
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight=data['need_weight'],
        )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        
        image_position = []
        if len(image_token_lengths) > 0:
            input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels, # 传递已经正确mask的labels
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths, 
            )

        # 9. 序列截断 (你的原始逻辑)
        max_sequence_length = 32768
        if input_ids.shape[1] > max_sequence_length:
            print(f"Warning: Sequence too long ({input_ids.shape[1]} > {max_sequence_length}), truncating...")
            original_len = input_ids.shape[1]
            input_ids = input_ids[:, -max_sequence_length:]
            labels = labels[:, -max_sequence_length:]
            offset = original_len - max_sequence_length
            if image_position:
                image_position = [pos - offset for pos in image_position if pos >= offset]

        # 10. 返回最终数据
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw, 
            "prompt": last_human_prompt,
            "ref_pixel_values": image_dict['ref_pixel_values'],
            "pil_pixel_values": image_dict['pil_pixel_values'], 
            "siglip_pixel_values": image_dict['siglip_pixel_values'], 
            "weights": image_dict['weights'], 
        }
        
        # 处理生成任务的额外字段
        if has_generated_image:
            image = Image.open(data["image"][-1]).convert("RGB")
            image_tensor = torch.tensor(np.array(image)) / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []

        return return_data

    def __getitem__(self, idx):
        # 我们将使用一个新的、干净的 getitem 实现
        # 你可以保留你的 try-except 包装器
        try:
            return self._getitem_impl(self.data[idx])
        except Exception as e:
            print(f"Error processing data at index {idx}: {e}. Loading a random sample.")
            random_idx = random.randint(0, len(self.data) - 1)
            return self._getitem_impl(self.data[random_idx])    

    def _getitem_impl(self, data: dict) -> dict:
        """
        【最终版】核心数据处理实现。
        加载文本、图像、以及新的Mask，并确保labels的正确性。
        """
        # --- 1. 处理文本和Labels (使用我们最终确定的、健壮的逻辑) ---
        
        # a. 从JSON数据构建conversations列表
        conversations = []
        last_human_prompt = ""
        for item in data["conversations"]:
            role = ""
            value = item["value"]
            
            if item["from"] == "human":
                role = self.prompter.user_role
                last_human_prompt = value
            elif item["from"] == "gpt":
                role = self.prompter.assistant_role
            elif item['from'] == "translation":
                role = self.prompter.translation_role
                # 按照你的数据格式，在translation内容外包裹特殊tag
                value = f"{value}"
            elif item['from'] == "ocr":
                continue
            else:
                # 跳过未知的角色
                continue
                
            conversations.append({"from": role, "value": value})

        # b. 从conversations获取包含所有特殊token的prompt_list
        prompt_list = self.prompter.get_train_prompt(conversations)

        # c. 使用 "差分" 逻辑精确定位并屏蔽 Labels
        last_answer_item_index = -1
        for i in range(len(prompt_list) - 1, -1, -1):
            if prompt_list[i].get("is_labels", False):
                last_answer_item_index = i
                break

        if last_answer_item_index == -1:
            raise ValueError(f"No turn with 'is_labels: True' found in data: {data}")

        prompt_list_before_answer = prompt_list[:last_answer_item_index]
        text_before_answer = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list_before_answer])
        
        tokens_before_answer = self.tokenizer(text_before_answer, return_tensors="pt", truncation=False)
        answer_start_index = tokens_before_answer.input_ids.shape[1]

        full_text = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list])
        input_ids = self.tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
        
        labels = input_ids.clone()
        labels[0, :answer_start_index] = -100

        # --- 2. 处理图像 (调用你现有的 _load_image 逻辑) ---
        has_generated_image = self.generated_image_token in full_text
        image_slice = data["image"]
        if has_generated_image:
            image_slice = data["image"][:-1]
        
        image_dict = self._load_image(
            image_slice=image_slice, 
            max_pixels=self.max_pixels, 
            min_pixels=self.min_pixels, 
            processor=self.processor, 
            image_token=self.image_token, 
            factor=self.factor, 
            last_image=data["image"][-1] if has_generated_image else None,
            vae_image_transform=self.transform, 
            drop_prompt=False, 
            prompt=last_human_prompt.replace('<image>', '').replace('\n', ''),
            mask_weight_type=self.mask_weight_type, 
            siglip_processor=self.siglip_processor, 
            need_weight=data['need_weight'],
        )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        
        # --- 3. ✨ 新增：加载和处理 Ground Truth Mask ✨ ---
        ground_truth_masks = []
        if "mask" in data and data["mask"]:
            for mask_path in data["mask"]:
                try:
                    mask_img = Image.open(mask_path).convert('L')
                    _, _, height, width = image_dict['ref_pixel_values'].shape
                    # 定义一个将 mask 转换为 tensor 的 transform
                    mask_transform = transforms.Compose([
                        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.ToTensor()
                    ])
                    
                    mask_tensor = mask_transform(mask_img)
                    
                    # 二值化，确保值为 0.0 或 1.0
                    mask_tensor = (mask_tensor > 0.5).float()
                    ground_truth_masks.append(mask_tensor)
                except Exception as e:
                    print(f"[WARN] Failed to load or process mask at {mask_path}: {e}")
                    # 如果加载失败，添加一个全0的mask作为占位符，保证batch对齐
                    ground_truth_masks.append(torch.zeros(1, height, width))

        # 将 list of tensors stack 成一个 batch
        if ground_truth_masks:
            ground_truth_masks = torch.stack(ground_truth_masks)
        else:
            # 如果json中没有mask字段，返回一个有正确形状但batch size为0的空tensor
            ground_truth_masks = torch.empty(0, 1, height, width)


        # --- 4. 处理图像Token占位符 (调用你现有的 _process_image_token) ---
        image_position = []
        if len(image_token_lengths) > 0:
            input_ids, labels, image_position = self._process_image_token(
                input_ids=input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths,
            )

        # --- 5. 序列截断 (保持你的逻辑) ---
        max_sequence_length = 32768
        if input_ids.shape[1] > max_sequence_length:
            print(f"Warning: Sequence too long ({input_ids.shape[1]} > {max_sequence_length}), truncating...")
            original_len = input_ids.shape[1]
            input_ids = input_ids[:, -max_sequence_length:]
            labels = labels[:, -max_sequence_length:]
            offset = original_len - max_sequence_length
            if image_position:
                image_position = [pos - offset for pos in image_position if pos >= offset]

        # --- 6. 返回最终的、包含所有数据的字典 ---
        return_data = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "ground_truth_mask": ground_truth_masks, # <-- ✨ 新增的、处理好的mask
            "prompt": last_human_prompt,
            # 保持你其他的返回字段
            "ref_pixel_values": image_dict['ref_pixel_values'],
            "pil_pixel_values": image_dict['pil_pixel_values'], 
            "siglip_pixel_values": image_dict['siglip_pixel_values'], 
            "weights": image_dict['weights'], 
            "gt_text": data['conversations'][3]['value']
        }

        # 处理生成任务的目标图像
        if has_generated_image:
            image = Image.open(data["image"][-1]).convert("RGB")
            image, _, _ = letterbox_pil(image)
            image_tensor = torch.tensor(np.array(image)) / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = [] # 或者 torch.empty(0)
            
        return return_data

    @staticmethod
    def _load_image(
        image_slice: List[str],
        max_pixels: int = 448*448,  
        min_pixels: int = 448*448, 
        processor: Callable = None, 
        image_processor: Callable = None, 
        image_token_lengths: int = 729, 
        image_token: str = '<|image_pad|>', 
        factor: int = 1, 
        last_image: Optional[str] = None, 
        vae_image_transform: Callable = None,
        drop_prompt: bool = False, 
        prompt: str = '', 
        mask_weight_type: str = None, 
        siglip_processor: Callable = None, 
        need_weight: str = 'true', 
    ):
        resize_ref_image = False
        pil_pixel_values_last = []
        if last_image is not None:
            last_vision_infos = dict(
                image=last_image, min_pixels=min_pixels, max_pixels=max_pixels
                )
            last_image_inputs, last_video_inputs = process_vision_info([last_vision_infos], factor=factor)

            pil_pixel_values_last.append(last_image_inputs[0])
            
            if len(image_slice) > 0 and not all([has_same_resolution(image_path, last_image) for image_path in image_slice]):
                resize_ref_image = True
                resize_w, resize_h = last_image_inputs[0].size

        image_token_lengths = []
        pixel_values = []
        image_grid_thw = []
        ref_pixel_values = []  # Formerly condition_pixel_values, renamed to fix collator
        pil_pixel_values = []
        siglip_pixel_values = []
        # Ignore the last image (generated image)
        for image_path in image_slice: 
            # --- 1. Processing for QwenVL Encoder ---
            # This part remains the same, it prepares images for the language model's vision encoder.
            vision_infos = dict(image=image_path, min_pixels=min_pixels, max_pixels=max_pixels)
            if resize_ref_image:
                vision_infos.update(
                    dict(resized_height=resize_h, resized_width=resize_w)
                    )
            image_inputs, _ = process_vision_info([vision_infos], factor=factor)
            inputs = processor(text=[f'dummy {image_token}'], images=image_inputs, videos=None, padding=True, return_tensors="pt")
            
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)).sum()
                image_token_lengths.append(image_token_length)
            
            # This is for logging/masking and should use the Qwen-resized image
            pil_pixel_values.append(image_inputs[0])

            # --- 2. Processing for FLUX VAE Conditioning ---
            # This logic is now aligned with how `generated_image` is processed in `getitem`.
            raw_cond_image = Image.open(image_path).convert("RGB")
            # Convert to tensor C, H, W for the transform pipeline
            # raw_cond_tensor = torch.tensor(np.array(raw_cond_image), dtype=torch.float32) / 255.0
            # raw_cond_tensor = rearrange(raw_cond_tensor, "h w c -> c h w")
            # ★★★ 1. 新增：在此处调用函数进行缩放和填充 ★★★
            padded_cond_image = scale_and_pad_image(raw_cond_image, target_size=(256, 256))

            # ★★★ 2. 修改：使用填充后的图像 (padded_cond_image) 进行张量转换 ★★★
            raw_cond_tensor = torch.tensor(np.array(padded_cond_image), dtype=torch.float32) / 255.0
            raw_cond_tensor = rearrange(raw_cond_tensor, "h w c -> c h w")

            if vae_image_transform:
                # Apply the full transform (resize + norm) passed from getitem
                transformed_cond_tensor = vae_image_transform(raw_cond_tensor)
            else:
                # Fallback if no transform is provided
                transformed_cond_tensor = (raw_cond_tensor - 0.5) / 0.5

            # Add batch dimension for collation
            transformed_cond_tensor = transformed_cond_tensor.unsqueeze(0)

            if drop_prompt:
                ref_pixel_values.append(torch.zeros_like(transformed_cond_tensor))
            else:
                ref_pixel_values.append(transformed_cond_tensor)

            # --- 3. Optional SigLIP processing ---
            if siglip_processor is not None:
                # siglip_pixel_value = siglip_processor.preprocess(
                #             images=raw_cond_image, # Use raw image
                #             do_resize=True, return_tensors="pt", do_convert_rgb=True
                #         ).pixel_values  # 1 c h w
                siglip_pixel_value = siglip_processor.preprocess(
                    images=padded_cond_image,
                    do_resize=True, 
                    return_tensors="pt", 
                    do_convert_rgb=True
                ).pixel_values

                if drop_prompt:
                    siglip_pixel_values.append(torch.zeros_like(siglip_pixel_value))
                else:
                    siglip_pixel_values.append(siglip_pixel_value)
            
        # if multi-image in a sample, concat them
        # assume pixel_values[0] (n1, 1176), pixel_values[1] (n2, 1176), pixel_values will be (n1+n2, 1176)
        if len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values)
            image_grid_thw = torch.concat(image_grid_thw)  # (b, 3), 3 mean the grid of t, h, w
        if len(ref_pixel_values) > 0:
            ref_pixel_values = torch.cat(ref_pixel_values, dim=0)

        if len(siglip_pixel_values) > 0: 
            siglip_pixel_values = torch.concat(siglip_pixel_values)  # b c h w

        pil_pixel_values = pil_pixel_values + pil_pixel_values_last
        
        if mask_weight_type is not None:
            _, weights = get_weight_mask(pil_pixel_values, prompt, mask_weight_type, need_weight)
            if need_weight.lower() == 'false':
                assert torch.all(weights == 1)
        else:
            weights = []
        return {
            'pixel_values': pixel_values, 
            'image_grid_thw': image_grid_thw, 
            'image_token_lengths': image_token_lengths, 
            'ref_pixel_values': ref_pixel_values,
            'pil_pixel_values': pil_pixel_values, 
            'siglip_pixel_values': siglip_pixel_values, 
            'weights': weights, 
        }

    @staticmethod
    def _process_image_token(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        image_token_id: int,
        image_begin_token_id: int,
        image_end_token_id: int,
        image_token_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        A robust and safe version to process image tokens.
        It expands image placeholders in input_ids and masks them correctly in labels.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("This function is designed to process one sample at a time.")

        # 将 tensors 转换为 list 以便灵活操作
        input_list = input_ids.squeeze(0).tolist()
        labels_list = labels.squeeze(0).tolist()
        
        image_positions = []
        offset = 0
        img_idx = 0

        # 从头开始查找单个的 <image> token (151654)
        i = 0
        while i < len(input_list):
            if input_list[i] == image_token_id:
                if img_idx >= len(image_token_lengths):
                    # 如果图像占位符比提供的图像数量还多，则跳过
                    i += 1
                    continue
                
                # 获取当前图像占位符的长度
                current_img_len = image_token_lengths[img_idx]
                
                # 创建要插入的图像块
                image_block = [image_begin_token_id] + [image_token_id] * current_img_len + [image_end_token_id]
                image_block_len = len(image_block)

                # 创建用于 labels 的 -100 屏蔽块
                ignore_block = [-100] * image_block_len

                # 替换 input_list 和 labels_list 中的单个 <image> token
                input_list = input_list[:i] + image_block + input_list[i+1:]
                labels_list = labels_list[:i] + ignore_block + labels_list[i+1:]
                
                # 记录图像在序列中的起始位置 (begin token 之后)
                image_positions.append(i + 1)
                
                # 更新索引和偏移量
                i += image_block_len
                offset += image_block_len - 1
                img_idx += 1
            else:
                i += 1
        
        # 将 list 转回 tensor
        final_input_ids = torch.tensor([input_list], dtype=input_ids.dtype)
        final_labels = torch.tensor([labels_list], dtype=labels.dtype)

        return final_input_ids, final_labels, image_positions

    

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = 28) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    return image

def process_vision_info(
    vision_infos: list,
    return_video_kwargs: bool = False,
    factor: int = 1, 
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28*factor))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
import sentencepiece as sp
import lmdb
import pickle
from typing import Any, Callable, Optional, List, Union
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import json
import numpy as np
from einops import rearrange
import random
from univa.utils.prompter import Prompter
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN
from univa.utils.get_mask import get_weight_mask
from univa.utils.get_ocr import get_ocr_result
from fractions import Fraction
from io import BytesIO
import base64
import requests
import re # ADDED: Import regular expressions module

# ==============================================================================
# Utility Functions (assumed to be present as in your original code)
# ==============================================================================

def letterbox_pil(
    img: Image.Image,
    target_size=(256, 256),
    fill=(255, 255, 255),
    resample=Image.Resampling.LANCZOS,
):
    """
    等比缩放 + padding 到 target_size.
    返回: new_img (PIL), scale, (pad_left, pad_top)
    """
    tw, th = target_size
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
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

def _get_pil_image(image_source: Union[str, Image.Image]) -> Image.Image:
    """Helper function to convert an image source (path or PIL Image) to a PIL.Image.Image object."""
    if isinstance(image_source, str):
        return Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    else:
        raise TypeError(f"Expected image source as str or PIL.Image.Image, got {type(image_source)}")

def has_same_resolution(img1, img2):
    return img1.size == img2.size

# ... (Include all other necessary helper functions like find_subsequence_positions, 
#      Step1XTokenizer, get_aspect_ratio, to_rgb, smart_resize, fetch_image, process_vision_info, etc.)
# NOTE: For brevity, other helper functions are omitted but should be included here.
def find_subsequence_positions(seq: List[int], subseq: List[int]) -> List[int]:
    if len(subseq) == 0 or len(seq) < len(subseq):
        return []
    starts = []
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            starts.append(i)
    return starts

class Step1XTokenizer:
    pass

def get_aspect_ratio(img):
    pass

def has_same_aspect_ratio(img1, img2):
    pass

def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    return image.convert('RGB')

def smart_resize(height, width, factor=32, min_pixels=None, max_pixels=None):
    if max_pixels and height * width > max_pixels:
        ratio = (max_pixels / (height * width))**0.5
        height, width = int(height * ratio), int(width * ratio)
    if min_pixels and height * width < min_pixels:
        ratio = (min_pixels / (height * width))**0.5
        height, width = int(height * ratio), int(width * ratio)
    height = (height // factor) * factor
    width = (width // factor) * factor
    return height, width

def fetch_image(ele: dict, size_factor: int = 28) -> Image.Image:
    image = ele["image"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    else:
        image_obj = Image.open(image)
    image = to_rgb(image_obj)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=size_factor)
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels")
        max_pixels = ele.get("max_pixels")
        resized_height, resized_width = smart_resize(height, width, factor=size_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)
    return image

def process_vision_info(vision_infos: list, return_video_kwargs: bool = False, factor: int = 1):
    image_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=28 * factor))
    return image_inputs, None

# ==============================================================================
# The New Merged and Improved Dataset Class
# ==============================================================================

class LMDB_Qwen_Dataset(Dataset):
    """
    A PyTorch Dataset that reads data from an LMDB database and dynamically
    processes it for multilingual translation tasks based on the data source.
    """
    
    # ADDED: A mapping from language codes to full language names.
    LANG_CODE_TO_NAME = {
        'de': 'German',
        'cs': 'Czech',
        'fr': 'French',
        'ro': 'Romanian',
        'ru': 'Russian',
    }
    
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        tokenizer: PreTrainedTokenizer,
        image_processor: Callable,
        transform: Callable,
        processor: Callable,
        prompter: Prompter,
        min_pixels: int = 384*384,
        max_pixels: int = 384*384,
        mask_weight_type: str = 'log',
        siglip_processor: Callable = None,
        notry: bool = False,
        image_token_length: int = 729,
        only_generated_task: bool = False,
        drop_prompt_rate: float = 0.0,
        joint_ref_feature: bool = False,
        anyres: bool = False, 
        ocr_enhancer: bool = False, 
        random_data: bool = False, 
        maxnum_per_data: int = -1, 
        use_step1x_preprocessing: bool = True,
    ):
        super().__init__()
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]
            
        self.data = []
        self._load_data(-1)
        
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
        
        self.factor = 2
        self.mask_weight_type = mask_weight_type
        self.siglip_processor = siglip_processor
        self.notry = notry
        
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(self.image_begin_token)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(self.image_end_token)
        assert isinstance(self.image_begin_token_id, int) and isinstance(self.image_end_token_id, int)

    def _load_data(self, maxnum_per_data=-1):
        # CHANGED: This method now extracts the language code from the file path.
        lmdb_path_list = []
        dataset_lang_codes = []  # To store lang code for each dataset/environment

        for dataset in self.datasets:
            image_root, lmdb_file, need_weight = dataset.split(",")
            lmdb_path_list.append(lmdb_file)
            
            # ADDED: Use regex to find the language code (e.g., 'de', 'cs') from the path.
            lang_code_match = re.search(r'subset_en_(\w+)_lmdb', lmdb_file)
            if lang_code_match:
                lang_code = lang_code_match.group(1)
                dataset_lang_codes.append(lang_code)
                print(f"Detected language code '{lang_code}' for {lmdb_file}")
            else:
                dataset_lang_codes.append(None) # Append None if no code is found
                print(f"Warning: Could not detect language code for {lmdb_file}")

        self.envs = [lmdb.open(path, readonly=True, lock=False) for path in tqdm(lmdb_path_list)]
        self.env_keys = []
        for env_id in range(len(self.envs)):
            lang_code = dataset_lang_codes[env_id]
            env = self.envs[env_id]
            with env.begin(write=False) as txn:
                keys = list(txn.cursor().iternext(values=False))
                for key in keys:
                    # CHANGED: Store lang_code along with env_id and key.
                    self.env_keys.append((env_id, key, lang_code))
        
        print(f"Load {len(self.env_keys)} data from file.")
        
    def __len__(self):
        return len(self.env_keys)

    def __getitem__(self, idx):
        if self.notry:
            return self.process_item(idx)
        try:
            return self.process_item(idx)
        except Exception as e:
            print(f'Error processing index {idx}: {e}. Trying a random sample.')
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

    def process_item(self, idx):
        # CHANGED: The prompt is now created dynamically based on the item's language code.
        
        # === 步骤 1: 从 LMDB 提取数据和语言代码 ===
        env_id, env_key, lang_code = self.env_keys[idx] # CHANGED: Unpack language code

        # ADDED: Dynamically generate the human prompt based on the language code.
        language_name = self.LANG_CODE_TO_NAME.get(lang_code, f"language '{lang_code}'")
        human_prompt_value = f"<image>\\nTranslate all English texts into {language_name}."
        
        with self.envs[env_id].begin(write=False) as txn:
            value = txn.get(env_key)
            lmdb_data = pickle.loads(value)

        # === 步骤 2: 获得编辑前后的图像和文本 ===
        src_text = lmdb_data["src_text"].decode().strip()
        tgt_text = lmdb_data["tgt_text"].decode().strip()
        src_img = Image.frombytes(lmdb_data["src_img_mode"], lmdb_data["src_img_size"], lmdb_data["src_img"]).convert("RGB")
        tgt_img = Image.frombytes(lmdb_data["tgt_img_mode"], lmdb_data["tgt_img_size"], lmdb_data["tgt_img"]).convert("RGB")
        
        # 1. 重建 `conversations` 列表
        conversations_for_processing = [
            # CHANGED: Use the dynamically generated prompt.
            {"from": "human", "value": human_prompt_value},
            {"from": "gpt", "value": self.generated_image_token},
            {"from": "ocr", "value": src_text},
            {"from": "translation", "value": tgt_text},
        ]

        # The rest of the function remains the same...
        conversations = []
        last_human_prompt = ""
        for item in conversations_for_processing:
            role = ""
            if item["from"] == "human":
                role = self.prompter.user_role
                last_human_prompt = item["value"]
            elif item["from"] == "gpt":
                role = self.prompter.assistant_role
            elif item['from'] == "translation":
                role = self.prompter.translation_role
            elif item['from'] == "ocr":
                continue
            else:
                raise ValueError(f"Unknown role: {item['from']}")
            value = item["value"]
            conversations.append({"from": role, "value": value})
        
        prompt_list = self.prompter.get_train_prompt(conversations)
        
        last_answer_item_index = -1
        for i in range(len(prompt_list) - 1, -1, -1):
            if prompt_list[i].get("is_labels", False):
                last_answer_item_index = i
                break
        if last_answer_item_index == -1:
            raise ValueError("No turn with 'is_labels: True' found in the prompt_list.")
        
        text_before_answer = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list[:last_answer_item_index]])
        tokens_before_answer = self.tokenizer(text_before_answer, return_tensors="pt", truncation=False)
        answer_start_index = tokens_before_answer.input_ids.shape[1]
        
        full_text = "".join([item['prompt'].replace('<image>', self.image_token) for item in prompt_list])
        input_ids = self.tokenizer(full_text, return_tensors="pt", truncation=False).input_ids
        
        labels = input_ids.clone()
        labels[0, :answer_start_index] = -100
        
        has_generated_image = self.generated_image_token in full_text
        
        image_slice_for_load_image = [src_img]
        last_image_for_load_image = tgt_img if has_generated_image else None

        image_dict = self._load_image(
            image_slice_for_load_image,
            self.max_pixels, self.min_pixels,
            processor=self.processor, image_token=self.image_token,
            factor=self.factor,
            last_image=last_image_for_load_image,
            vae_image_transform=self.transform,
            drop_prompt=False,
            prompt=last_human_prompt.replace('', '').replace('\\n', ''), 
            mask_weight_type=self.mask_weight_type,
            siglip_processor=self.siglip_processor,
            need_weight=lmdb_data.get('need_weight', 'False'), 
        )
        
        image_token_lengths = image_dict['image_token_lengths']
        pixel_values = image_dict['pixel_values']
        image_grid_thw = image_dict['image_grid_thw']
        
        image_position = []
        if len(image_token_lengths) > 0:
            input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths,
            )

        max_sequence_length = 32768
        if input_ids.shape[1] > max_sequence_length:
            print(f"Warning: Sequence too long ({input_ids.shape[1]} > {max_sequence_length}), truncating...")
            original_len = input_ids.shape[1]
            input_ids = input_ids[:, -max_sequence_length:]
            labels = labels[:, -max_sequence_length:]
            offset = original_len - max_sequence_length
            if image_position:
                image_position = [pos - offset for pos in image_position if pos >= offset]

        return_data = {
            "input_ids": input_ids,
            "labels": labels, # bug
            "pixel_values": pixel_values,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "prompt": last_human_prompt,
            "ref_pixel_values": image_dict['ref_pixel_values'],
            "pil_pixel_values": image_dict['pil_pixel_values'],
            "siglip_pixel_values": image_dict['siglip_pixel_values'],
            "weights": image_dict['weights'],
        }
        
        if has_generated_image:
            tgt_img, _, _ = letterbox_pil(tgt_img)
            image_tensor = torch.tensor(np.array(tgt_img)) / 255.0
            image_tensor = rearrange(image_tensor, "h w c -> c h w")
            return_data["generated_image"] = self.transform(image_tensor)
        else:
            return_data["generated_image"] = []
            
        return return_data
        
    @staticmethod
    def _load_image(
        image_slice: List[Union[str, Image.Image]],
        max_pixels: int = 448*448,
        min_pixels: int = 448*448,
        processor: Callable = None, 
        image_processor: Callable = None, 
        image_token_lengths_param: int = 729,
        image_token: str = '<|image_pad|>', 
        factor: int = 1, 
        last_image: Optional[Union[str, Image.Image]] = None,
        vae_image_transform: Callable = None,
        drop_prompt: bool = False, 
        prompt: str = '', 
        mask_weight_type: str = None, 
        siglip_processor: Callable = None, 
        need_weight: str = 'true', 
    ):
        resize_ref_image = False
        pil_pixel_values_last = []
        
        last_image_pil = None
        if last_image is not None:
            last_image_pil = _get_pil_image(last_image)
            last_image_pil, _, _ = letterbox_pil(last_image_pil, target_size=(256, 256))
            last_vision_infos = dict(image=last_image_pil, min_pixels=min_pixels, max_pixels=max_pixels)
            last_image_inputs, _ = process_vision_info([last_vision_infos], factor=factor)
            pil_pixel_values_last.append(last_image_inputs[0])

            if len(image_slice) > 0:
                all_same_resolution = True
                for img_src in image_slice:
                    current_slice_image_pil = _get_pil_image(img_src)
                    current_slice_image_pil, _, _ = letterbox_pil(current_slice_image_pil, target_size=(256, 256))
                    if not has_same_resolution(current_slice_image_pil, last_image_pil):
                        all_same_resolution = False
                        break
                if not all_same_resolution:
                    resize_ref_image = True
                    resize_w, resize_h = last_image_inputs[0].size
        
        image_token_lengths, pixel_values, image_grid_thw, ref_pixel_values, pil_pixel_values, siglip_pixel_values = [], [], [], [], [], []

        for image_src in image_slice:
            current_image_pil = _get_pil_image(image_src)
            current_image_pil, _, _ = letterbox_pil(current_image_pil, target_size=(256, 256))
            
            vision_infos = dict(image=current_image_pil, min_pixels=min_pixels, max_pixels=max_pixels)
            if resize_ref_image:
                vision_infos.update(dict(resized_height=resize_h, resized_width=resize_w))
            
            image_inputs, _ = process_vision_info([vision_infos], factor=factor)
            inputs = processor(text=[f'dummy {image_token}'], images=image_inputs, videos=None, padding=True, return_tensors="pt")
            
            if not drop_prompt:
                pixel_values.append(inputs.pixel_values)
                image_grid_thw.append(inputs.image_grid_thw)
                image_token_length = (inputs.input_ids[0] == processor.tokenizer.convert_tokens_to_ids(image_token)).sum().item()
                image_token_lengths.append(image_token_length)
            
            pil_pixel_values.append(image_inputs[0])
            raw_cond_image = current_image_pil
            raw_cond_tensor = torch.tensor(np.array(raw_cond_image), dtype=torch.float32) / 255.0
            raw_cond_tensor = rearrange(raw_cond_tensor, "h w c -> c h w")
            transformed_cond_tensor = vae_image_transform(raw_cond_tensor) if vae_image_transform else (raw_cond_tensor - 0.5) / 0.5
            transformed_cond_tensor = transformed_cond_tensor.unsqueeze(0)
            
            ref_pixel_values.append(torch.zeros_like(transformed_cond_tensor) if drop_prompt else transformed_cond_tensor)
            
            if siglip_processor is not None:
                siglip_pixel_value = siglip_processor.preprocess(images=raw_cond_image, do_resize=True, return_tensors="pt", do_convert_rgb=True).pixel_values
                siglip_pixel_values.append(torch.zeros_like(siglip_pixel_value) if drop_prompt else siglip_pixel_value)
        
        pixel_values = torch.concat(pixel_values) if pixel_values else torch.empty(0)
        image_grid_thw = torch.concat(image_grid_thw) if image_grid_thw else torch.empty(0, 3)
        ref_pixel_values = torch.cat(ref_pixel_values, dim=0) if ref_pixel_values else torch.empty(0)
        siglip_pixel_values = torch.concat(siglip_pixel_values) if siglip_pixel_values else torch.empty(0)
        pil_pixel_values = pil_pixel_values + pil_pixel_values_last

        weights = []
        if mask_weight_type is not None:
            _, weights = get_weight_mask(pil_pixel_values, prompt, mask_weight_type, need_weight)
            if need_weight.lower() == 'false':
                assert torch.all(weights == 1)

        return {
            'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw, 
            'image_token_lengths': image_token_lengths, 'ref_pixel_values': ref_pixel_values,
            'pil_pixel_values': pil_pixel_values, 'siglip_pixel_values': siglip_pixel_values, 
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
        if input_ids.shape[0] != 1:
            raise ValueError("This function processes one sample at a time.")
        
        input_list = input_ids.squeeze(0).tolist()
        labels_list = labels.squeeze(0).tolist()
        
        image_positions = []
        img_idx = 0
        i = 0
        while i < len(input_list):
            if input_list[i] == image_token_id:
                if img_idx >= len(image_token_lengths):
                    i += 1
                    continue
                
                current_img_len = image_token_lengths[img_idx]
                image_block = [image_begin_token_id] + [image_token_id] * current_img_len + [image_end_token_id]
                image_block_len = len(image_block)
                ignore_block = [-100] * image_block_len
                
                input_list = input_list[:i] + image_block + input_list[i+1:]
                labels_list = labels_list[:i] + ignore_block + labels_list[i+1:]
                
                image_positions.append(i + 1)
                
                i += image_block_len
                img_idx += 1
            else:
                i += 1
        
        final_input_ids = torch.tensor([input_list], dtype=input_ids.dtype)
        final_labels = torch.tensor([labels_list], dtype=labels.dtype)
        
        return final_input_ids, final_labels, image_positions


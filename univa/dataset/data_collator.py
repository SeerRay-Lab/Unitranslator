from typing import List, Dict

from transformers import PreTrainedTokenizer

import torch
import torch.nn.functional as F

import re
import unicodedata

T5_TEMPLATE = (
    "Replace all text in the image with the EXACT string below.\n"
    "Do NOT translate. Copy characters exactly, including punctuation and spacing.\n"
    'TARGET: "{target}"'
)

def normalize_target_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def escape_for_quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def build_t5_prompt(gt: str) -> str:
    gt = escape_for_quote(normalize_target_text(gt))
    return T5_TEMPLATE.format(target=gt)
    
def pad_list_of_tensors(tensor_list, padding_value=0):
    # tensor_list: list of tensors, each of shape (b, c, h, w)
    # if all empty list, which means all data are t2i within this batch
    if all(not isinstance(tensor, torch.Tensor) for tensor in tensor_list):
        return []
    else:
        for tmp_tensor in tensor_list:
            if isinstance(tmp_tensor, torch.Tensor):
                # find a tensor
                break
        # this line pad zero_tensor when batch mixed between t2i and others.
        # t2i can be considered a uncondition (no-reference image) editing
        tensor_list = [
            torch.zeros_like(tmp_tensor) if not isinstance(tensor, torch.Tensor) else tensor for tensor in tensor_list
            ]
    assert all(tensor.shape[1] == tensor_list[0].shape[1] for tensor in tensor_list)
    # 找到最大的 b, h, w
    max_b = max(tensor.shape[0] for tensor in tensor_list)
    max_c = tensor_list[0].shape[1]  # 假设c都是一样的
    max_h = max(tensor.shape[2] for tensor in tensor_list)
    max_w = max(tensor.shape[3] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        b, c, h, w = tensor.shape
        pad_b = max_b - b
        pad_h = max_h - h
        pad_w = max_w - w
        # 先 pad h, w (最后两维)
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=padding_value)
        # 再 pad b 维（最前面），要扩成 (max_b, c, h, w)
        if pad_b > 0:
            padding_shape = (pad_b, c, max_h, max_w)
            pad_tensor = torch.full(padding_shape, fill_value=padding_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad_tensor], dim=0)
        padded_tensors.append(tensor)
    # 最后 stack 成 (B, b_max, c, h_max, w_max)
    return torch.stack(padded_tensors)

def resize_list_of_tensors(weights):
    # suppose weights is your list of [1, H, W] tensors
    # 1) find the max height and width
    heights = [w.shape[-2] for w in weights]
    widths  = [w.shape[-1] for w in weights]
    max_h, max_w = max(heights), max(widths)
    # 2) interpolate each mask to (max_h, max_w)
    resized = []
    for w in weights:
        # F.interpolate expects a 4D tensor: (N, C, H, W)
        w_4d = w.unsqueeze(0)        # -> [1, 1, H, W]
        w_4d = w_4d.unsqueeze(0) if w_4d.ndim == 3 else w_4d
        # but since w is already [1,H,W], unsqueeze once is enough:
        # w_4d = w.unsqueeze(0) # [1, 1, H, W]
        w_resized = F.interpolate(
            w_4d, size=(max_h, max_w), mode='nearest'
        )
        # back to [1, H', W']
        w_resized = w_resized.squeeze(0)
        resized.append(w_resized)
    # 3) stack into a single tensor [N, 1, max_h, max_w]
    weights = torch.stack(resized)  # -> [N, 1, max_h, max_w]
    return weights

class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, padding_side="left"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def _to_1d(self, x):
        # Accept [L] or [1, L]
        if isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == 1:
            return x[0]
        return x

    def _pad_1d(self, seqs, pad_value: int):
        # seqs: list[Tensor[L]]
        lengths = [s.numel() for s in seqs]
        max_len = max(lengths) if lengths else 0

        out = seqs[0].new_full((len(seqs), max_len), pad_value)  # dtype/device match
        for i, s in enumerate(seqs):
            l = s.numel()
            if self.padding_side == "left":
                out[i, max_len - l : max_len] = s
            else:
                out[i, :l] = s
        return out

    def __call__(self, instances: List[Dict]) -> Dict:
        input_ids_list = [self._to_1d(ins["input_ids"]) for ins in instances]
        labels_list    = [self._to_1d(ins["labels"]) for ins in instances]

        # pad input_ids / labels consistently
        input_ids = self._pad_1d(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        labels    = self._pad_1d(labels_list,    pad_value=-100)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Optional fields
        image_position   = [ins.get("image_position") for ins in instances]
        prompts          = [ins.get("prompt") for ins in instances]
        pil_pixel_values = [ins.get("pil_pixel_values") for ins in instances]

        # These depend on your per-sample shape conventions; keep your original logic,
        # but normalize empty returns to None for type stability.
        pixel_values_list = [ins.get("pixel_values") for ins in instances]
        pixel_values_list = [v for v in pixel_values_list if isinstance(v, torch.Tensor)]
        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None

        image_grid_thw_list = [ins.get("image_grid_thw") for ins in instances]
        image_grid_thw_list = [v for v in image_grid_thw_list if isinstance(v, torch.Tensor)]
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None

        ref_pixel_values_list = [ins.get("ref_pixel_values") for ins in instances]
        ref_pixel_values = pad_list_of_tensors(ref_pixel_values_list, padding_value=0)  # assuming this handles None

        siglip_list = [ins.get("siglip_pixel_values") for ins in instances]
        siglip_list = [v for v in siglip_list if isinstance(v, torch.Tensor)]
        siglip_pixel_values = torch.cat(siglip_list, dim=0) if siglip_list else None

        weights_list = [ins.get("weights") for ins in instances]
        weights_list = [v for v in weights_list if isinstance(v, torch.Tensor)]
        weights = torch.stack(weights_list) if weights_list and all(w.shape == weights_list[0].shape for w in weights_list) else (weights_list if weights_list else None)

        gen_list = [ins.get("generated_image") for ins in instances]
        gen_list = [v for v in gen_list if isinstance(v, torch.Tensor)]
        generated_image = torch.stack(gen_list) if gen_list and all(g.shape == gen_list[0].shape for g in gen_list) else (gen_list if gen_list else None)

        gt_mask_list = [ins.get("ground_truth_mask") for ins in instances]
        gt_mask_list = [v for v in gt_mask_list if isinstance(v, torch.Tensor)]
        ground_truth_mask = torch.stack(gt_mask_list) if gt_mask_list else None

        gt_text = [ins.get("gt_text", "") for ins in instances]



        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_position": image_position,
            "image_grid_thw": image_grid_thw,
            "prompts": prompts,
            "ref_pixel_values": ref_pixel_values,
            "pil_pixel_values": pil_pixel_values,
            "siglip_pixel_values": siglip_pixel_values,
            "weights": weights,
            "generated_image": generated_image,
            "ground_truth_mask": ground_truth_mask,
            "gt_text": gt_text,
            "t5_text": [build_t5_prompt(t) for t in gt_text],
        }


# class DataCollator:
#     def __init__(self, tokenizer: PreTrainedTokenizer, padding_side='left'):
#         self.tokenizer = tokenizer
#         self.padding_side = padding_side

#     def __call__(self, instances: List[Dict]) -> Dict:
#         # Assuming these keys are mandatory
#         input_ids = [instance["input_ids"][0] for instance in instances]
#         labels = [instance["labels"][0] for instance in instances]
        
#         # Safely get potentially missing keys
#         image_position = [instance.get("image_position") for instance in instances]
#         prompts = [instance.get("prompt") for instance in instances]
#         pil_pixel_values = [instance.get("pil_pixel_values") for instance in instances]

#         # --- Safely process list/tensor fields ---
        
#         pixel_values_list = [v for v in (instance.get("pixel_values") for instance in instances) if v is not None and len(v) > 0]
#         pixel_values = torch.cat(pixel_values_list) if pixel_values_list else None
        
#         image_grid_thw_list = [v for v in (instance.get("image_grid_thw") for instance in instances) if v is not None and len(v) > 0]
#         image_grid_thw = torch.cat(image_grid_thw_list) if image_grid_thw_list else None

#         ref_pixel_values_list = [instance.get("ref_pixel_values") for instance in instances]
#         ref_pixel_values = pad_list_of_tensors(ref_pixel_values_list, padding_value=0)

#         siglip_pixel_values_list = [v for v in (instance.get("siglip_pixel_values") for instance in instances) if v is not None and len(v) > 0]
#         siglip_pixel_values = torch.cat(siglip_pixel_values_list, dim=0) if siglip_pixel_values_list else []

#         weights_list = [v for v in (instance.get("weights") for instance in instances) if v is not None and len(v) > 0]
#         if weights_list:
#             if all(i.shape == weights_list[0].shape for i in weights_list):
#                 weights = torch.stack(weights_list)
#             else:
#                 weights = [i.unsqueeze(0) for i in weights_list]
#         else:
#             weights = None
            
#         generated_image_list = [v for v in (instance.get("generated_image") for instance in instances) if v is not None and len(v) > 0]
#         if generated_image_list:
#             if all(i.shape == generated_image_list[0].shape for i in generated_image_list):
#                 generated_image = torch.stack(generated_image_list)
#             else:
#                 generated_image = [i.unsqueeze(0) for i in generated_image_list]
#         else:
#             generated_image = []

#         ground_truth_mask_list = [v for v in (instance.get("ground_truth_mask") for instance in instances) if v is not None and len(v) > 0]
#         if ground_truth_mask_list:
#             ground_truth_mask = torch.stack(ground_truth_mask_list)
#         else:
#             ground_truth_mask = None # Set to None if not found

#         # --- Padding and final dict construction ---
        
#         # Note: Original code used padding_side for pad_sequence, but it's not a valid argument.
#         # It should be applied during tokenization. I am removing it here to prevent errors.
#         # If you need right-padding, ensure your tokenizer is configured with `padding_side='right'`.
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=-100
#         )
#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
#         return {
#             "input_ids": input_ids,
#             "pixel_values": pixel_values,
#             "labels": labels,
#             "attention_mask": attention_mask,
#             "image_position": image_position,
#             "image_grid_thw": image_grid_thw,
#             "prompts": prompts,
#             "ref_pixel_values": ref_pixel_values,
#             "pil_pixel_values": pil_pixel_values,
#             "siglip_pixel_values": siglip_pixel_values,
#             "weights": weights,
#             "generated_image": generated_image,
#             "ground_truth_mask": ground_truth_mask,
#         }

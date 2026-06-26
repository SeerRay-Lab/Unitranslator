import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import easyocr
from torchvision.ops import roi_align
import numpy as np

class MultilingualOCRLoss(nn.Module):
    # ... (docstring and other parts) ...
    def __init__(self, lang_list: list, device):
        super().__init__()
        print(f"Initializing Multilingual OCR Loss (Unwrapped Version) on device: {device}")
        self.device = device
        self.max_text_len = 100
        
        try:
            self.reader = easyocr.Reader(lang_list, gpu=True)
        except Exception as e:
            print(f"Failed to initialize EasyOCR. Error: {e}")
            raise
            
        self.recognizer = self.reader.recognizer

        # ======================= 核心修正 #2：解包 DataParallel =======================
        # 检查 recognizer 是否被 nn.DataParallel 包装，如果是，则解包，获取真正的模型
        if isinstance(self.recognizer, torch.nn.DataParallel):
            print("Unwrapping the recognizer model from nn.DataParallel wrapper.")
            self.recognizer = self.recognizer.module
        # =========================================================================

        # 现在，将（已经解包的）纯净模型移动到当前进程分配到的 device
        self.recognizer.to(device)

        self.converter = self.reader.converter
        self.character_set = set(self.converter.character)
        print(f"OCR character set loaded. Total characters: {len(self.character_set)}")

        self.recognizer.eval()
        for param in self.recognizer.parameters():
            param.requires_grad = False
        print(f"EasyOCR 'recognizer' model has been moved to {device}, frozen, and set to evaluation mode.")
            
        self.ctc_loss = nn.CTCLoss(blank=self.converter.character.index('-'), reduction='mean', zero_infinity=True)

    # ... forward 函数和 _filter_unsupported_chars 函数保持不变 ...
    def _filter_unsupported_chars(self, text: str) -> str:
        """
        一个辅助函数，用于移除所有不在 OCR 字典中的字符。
        """
        return "".join([char for char in text if char in self.character_set])

    def forward(self, decoded_pred_images: torch.Tensor, gt_texts: list[str]) -> torch.Tensor:
        if decoded_pred_images.shape[0] == 0 or not gt_texts:
            return torch.tensor(0.0, device=self.device)
        image_list_for_detection = []
        with torch.no_grad():
            temp_images_batch = (decoded_pred_images / 2 + 0.5).clamp(0, 1)
            temp_images_batch = temp_images_batch.cpu().permute(0, 2, 3, 1).numpy()
            temp_images_batch = (temp_images_batch * 255).astype(np.uint8)
            for i in range(temp_images_batch.shape[0]):
                image_list_for_detection.append(temp_images_batch[i])

        all_cropped_images = []
        all_gt_texts_for_crops = []
        
        for i in range(len(image_list_for_detection)):
            single_image_np = image_list_for_detection[i]
            gt_text_for_image = self._filter_unsupported_chars(gt_texts[i])
            
            if not gt_text_for_image:
                continue
            
            main_bbox_coords = None
            with torch.no_grad():
                try:
                    horizontal_list, free_list = self.reader.detect(single_image_np, text_threshold=0.3, low_text=0.2)
                    bboxes = horizontal_list + free_list
                    if bboxes:
                        points = bboxes[0]
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        main_bbox_coords = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                except Exception:
                    pass 

            if main_bbox_coords is None:
                B, C, H, W = decoded_pred_images.shape
                box_h, box_w = H * 0.8, W * 0.9
                center_h, center_w = H // 2, W // 2
                x1 = center_w - box_w / 2
                y1 = center_h - box_h / 2
                x2 = center_w + box_w / 2
                y2 = center_h + box_h / 2
                main_bbox_coords = [x1, y1, x2, y2]
                
            main_bbox_tensor = torch.tensor([main_bbox_coords], device=self.device, dtype=decoded_pred_images.dtype)
            cropped_tensor = roi_align(decoded_pred_images[i:i+1], [main_bbox_tensor], output_size=(64, 320))
            all_cropped_images.append(cropped_tensor)
            all_gt_texts_for_crops.append(gt_text_for_image)

        if not all_cropped_images:
            return torch.tensor(0.0, device=self.device)

        batched_crops = torch.cat(all_cropped_images, dim=0)
        batched_crops_gray = TF.rgb_to_grayscale(batched_crops)
        
        batch_size_crops = batched_crops_gray.shape[0]
        dummy_text = torch.LongTensor(batch_size_crops, self.max_text_len).fill_(0).to(self.device)
        
        preds = self.recognizer(batched_crops_gray, dummy_text)
        
        preds_seq_len = preds.size(1)
        preds_size = torch.IntTensor([preds_seq_len] * batch_size_crops).to(self.device)
        
        targets, targets_len = self.converter.encode(all_gt_texts_for_crops)
        
        log_probs = F.log_softmax(preds, dim=2).permute(1, 0, 2)
        
        targets_len = targets_len.to(self.device)
        targets = targets.to(self.device)
        
        if torch.any(targets_len > preds_size.max()):
            return torch.tensor(0.0, device=self.device)
            
        loss = self.ctc_loss(log_probs, targets, preds_size, targets_len)
        
        if torch.isinf(loss) or torch.isnan(loss):
            return torch.tensor(0.0, device=self.device)
        return loss

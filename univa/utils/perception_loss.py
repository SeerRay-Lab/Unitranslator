# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Dict

# def get_nested_module(model, module_path: str):
#     """辅助函数：安全地获取嵌套模块"""
#     modules = module_path.split('.')
#     for mod_name in modules:
#         model = getattr(model, mod_name)
#     return model

# class PerceptualLoss(nn.Module):
#     """
#     感知损失计算模块 (最终版)
#     - 内部处理图像尺寸调整 (Resize)
#     - 内部处理视频维度适配 (repeat frames)
#     """
#     def __init__(
#         self,
#         main_model: nn.Module,
#         vision_encoder_path: str,
#         loss_fn: nn.Module = nn.L1Loss(),
#         target_layers: Optional[List[int]] = None
#     ):
#         super().__init__()
#         print("Initializing Perceptual Loss (FINAL CONFIRMED VERSION)...")
#         self.vision_encoder = get_nested_module(main_model, vision_encoder_path)
#         self.vision_encoder.eval()
#         for param in self.vision_encoder.parameters():
#             param.requires_grad = False
#         self.loss_fn = loss_fn
#         if target_layers is None:
#             self.target_layers = [8, 16, 24, 31]
#         else:
#             self.target_layers = target_layers
#         self.features: Dict[int, torch.Tensor] = {}
#         # 预先定义好模型期望的尺寸
#         self.target_resolution = (448, 448)
#         print(f"Set to resize all inputs to {self.target_resolution} before feature extraction.")

#     def _get_features_hook(self, layer_idx: int):
#         """为特定层创建前向钩子"""
#         def hook(model, input, output):
#             self.features[layer_idx] = output[0] if isinstance(output, tuple) else output
#         return hook
#     # In PerceptualLoss class

#     def forward(self, pred_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
#         total_loss = 0.0
#         hook_handles = []

#         # ... (resize 和 repeat 的部分保持不变) ...
#         pred_images_resized = F.interpolate(pred_images, size=self.target_resolution, mode='bilinear', align_corners=False)
#         target_images_resized = F.interpolate(target_images, size=self.target_resolution, mode='bilinear', align_corners=False)
#         if pred_images_resized.dim() == 4:
#             pred_images_resized = pred_images_resized.unsqueeze(2).repeat(1, 1, 2, 1, 1)
#             target_images_resized = target_images_resized.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        
#         B, C, T, H, W = pred_images_resized.shape
#         grid_t = T // 2
#         grid_h = H // 14
#         grid_w = W // 14
#         grid_thw = torch.tensor([grid_t, grid_h, grid_w], device=pred_images_resized.device)
#         grid_thw = grid_thw.unsqueeze(0).repeat(B, 1)

#         # 注册钩子
#         for layer_idx in self.target_layers:
#             block = self.vision_encoder.blocks[layer_idx]
#             handle = block.register_forward_hook(self._get_features_hook(layer_idx))
#             hook_handles.append(handle)

#         # ========================= START: CRITICAL GRADIENT FIX =========================

#         # 步骤 5.1: 提取 PRED 特征 (必须在梯度计算图中！)
#         self.features.clear()
#         # PREDICION path - DO NOT use torch.no_grad() here!
#         self.vision_encoder(pred_images_resized, grid_thw=grid_thw)
#         pred_features = dict(self.features)

#         # 步骤 5.2: 提取 TARGET 特征 (这里才用 no_grad，因为 target 不需要梯度)
#         self.features.clear()
#         with torch.no_grad():
#             self.vision_encoder(target_images_resized, grid_thw=grid_thw)
#         target_features = dict(self.features)

#         # ========================== END: CRITICAL GRADIENT FIX ==========================

#         # 步骤 6 & 7: 清理并计算损失
#         for handle in hook_handles:
#             handle.remove()
            
#         for layer_idx in self.target_layers:
#             # 现在，pred_features[layer_idx] 包含了梯度历史
#             total_loss += self.loss_fn(pred_features[layer_idx], target_features[layer_idx])
                
#         return total_loss / len(self.target_layers)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from univa.utils.resnet import ResNet # 1. 从您保存的文件中导入 ResNet 类

class PerceptualLossResNet(nn.Module):
    """
    使用您提供的、纯视觉的 ResNet 模型来计算感知损失。
    """
    def __init__(self, weights_path, device):
        super().__init__()
        self.device = device
        
        # 2. 初始化 ResNet 模型
        #    这里的参数需要根据 ODM 的实现来确定，但通常是固定的
        #    train_backbone=False, return_interm_layers=True, dilation=False, freeze_bn=True
        self.encoder = ResNet(
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
            freeze_bn=True
        ).to(device)
        
        # 3. 加载您提供的预训练权重
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # 根据 ODM 推理代码的习惯，权重可能嵌套在 'state_dict' 中，并带有 'module.' 前缀
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict_cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.encoder.load_state_dict(state_dict_cleaned, strict=False) # strict=False 更稳健
            print("Successfully loaded pre-trained ResNet weights for Perceptual Loss.")
        except Exception as e:
            print(f"Failed to load ResNet weights: {e}")
            raise

        # 4. 冻结所有权重
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("ResNet 'teacher' model has been frozen and set to evaluation mode.")

        self.loss_fn = nn.L1Loss() # L1 Loss 通常比 MSE 在感知上效果更好

        # 5. 准备图像预处理
        #    感知损失模型通常需要特定的归一化
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, pred_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        # 输入的 pred_images 和 target_images 都是 VAE 解码后的 [-1, 1] 范围
        
        # 1. 将图像从 [-1, 1] 转换到 [0, 1] 范围，以进行标准化
        pred_images_0_1 = pred_images / 2 + 0.5
        target_images_0_1 = target_images / 2 + 0.5
        
        # 2. 应用 ImageNet 标准化
        pred_images_norm = self.normalize(pred_images_0_1)
        target_images_norm = self.normalize(target_images_0_1)
        
        # 3. 提取 PRED 特征 (梯度必须流动)
        #    ResNet 的 forward 直接返回了多层特征的字典
        pred_features_dict = self.encoder.body(pred_images_norm)

        # 4. 提取 TARGET 特征 (不需要梯度)
        with torch.no_grad():
            target_features_dict = self.encoder.body(target_images_norm)

        # 5. 计算多层损失
        total_loss = 0.0
        # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        for layer_key in pred_features_dict.keys():
            pred_feat = pred_features_dict[layer_key]
            target_feat = target_features_dict[layer_key]
            total_loss += self.loss_fn(pred_feat, target_feat)
            
        return total_loss / len(pred_features_dict.keys())



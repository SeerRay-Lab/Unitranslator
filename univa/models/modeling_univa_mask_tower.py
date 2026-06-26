import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from collections import OrderedDict

# --- ✨ 1. 将你提供的所有MGM相关类，全部粘贴到这里 ✨ ---
# (QuickGELU, ResidualAttentionBlockDecoder, TransformerDecoder, MGM)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlockDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        # 注意：这里的attn_mask是固定的，对于Causal LM可能需要动态创建
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)

    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x
        
        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]

class TransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)

class MGM(nn.Module):
    # (你提供的完整MGM代码，包括 __init__, _initialize_weights, att_image_to_text, forward)
    # ...
    # 为了完整性，我将它粘贴在这里
    def __init__(self, input_size, hidden_size, dev_convs_nums, out_channels, layer_num=2, head_num=8):
        super(MGM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.dev_convs_nums = dev_convs_nums
        self.proj_q = nn.Linear(input_size, hidden_size)
        self.proj_kv = nn.Linear(input_size, hidden_size)
        self.ln_q= nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(hidden_size)
        self.ln_final_decoder = nn.LayerNorm(hidden_size)
        self.cross_attn = TransformerDecoder(width=hidden_size, layers=layer_num, heads=head_num)
        self.dev_convs = nn.ModuleList()
        cur_chn = hidden_size
        for _ in range(dev_convs_nums):
            self.dev_convs.append(nn.Sequential(
                nn.ConvTranspose2d(cur_chn, cur_chn // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(cur_chn // 2),
                nn.GELU(),
            ))
            cur_chn = cur_chn // 2
        self.final_d = nn.Sequential(
            nn.Conv2d(cur_chn, cur_chn // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cur_chn // 2),
            nn.GELU(),
            nn.Conv2d(cur_chn // 2, out_channels, kernel_size=1),
        )
        self._initialize_weights() # 在初始化时应用权重

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                # --- ✨ 关键修复：采用健壮的初始化方法 ---
                # 1. 创建一个与权重形状相同，但类型为 float32 的临时张量。
                # 2. 在这个 float32 张量上进行 Kaiming He 初始化 (通常比 Xavier 更适合 ReLU/GELU)。
                # 3. 使用 .data.copy_() 将初始化好的值安全地复制回原始权重，
                #    .copy_() 会自动处理从 float32 到 bfloat16/float16 的类型转换。
                with torch.no_grad():
                    temp_weights = torch.empty_like(m.weight, dtype=torch.float32)
                    nn.init.kaiming_normal_(temp_weights, mode='fan_out', nonlinearity='relu')
                    m.weight.data.copy_(temp_weights)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # --- Transformer部分的初始化保持不变，它们通常是正确的 ---
        proj_std = (self.hidden_size ** -0.5) * ((2 * self.layer_num) ** -0.5)
        attn_std = self.hidden_size ** -0.5
        fc_std = (2 * self.hidden_size) ** -0.5
        nn.init.normal_(self.proj_q.weight, std=attn_std)
        nn.init.normal_(self.proj_kv.weight, std=attn_std)
        for block in self.cross_attn.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)



    def att_image_to_text(self, encoded_image, llm_image):
        encoded_image = encoded_image.permute(1, 0, 2).contiguous()
        llm_image = llm_image.permute(1, 0, 2).contiguous()
        tmp = self.cross_attn([llm_image, encoded_image, encoded_image, None])
        x = tmp[0].permute(1, 0, 2).contiguous()
        x = self.ln_final_decoder(x)
        return x, tmp[4]

    def forward(self, image_llm_hidden_features, image_encoder_attn_features, output_size, image_patch_size, image_token_num):
        image_llm_hidden_features = self.ln_q(self.proj_q(image_llm_hidden_features))
        image_encoder_attn_features = self.ln_kv(self.proj_kv(image_encoder_attn_features))
        image_embedding_features, attn_map = self.att_image_to_text(image_encoder_attn_features, image_llm_hidden_features)
        b, n, c = image_embedding_features.shape

        x = image_embedding_features.reshape(b * image_token_num, image_patch_size, image_patch_size, c).permute(0, 3, 1, 2)
        for i in range(self.dev_convs_nums):
            x = self.dev_convs[i](x)
        x = F.interpolate(x, size=(output_size, output_size), mode='nearest')
        x = self.final_d(x)
        x = torch.sigmoid(x) # ✨ 确保输出在[0,1]之间
        return x, attn_map


# --- ✨ 2. 改造 UnivaMaskTower，让它成为 MGM 的“包装器” ✨ ---
class UnivaMaskTower(nn.Module):
    """
    【MGM集成版】
    这个模块现在内部使用MGM来完成复杂的预测任务。
    它的外部接口保持不变，以便与主模型无缝对接。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_marten_layer_idx = 0
        # 从主config中，为MGM准备它需要的参数
        self.mgm_head = MGM(
            input_size=config.input_hidden_size,
            hidden_size=config.mgm_hidden_size,      # e.g., 512
            dev_convs_nums=config.mgm_dev_convs_nums, # e.g., 3
            out_channels=config.output_channels,      # 1
            layer_num=config.mgm_layer_num,         # e.g., 2
            head_num=config.mgm_head_num            # e.g., 8
        )

    def init_mgm_weights(self):
        """
        提供一个公共接口，用于在外部安全地调用 MGM 模块的特殊权重初始化。
        """
        print("--- Manually initializing weights for the MGM head... ---")
        self.mgm_head._initialize_weights()

    def forward(
        self,
        mllm_intermediate_features: torch.Tensor,       # <-- 新的输入
        teacher_forced_text_features: torch.Tensor,   # <-- 新的输入
        image_positions: List[int],
        image_token_lengths: List[int],
        target_height: int,
        target_width: int,
        selected: torch.Tensor,
        pixel_values=None,
        start_text_idx=None,
    ) -> torch.Tensor:
        
        # ✨ --- 核心逻辑重构 --- ✨
        
        # 1. 准备 MGM 的 Query (Q): 从 MLLM 中间层提取具有空间多样性的图像特征
        #    mllm_intermediate_features 的形状是 [B, N, C]
        B, N, C = mllm_intermediate_features.shape
        
        # 使用 'selected' 掩码从原始中间层特征中安全地提取图像部分
        # selected 的形状是 [B, N], bool 类型
        # 确保数据类型正确以进行后续操作
        image_features_for_mgm = mllm_intermediate_features[selected].reshape(B, -1, C).to(mllm_intermediate_features.dtype)

        # 2. 准备 MGM 的 Key/Value (K,V): 直接使用最精确的 GT 文本特征
        text_features_for_mgm = teacher_forced_text_features.to(mllm_intermediate_features.dtype)

        # 3. 获取图像 patch 数量，用于 MGM 内部的 reshape
        num_image_token = image_features_for_mgm.shape[1]
        
        # 确保 patch 数量是平方数，否则 reshape 会失败
        if int(num_image_token ** 0.5) ** 2 != num_image_token:
             raise ValueError(f"Number of image tokens ({num_image_token}) must be a perfect square for MGM reshape.")

        # 4. 调用 MGM head
        predicted_mask, _ = self.mgm_head(
            image_llm_hidden_features=image_features_for_mgm,      # 这是 Q (空间)
            image_encoder_attn_features=text_features_for_mgm,   # 这是 K,V (文本引导)
            output_size=target_height,
            image_patch_size=int(num_image_token ** 0.5),
            image_token_num=1 # 假设每批次一个图片
        )

        return predicted_mask


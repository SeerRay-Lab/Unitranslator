from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import numpy as np
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput

# --- 新增：交叉注意力融合模块 ---
class CrossAttentionFusionLayer(nn.Module):
    """
    一个专门用于融合两个不同来源的 hidden_states 的模块。
    它使用一个输入作为 Query，另一个作为 Key/Value，并通过一个标准 Transformer Block 进行处理。
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_dim_mult),
            nn.GELU(),
            nn.Linear(embed_dim * ff_dim_mult, embed_dim),
        )

    # def forward(self, query_states: torch.Tensor, kv_states: torch.Tensor) -> torch.Tensor:
    #     # query_states 来自一个源, kv_states 来自另一个源
    #     # 1. 交叉注意力
    #     attn_output, _ = self.attn(query=query_states, key=kv_states, value=kv_states)
        
    #     # 2. 残差连接和归一化
    #     x = self.norm1(query_states + attn_output)
        
    #     # 3. 前馈网络
    #     ffn_output = self.ffn(x)
        
    #     # 4. 第二个残差连接和归一化
    #     output = self.norm2(x + ffn_output)
        
    #     return output

    # def forward(self, query_states, kv_states, query_mask=None, kv_mask=None):
    #     # query_states: [B, Lq, D], kv_states: [B, Lk, D]
    #     B, Lq, D = query_states.shape
    #     _, Lk, _ = kv_states.shape

    #     # 0) cache kv_states（按需 detach，避免占用计算图）
    #     self.cached_kv_states = kv_states.detach()

    #     # helper: build kv residual aligned to [B, Lq, D]
    #     if Lk == Lq:
    #         kv_residual = kv_states
    #     else:
    #         if kv_mask is not None:
    #             # kv_mask: [B, Lk] bool, True=valid
    #             denom = kv_mask.sum(dim=1, keepdim=True).clamp(min=1).to(kv_states.dtype)  # [B,1]
    #             kv_sum = (kv_states * kv_mask.unsqueeze(-1).to(kv_states.dtype)).sum(dim=1)  # [B,D]
    #             kv_pooled = kv_sum / denom  # [B,D]
    #         else:
    #             kv_pooled = kv_states.mean(dim=1)  # [B,D]
    #         kv_residual = kv_pooled.unsqueeze(1).expand(B, Lq, D)  # [B,Lq,D]

    #     if query_mask is None:
    #         attn_out, _ = self.attn(
    #             query_states, kv_states, kv_states,
    #             key_padding_mask=(~kv_mask) if kv_mask is not None else None
    #         )
    #         x = self.norm1(query_states + attn_out)
    #         out = self.norm2(x + self.ffn(x))

    #         # 1) add kv residual
    #         out = out + kv_residual
    #         return out

    #     # 1) 每个样本有效长度
    #     lengths = query_mask.sum(dim=1).tolist()
    #     max_len = max(lengths) if max(lengths) > 0 else 1

    #     # 2) gather 有效 query 到 padded batch
    #     packed = query_states.new_zeros((B, max_len, D))
    #     packed_mask = query_states.new_zeros((B, max_len), dtype=torch.bool)
    #     for b in range(B):
    #         idx = torch.where(query_mask[b])[0]
    #         n = idx.numel()
    #         if n > 0:
    #             packed[b, :n] = query_states[b, idx]
    #             packed_mask[b, :n] = True

    #     # 3) cross-attn
    #     attn_out, _ = self.attn(
    #         query=packed,
    #         key=kv_states,
    #         value=kv_states,
    #         key_padding_mask=(~kv_mask) if kv_mask is not None else None
    #     )

    #     x = self.norm1(packed + attn_out)
    #     out_packed = self.norm2(x + self.ffn(x))

    #     # 4) scatter 回原长，其他位置保持 0
    #     out = query_states.new_zeros((B, Lq, D))
    #     for b in range(B):
    #         idx = torch.where(query_mask[b])[0]
    #         n = idx.numel()
    #         if n > 0:
    #             out[b, idx] = out_packed[b, :n]
    #             # 2) add kv residual only on valid query positions
    #             out[b, idx] = out[b, idx] + kv_residual[b, idx]

    #     return out

    # BEFORE 2026-01-16
    def forward(self, query_states, kv_states, query_mask=None, kv_mask=None):
        # query_states: [B, Lq, D], kv_states: [B, Lk, D]
        B, Lq, D = query_states.shape

        if query_mask is None:
            attn_out, _ = self.attn(query_states, kv_states, kv_states,
                                    key_padding_mask=(~kv_mask) if kv_mask is not None else None)
            x = self.norm1(query_states + attn_out)
            out = self.norm2(x + self.ffn(x))
            return out

        # 1) 每个样本有效长度
        lengths = query_mask.sum(dim=1).tolist()
        max_len = max(lengths) if max(lengths) > 0 else 1

        # 2) gather 有效 query 到 padded batch
        packed = query_states.new_zeros((B, max_len, D))
        packed_mask = query_states.new_zeros((B, max_len), dtype=torch.bool)
        for b in range(B):
            idx = torch.where(query_mask[b])[0]
            n = idx.numel()
            if n > 0:
                packed[b, :n] = query_states[b, idx]
                packed_mask[b, :n] = True

        # 3) cross-attn（kv 的 padding mask 也可以传）
        attn_out, _ = self.attn(
            query=packed,
            key=kv_states,
            value=kv_states,
            key_padding_mask=(~kv_mask) if kv_mask is not None else None
        )

        x = self.norm1(packed + attn_out)
        out_packed = self.norm2(x + self.ffn(x))

        # 4) scatter 回原长，其他位置保持 0
        out = query_states.new_zeros((B, Lq, D))
        for b in range(B):
            idx = torch.where(query_mask[b])[0]
            n = idx.numel()
            if n > 0:
                out[b, idx] = out_packed[b, :n]
        return out

class UnivaDenoiseTower(PreTrainedModel):
    config_class = UnivaDenoiseTowerConfig
    base_model_prefix = "model"

    def __init__(self, config: UnivaDenoiseTowerConfig):
        super().__init__(config)
        self.config = config

        if config.denoiser_type == "flux":
            self.denoiser = FluxTransformer2DModel.from_config(config.denoiser_config)
        elif config.denoiser_type == "sd3":
            self.denoiser = SD3Transformer2DModel.from_config(config.denoiser_config)
        else:
            raise ValueError(f"Unknown denoiser type: {config.denoiser_type}")
            
        # --- 修改：初始化交叉注意力融合层 ---
        # 假设两个输入源的维度与 denoiser 的输入维度一致
        # 你需要根据实际情况调整 embed_dim 和 num_heads
        self.fusion_layer = CrossAttentionFusionLayer(
            embed_dim=config.input_hidden_size,
            num_heads=config.fusion_layer_num_heads
        )

        if hasattr(config, 'denoise_projector_type') and config.denoise_projector_type:
            self._init_denoise_projector()

    def _init_denoise_projector(self):
        """Initialize the denoise_projector for QwenVL -> FLUX dimension mapping."""
        if self.config.denoise_projector_type == "mlp2x_gelu":
            self.denoise_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_hidden_size,
                    self.config.output_hidden_size * 3,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.output_hidden_size * 3, self.config.output_hidden_size
                ),
            )
        else:
            raise ValueError(
                f"Unknown denoise_projector_type: {self.config.denoise_projector_type}"
            )

    def forward(
        self,
        # ✨ 最终版、最明确的参数名
        text_query: torch.Tensor,
        context_kv: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Performs cross-attention where a pure text signal (query)
        attends to a rich multimodal context (key/value).
        """
        query_mask = kwargs.pop("query_mask", None)
        kv_mask = kwargs.pop("kv_mask", None)
        
        # 1. 使用融合层进行 Text-Query-to-Context-KV 交叉注意力
        fused_features = self.fusion_layer(
            query_states=text_query, 
            kv_states=context_kv,
            query_mask=query_mask,
            kv_mask=kv_mask,
        )

        fused_features = self.denoise_projector(fused_features)
        
        # 2. 将融合后的特征作为 denoiser 的最终上下文
        encoder_hidden_states = fused_features
        
        # ... denoiser 的其他参数和后续逻辑保持不变 ...
        hidden_states = kwargs.pop("hidden_states")
        timestep = kwargs.pop("timestep")
        pooled_projections = kwargs.pop("pooled_projections")
        
        if self.config.denoiser_type == "flux":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            
            if encoder_hidden_states is not None:
                if prefix_prompt_embeds is not None:
                    encoder_hidden_states = torch.concat(
                        [encoder_hidden_states, prefix_prompt_embeds], dim=1
                    )
            else:
                assert prefix_prompt_embeds is not None
                encoder_hidden_states = prefix_prompt_embeds
            
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                hidden_states.device, dtype=hidden_states.dtype
            )
            enc_attention_mask = kwargs.pop('enc_attention_mask', None)
            return self.denoiser(
                hidden_states=hidden_states,  # 纯视觉特征
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states, # 使用融合后的特征
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                **kwargs,
            )[0]
            
        elif self.config.denoiser_type == "sd3":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            if prefix_prompt_embeds is not None:
                encoder_hidden_states = torch.concat(
                    [prefix_prompt_embeds, encoder_hidden_states], dim=1
                )
            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states, # 使用融合后的特征
                pooled_projections=pooled_projections,
                **kwargs,
            )[0]


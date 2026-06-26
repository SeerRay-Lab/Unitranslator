from transformers.configuration_utils import PretrainedConfig
from typing import Literal, Optional, Union
import json

class UnivaDenoiseTowerConfig(PretrainedConfig):
    model_type = "univa_denoise_tower"

    def __init__(
        self,
        denoiser_type: Literal["flux", "sd3"] = "flux",
        denoise_projector_type: str = "mlp2x_gelu",
        input_hidden_size: int = 1152,  # QwenVL hidden size
        output_hidden_size: int = 4096, # FLUX/SD3 hidden size
        denoiser_config: Optional[Union[str, dict]] = None,
        
        # --- 新增: 交叉注意力融合层配置 ---
        fusion_layer_num_heads: int = 16, # 融合层中的注意力头数
        fusion_layer_ff_dim_mult: int = 4,  # 融合层中前馈网络的维度乘数
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attn_implementation_autoset = True
        self.denoiser_type = denoiser_type
        self.denoise_projector_type = denoise_projector_type
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        
        # --- 新增: 将新参数保存到config对象中 ---
        self.fusion_layer_num_heads = fusion_layer_num_heads
        self.fusion_layer_ff_dim_mult = fusion_layer_ff_dim_mult
        
        # 处理 denoiser_config 的逻辑保持不变
        if isinstance(denoiser_config, str):
            with open(denoiser_config, "r") as f:
                self.denoiser_config = json.load(f)
        else:
            self.denoiser_config = denoiser_config

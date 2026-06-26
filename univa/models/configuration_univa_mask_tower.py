from transformers.configuration_utils import PretrainedConfig

class UnivaMaskTowerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`UnivaMaskTower`].
    """
    model_type = "univa_mask_tower"

    def __init__(
        self,
        # 定义 Mask Head 的输入维度，应与Qwen-VL的hidden_size匹配
        input_hidden_size: int = 2560, 
        # 定义 Mask Head 中间层的通道数
        intermediate_channels: tuple = (1280, 640, 320),
        # 定义最终输出通道数，对于二值mask，通常是1
        output_channels: int = 1,
        # 定义上采样因子，如果输入是patchified，需要知道如何恢复到目标尺寸
        upsample_scale: int = 16,
        mask_tower_config=None,
        **kwargs,
    ):
        self.input_hidden_size = input_hidden_size
        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        if mask_tower_config is None:
            self.mgm_hidden_size = 768
            self.mgm_dev_convs_nums = 4
            self.output_channels = 1
            self.mgm_layer_num = 2
            self.mgm_head_num = 8
        super().__init__(**kwargs)

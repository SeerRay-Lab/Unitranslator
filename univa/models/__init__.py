from .modeling_univa import UnivaQwen2ForCausalLM
from .qwen2vl.modeling_univa_qwen2vl import UnivaQwen2VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_wmask import MaskUnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_transcond import CondUnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_tf_v2 import TFUnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_cl import CLUnivaQwen2p5VLForConditionalGeneration
from .qwen2p5vl.modeling_univa_qwen2p5vl_tf_v2_mask import MaskTFUnivaQwen2p5VLForConditionalGeneration

MODEL_TYPE = {
    'llava': UnivaQwen2ForCausalLM, 
    'qwen2vl': UnivaQwen2VLForConditionalGeneration, 
    'qwen2p5vl': UnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_trans': UnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_mask': MaskUnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_trans_cond': CondUnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_tf': TFUnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_cl': CLUnivaQwen2p5VLForConditionalGeneration,
    'qwen2p5vl_tf_mask': MaskTFUnivaQwen2p5VLForConditionalGeneration,
    'lmdb_qwen': TFUnivaQwen2p5VLForConditionalGeneration,
}
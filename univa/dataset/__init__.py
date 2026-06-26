from .llava_dataset import LlavaDataset
from .qwen2vl_dataset import Qwen2VLDataset
from .qwen2vl_dataset_translation import Qwen2VLDataset_trans
from .qwen2vl_dataset_translation_mask import Qwen2VLDataset_mask
from .lmdb_qwen_dataset import LMDB_Qwen_Dataset

DATASET_TYPE = {
    'llava': LlavaDataset, 
    'qwen2vl': Qwen2VLDataset, 
    'qwen2p5vl': Qwen2VLDataset, 
    'qwen2p5vl_trans': Qwen2VLDataset_trans,
    'qwen2p5vl_mask': Qwen2VLDataset_mask,
    'qwen2p5vl_trans_cond': Qwen2VLDataset_trans,
    'qwen2p5vl_tf': Qwen2VLDataset_trans,
    'qwen2p5vl_cl': Qwen2VLDataset_trans,
    'qwen2p5vl_tf_mask': Qwen2VLDataset_mask,
    'lmdb_qwen': LMDB_Qwen_Dataset,
}
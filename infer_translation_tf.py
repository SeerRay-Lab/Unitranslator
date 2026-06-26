from infer_translation_v2 import BASE_MODEL_PATH
import torch
from peft import PeftModel
from transformers import AutoProcessor
# 确保导入你最终版本的模型类
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl_tf_v2 import TFUnivaQwen2p5VLForConditionalGeneration
import os
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from PIL import Image
from qwen_vl_utils import process_vision_info
from univa.utils.flux_pipeline import FluxKontextPipeline
from typing import List
import numpy as np
import uuid
import glob
from safetensors.torch import load_file

import random
# --- 强制确定性的“魔法咒语” ---
torch.manual_seed(42) # 为 CPU 设置随机种子
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42) # 为所有 GPU 设置随机种子
    
# 关键！
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- 1. 配置区域 (保持不变) ---


# ----- 7B 模型 -----
# BASE_MODEL_PATH = "UniWorld_Kontext"
# FINETUNE_MODEL_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_7b_onlyde2en_tf_v2_de_en_test/checkpoint-10000/univa/"
# OUTPUT_DIR = "results/outputs/"
# LORA_ADAPTER_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_7b_onlyde2en_tf_v2_de_en_test/checkpoint-10000/lora"
# FLUX_PATH = "/mnt/vlm-ks3/ljh/hf-model/FLUX.1-Kontext-dev"

# ----- 3B 模型 -----
# BASE_MODEL_PATH = "UniWorld_Kontext_3b"
# FINETUNE_MODEL_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_3b_onlyde2en_all/checkpoint-20000/univa"
# OUTPUT_DIR = "results/outputs/"
# LORA_ADAPTER_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_3b_onlyde2en_all/checkpoint-20000/lora"

BASE_MODEL_PATH = "UniWorld_Kontext_3b_TF"
FINETUNE_MODEL_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_3b_tf_finetune_mask_all/checkpoint-10000/univa"
OUTPUT_DIR = "results/outputs/"
LORA_ADAPTER_PATH = "./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_3b_tf_finetune_mask_all/checkpoint-10000/lora"

FLUX_PATH = "/mnt/vlm-ks3/ljh/hf-model/FLUX.1-Kontext-dev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

def prepare_condition_images(image_paths: List[str], device):
    if not image_paths:
        return None
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        imgs.append(t)
    return torch.stack(imgs).to(device=device, dtype=torch.float32)


def load_flux_weights_from_checkpoint(model, ckpt_path):
    all_files = glob.glob(os.path.join(ckpt_path, "*.safetensors")) + \
                glob.glob(os.path.join(ckpt_path, "*.bin"))

    flux_sd = {}
    for f in all_files:
        sd = load_file(f, device="cpu") if f.endswith(".safetensors") else torch.load(f, map_location="cpu")
        for k, v in sd.items():
            if "denoise_tower." in k:
                flux_sd[k.replace("base_model.model.", "")] = v

    if flux_sd:
        print(f"--- Attempting to load {len(flux_sd)} keys into denoise_tower ---")
        
        # 捕获加载结果
        missing_keys, unexpected_keys = model.load_state_dict(flux_sd, strict=False)
        
        # 打印不匹配的键
        if missing_keys:
            print("\n❌ WARNING: Some weights of the model were not initialized from the checkpoint:")
            for key in missing_keys:
                # 只打印与 denoise_tower 相关的缺失键
                if 'denoise_tower' in key:
                    print(f"  - {key}")

        if unexpected_keys:
            print("\n❌ WARNING: Some weights of the checkpoint were not used:")
            for key in unexpected_keys:
                print(f"  - {key}")

        if not missing_keys and not unexpected_keys:
            print("✅ All checkpoint keys were successfully matched and loaded.")

    return model



def load_hybrid_model(base_model_path: str, lora_adapter_path: str, flux_finetune_path: str, device: str):
    """
    【混合加载】
    1. 加载基础Qwen-VL模型。
    2. 加载全量微调的FLUX (denoise_tower) 权重。
    3. 加载LoRA适配器并合并。
    """
    print(f"--- Step 1/3: Loading Base Model from: {base_model_path} ---")
    model = TFUnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
    ).to(device)
    print("✅ Base model loaded.")

    if flux_finetune_path:
        print(f"--- Step 2/3: Loading finetuned FLUX weights from: {flux_finetune_path} ---")
        model = load_flux_weights_from_checkpoint(model, flux_finetune_path)
        print("✅ Finetuned FLUX weights loaded.")
    else:
        print("--- Step 2/3: Skipping finetuned FLUX loading (no path provided). ---")

    if lora_adapter_path:
        print(f"--- Step 3/3: Loading and merging LoRA Adapter from: {lora_adapter_path} ---")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
        print("✅ LoRA adapter merged.")
    else:
        print("--- Step 3/3: Skipping LoRA loading (no path provided). ---")
        
    model.eval()
    print("\nHybrid model loading complete!")
    return model


# =========================
# 核心推理函数
# =========================

@torch.no_grad()
def run_unified_inference(
    model: TFUnivaQwen2p5VLForConditionalGeneration,
    processor: AutoProcessor,
    text_prompt: str,
    pipe: FluxKontextPipeline,
    source_image_path: str,
    seed: int = 42,
    joint_with_t5: bool = True
):
    """
    两阶段推理：
    1. LVLM 翻译
    2. 用翻译文本作为 label / query 引导 denoiser
    """

    # ---------------------
    # Step 0: 构造对话
    # ---------------------
    convo = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {
                "type": "image",
                "image": source_image_path,
                "min_pixels": 256 * 256,
                "max_pixels": 256 * 256,
                # "min_pixels": 448 * 448,
                # "max_pixels": 448 * 448,
            },
        ],
    }]
    
    chat_text = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    chat_text = "<|im_end|>\n".join(chat_text.split("<|im_end|>\n")[1:])

    image_inputs, video_inputs = process_vision_info(convo)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    torch.manual_seed(seed)

    # ---------------------
    # Step 1: 翻译文本生成
    # ---------------------
    model.eval()
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values.to(torch.bfloat16),
        image_grid_thw=inputs.image_grid_thw,
        max_new_tokens=256,
        # temperature=0.1,
        # top_k=10,
        # top_p=0.9,
        do_sample=False,
        # repetition_penalty=1.2,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    translation_text = processor.tokenizer.decode(
        new_token_ids, skip_special_tokens=True
    ).strip().replace('"', '')

    print(f"[INFO] Translation: {translation_text}")
    # translation_text = "und natürlich teilen wir alle dieselben anpassungsnotwer."
    # ---------------------
    # Step 2: 用翻译文本构造 labels
    # ---------------------
    print("[INFO] Re-constructing inputs and labels for teacher-forcing pass...")

    # 2a. 获取原始 prompt 的 token
    original_prompt_ids = inputs.input_ids

    # 2b. 获取翻译文本的 token
    translation_ids = processor.tokenizer(
        translation_text, return_tensors="pt", add_special_tokens=False # <-- 视情况决定是否加特殊token
    ).input_ids.to(DEVICE)
    
    # 2c. 拼接成新的 input_ids，模拟 "prompt + answer" 的完整序列
    teacher_forcing_input_ids = torch.cat([original_prompt_ids, translation_ids], dim=1)
    teacher_forcing_attention_mask = (teacher_forcing_input_ids != processor.tokenizer.pad_token_id).long()

    # 2d. 创建 labels，将 prompt 部分 mask 掉
    prompt_length = original_prompt_ids.shape[1]
    labels = teacher_forcing_input_ids.clone()
    labels[:, :prompt_length] = -100 # 关键步骤：mask掉 prompt

    print(f"[DEBUG] New input shape for Step 3: {teacher_forcing_input_ids.shape}")
    print(f"[DEBUG] New labels shape for Step 3: {labels.shape}")


    # ---------------------
    # Step 3: 再 forward 一次，引导 denoiser
    # ---------------------
    model.eval() 

    # denoise_out = model(
    #     input_ids=inputs.input_ids,
    #     attention_mask=inputs.attention_mask,
        
    #     pixel_values=inputs.pixel_values.to(DTYPE),
    #     image_grid_thw=inputs.image_grid_thw,
    #     labels=labels,
        
    #     output_type="denoise_embeds",
    #     use_teacher_forcing=True, # 确保 use_teacher_forcing 标志被传递
    # )

    denoise_out = model(
        input_ids=teacher_forcing_input_ids,
        attention_mask=teacher_forcing_attention_mask,
        pixel_values=inputs.pixel_values.to(DTYPE),
        image_grid_thw=inputs.image_grid_thw,
        labels=labels,
        output_type="denoise_embeds",
        use_teacher_forcing=True,
    )
    detailed_prompt_embeds = denoise_out["model_pred"] # 假设 key 是 'model_pred'，请根据你的模型输出确认

    # --- ✨ Step 4 (新增): 使用FLUX的文本编码器，为翻译文本生成【全局】嵌入 ---
    print("[INFO] Generating pooled embeddings for the translation text...")
    tokenizers  = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    # 我们只需要 pooled_prompt_embeds，所以忽略第一个返回值 (用 _ 接收)
    prm_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        # f"{text_prompt.replace('Translate', 'Replace')}: '{translation_text}'", # 使用生成的翻译文本
        f'''Replace all text in the image with "{translation_text}".''',
        # translation_text,
        # "",
        # '<image>\n' + translation_text,
        256,         # 与你的训练配置保持一致
        DEVICE,
        1
    )

    if joint_with_t5:
        prompt_embeds = torch.cat([detailed_prompt_embeds, prm_embeds], dim=1)  # (1, 256+273, 4096)
    else:
        prompt_embeds = detailed_prompt_embeds
    # --- Step 5: 调用 FLUX Pipeline (现在传入两种嵌入) ---
    print("[INFO] Calling FLUX pipeline to generate the final image...")
    condition_images = prepare_condition_images([source_image_path], DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    images = pipe(
        image=condition_images,
        prompt_embeds=prompt_embeds,      
        pooled_prompt_embeds=pooled_prompt_embeds, 
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=5.0,
        generator=generator,
    ).images


    save_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.png")
    images[0].save(save_path)

    print(f"[INFO] Image saved to {save_path}")

    return translation_text, save_path


# --- 5. 主执行逻辑 (使用新的推理函数) ---
if __name__ == "__main__":
    try:
        # a. 加载模型和Processor
        model = load_hybrid_model(
            base_model_path=BASE_MODEL_PATH,
            lora_adapter_path=LORA_ADAPTER_PATH,
            flux_finetune_path=FINETUNE_MODEL_PATH,
            device=DEVICE
        )
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)

        # b. 加载FLUX Pipeline
        print(f"--- Loading FLUX Pipeline from: {FLUX_PATH} ---")
        pipe = FluxKontextPipeline.from_pretrained(
            FLUX_PATH,
            transformer=model.denoise_tower.denoiser,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        print("✅ FLUX Pipeline loaded.")

        # c. 定义任务
        source_language = "Romanian"
        target_language = "English"
        german_image_path = "/mnt/vlm-ks3/ljh/data/translationV/iwslt17.ro-en-images/test_ro/1160.jpg"
        initial_prompt = f"Translate all {source_language} texts into {target_language}."
        
        # d. ✨ 调用统一的、逻辑正确的推理函数 ✨
        english_translation, final_image_path = run_unified_inference(
            model=model,
            processor=processor,
            pipe=pipe,
            source_image_path=german_image_path,
            text_prompt=initial_prompt,
        )
        
        # e. 打印结果
        print("\n" + "="*50)
        print("🚀 UNIFIED INFERENCE COMPLETE 🚀")
        print("="*50)
        print(f"SOURCE IMAGE:        '{german_image_path}'")
        print(f"GENERATED TRANSLATION: '{english_translation}'")
        print(f"FINAL EDITED IMAGE:  '{final_image_path}'")
        print("="*50)

    except Exception as e:
        print(f"\n❌ An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()

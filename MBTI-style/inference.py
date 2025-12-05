import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download



def inference(
    model_path: str,
    neutral_text: str,
    mbti_type: str,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """使用訓練好的 LoRA 進行推理"""
    from peft import PeftModel
    
    # 載入 base model
    model, tokenizer = load_model_and_tokenizer(ModelConfig(model_name=base_model))
    
    # 載入 LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    # 構建 prompt
    prompt = f"""<|im_start|>system
    You are a text style transfer assistant. Transform neutral text into text that reflects a specific MBTI personality type's writing style.<|im_end|>
    <|im_start|>user
    Transform the following neutral text to sound like an {mbti_type} personality:

    {neutral_text}<|im_end|>
    <|im_start|>assistant
    """
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 assistant 回應
    response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

CONFIG = {
    # 改用 7B，目前速度與效果的 CP 值之王，適合快速迭代實驗
    "model_name": "Qwen/Qwen2.5-7B-Instruct", 
    
    # 序列長度
    "max_seq_length": 2048,
    
    # 必開 bfloat16
    "compute_dtype": "bfloat16",
    
    # 強烈建議安裝 flash-attn (pip install flash-attn)
    "attn_implementation": "flash_attention_2", 
    # 如果沒裝 flash-attn，請改回 "sdpa"
    # "attn_implementation": "sdpa",
    
    # LoRA Rank 維持 64，7B 模型容量較小，高 Rank 有助於記住風格細節
    "lora_r": 64,
    "lora_alpha": 128,
    
    "epochs": 3,
    
    # 關鍵調整：7B 模型很小，我們可以把單卡 Batch Size 拉大
    # 這能大幅提升訓練速度 (Throughput)
    "batch_size": 16,  # 單卡一次處理 32 筆 (5090/A100 絕對吃得下)
    "grad_accum": 2,   # 16 * 2 = 32 (Effective Batch Size)
    
    "fp16": False,
    "bf16": True,
    
    "max_samples": None,
    
    "logging_steps": 10,
    
    # 因為訓練速度極快，step 數會變少，建議調整儲存頻率
    "save_steps": 5000,
    "eval_steps": 5000,
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MBTI Style Transfer QLoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # 推理（使用 4 種類型時）
    python train.py --mode inference --adapter_path ./mbti_lora_output/final --text "Hello world" --mbti ET
        """
    )

    # 推理參數
    parser.add_argument("--model", type=str, default=None,
                        help="模型名稱（覆蓋預設）")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路徑（推理用）")
    parser.add_argument("--text", type=str, default=None,
                        help="輸入文本（推理用）")
    parser.add_argument("--mbti", type=str, default="INTJ",
                        help="目標 MBTI 類型（推理用）")
    
    args = parser.parse_args()
    
    if args.adapter_path is None or args.text is None:
        print("Error: --adapter_path and --text are required for inference mode")
        exit(1)
    
    # 選擇基礎配置（用於確定 base model）
    base_config = CONFIG["model_name"]
    base_model = args.model or base_config["model_name"]
    
    result = inference(
        model_path=args.adapter_path,
        neutral_text=args.text,
        mbti_type=args.mbti,
        base_model=base_model,
    )
    print(f"\n[{args.mbti} Style Output]:\n{result}")

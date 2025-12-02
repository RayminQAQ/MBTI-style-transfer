"""
MBTI Style Transfer - QLoRA Fine-tuning Script
===============================================
使用 QLoRA 訓練 MBTI 風格轉換模型
輸入: neutral text + MBTI type
輸出: styled text (帶有該 MBTI 人格特徵的文本)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import evaluate
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置區 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"  # 或 "meta-llama/Llama-3.2-3B-Instruct"
    # model_name: str = "Qwen/Qwen2.5-1.5B"  # 或 "meta-llama/Llama-3.2-3B-Instruct"
    
    max_seq_length: int = 1024
    # max_seq_length: int = 512

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"

    bnb_4bit_compute_dtype: str = "bfloat16"
    # bnb_4bit_compute_dtype: str = "float16"

    use_nested_quant: bool = True  # 二次量化，進一步省顯存
    
    attn_implementation: str = "sdpa"
    # attn_implementation: str = "eager"


@dataclass
class LoraConfigParams:
    """LoRA 配置"""
    r: int = 64                    # LoRA rank (越大效果越好，但顯存越多)
    # r: int = 16                    

    lora_alpha: int = 128          # LoRA alpha (通常設為 2*r)
    # lora_alpha: int = 32         

    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    """訓練配置"""
    output_dir: str = "./mbti_lora_output"

    num_train_epochs: int = 3
    # num_train_epochs: int = 1

    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 有效 batch size = 4 * 4 = 16
    # per_device_train_batch_size: int = 2
    # per_device_eval_batch_size: int = 2
    # gradient_accumulation_steps: int = 8 

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    logging_steps: int = 10
    # logging_steps: int = 5

    save_steps: int = 200
    eval_steps: int = 200
    # save_steps: int = 100
    # eval_steps: int = 100

    save_total_limit: int = 3

    fp16: bool = False
    bf16: bool = True              # 使用 bfloat16 (需要 Ampere+ GPU)
    # fp16: bool = True
    # bf16: bool = False             

    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    group_by_length: bool = True
    report_to: str = "tensorboard"


@dataclass
class DataConfig:
    """資料配置"""
    data_path: str = "./mbti_1_neutral.csv"
    # progress_path: str = "../data/mbti_1_progress.jsonl"
    train_split: float = 0.95
    max_samples: Optional[int] = None  # None = 使用全部資料


# ==================== 資料處理 ====================

PROMPT_TEMPLATE = """<|im_start|>system
You are a text style transfer assistant. Transform neutral text into text that reflects a specific MBTI personality type's writing style.<|im_end|>
<|im_start|>user
Transform the following neutral text to sound like an {mbti_type} personality:

{neutral_text}<|im_end|>
<|im_start|>assistant
{styled_text}<|im_end|>"""

# Llama 格式 (備用)
LLAMA_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a text style transfer assistant. Transform neutral text into text that reflects a specific MBTI personality type's writing style.<|eot_id|><|start_header_id|>user<|end_header_id|>
Transform the following neutral text to sound like an {mbti_type} personality:

{neutral_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{styled_text}<|eot_id|>"""


def load_and_prepare_data(config: DataConfig) -> tuple[Dataset, Dataset]:
    """載入並準備訓練資料"""
    print("Loading data...")
    
    # 載入進度檔（包含 post-level 的 neutral text）
    # progress = {}
    # if os.path.exists(config.progress_path):
    #     with open(config.progress_path, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             try:
    #                 record = json.loads(line.strip())
    #                 progress[record['post_id']] = record['neutral_text']
    #             except:
    #                 continue
    
    # 載入原始資料
    df = pd.read_csv(config.data_path)
    
    # 展開成 post-level 訓練資料
    training_samples = []
    
    for _, row in df.iterrows():
        person_id = row['id']
        mbti_type = row['type']
        original_posts = row['original_posts'].split('|||')
        neutral_posts = row['neutral_posts'].split('|||')
        
        for post_idx, original_post in enumerate(original_posts):
            post_id = f"{person_id}_{post_idx}"
            original_post = original_post.strip()
            
            # 過濾太短或 URL
            if len(original_post) < 30 or original_post.startswith('http'):
                continue
            
            # 獲取 neutral text
            neutral_text = neutral_posts[post_idx]
            if neutral_text is None or len(neutral_text) < 20:
                continue
            
            # 過濾 neutral 和 original 幾乎相同的情況
            if neutral_text.strip() == original_post.strip():
                continue
            
            training_samples.append({
                'mbti_type': mbti_type,
                'neutral_text': neutral_text,
                'styled_text': original_post,
            })
    
    print(f"Total training samples: {len(training_samples)}")
    
    # 限制樣本數
    if config.max_samples:
        training_samples = training_samples[:config.max_samples]
        print(f"Limited to {config.max_samples} samples")
    
    # 分割訓練/驗證集
    split_idx = int(len(training_samples) * config.train_split)
    train_samples = training_samples[:split_idx]
    eval_samples = training_samples[split_idx:]
    
    print(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")
    
    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)


def format_prompt(sample: dict, tokenizer, max_length: int) -> dict:
    """格式化成訓練 prompt"""
    # 根據 tokenizer 選擇模板
    if "qwen" in tokenizer.name_or_path.lower():
        template = PROMPT_TEMPLATE
    else:
        template = LLAMA_PROMPT_TEMPLATE
    
    full_prompt = template.format(
        mbti_type=sample['mbti_type'],
        neutral_text=sample['neutral_text'],
        styled_text=sample['styled_text'],
    )
    
    # Tokenize
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # 設置 labels (用於計算 loss)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


# ==================== 模型載入 ====================

def load_model_and_tokenizer(model_config: ModelConfig):
    """載入量化模型和 tokenizer"""
    print(f"Loading model: {model_config.model_name}")
    
    # BitsAndBytes 配置
    compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_config.use_nested_quant,
    )
    
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation=model_config.attn_implementation,
    )
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # 設置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_lora(model, lora_config: LoraConfigParams):
    """設置 LoRA adapter"""
    print("Setting up LoRA...")
    
    # 準備模型進行 k-bit 訓練
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # LoRA 配置
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 添加 LoRA adapter
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


# ==================== 訓練 ====================

def save_hyperparameters(
    model_config: ModelConfig,
    lora_config: LoraConfigParams,
    train_config: TrainConfig,
    data_config: DataConfig,
    train_result,
    output_dir: str,
):
    """保存所有 hyperparameters 和訓練資訊到 JSON 文件"""
    import datetime
    
    # 收集所有 hyperparameters
    hyperparams = {
        "training_date": datetime.datetime.now().isoformat(),
        "model_config": {
            "model_name": model_config.model_name,
            "max_seq_length": model_config.max_seq_length,
            "load_in_4bit": model_config.load_in_4bit,
            "bnb_4bit_quant_type": model_config.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": model_config.bnb_4bit_compute_dtype,
            "use_nested_quant": model_config.use_nested_quant,
            "attn_implementation": model_config.attn_implementation,
        },
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "target_modules": lora_config.target_modules,
            "bias": lora_config.bias,
            "task_type": lora_config.task_type,
        },
        "train_config": {
            "output_dir": train_config.output_dir,
            "num_train_epochs": train_config.num_train_epochs,
            "per_device_train_batch_size": train_config.per_device_train_batch_size,
            "per_device_eval_batch_size": train_config.per_device_eval_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "effective_batch_size": (
                train_config.per_device_train_batch_size
                * train_config.gradient_accumulation_steps
            ),
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "warmup_ratio": train_config.warmup_ratio,
            "lr_scheduler_type": train_config.lr_scheduler_type,
            "logging_steps": train_config.logging_steps,
            "save_steps": train_config.save_steps,
            "eval_steps": train_config.eval_steps,
            "save_total_limit": train_config.save_total_limit,
            "fp16": train_config.fp16,
            "bf16": train_config.bf16,
            "gradient_checkpointing": train_config.gradient_checkpointing,
            "optim": train_config.optim,
            "max_grad_norm": train_config.max_grad_norm,
            "group_by_length": train_config.group_by_length,
            "report_to": train_config.report_to,
        },
        "data_config": {
            "data_path": data_config.data_path,
            "train_split": data_config.train_split,
            "max_samples": data_config.max_samples,
        },
        "training_results": {
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "train_runtime": train_result.metrics.get("train_runtime", None),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", None),
            "total_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
            "num_epochs": train_result.epoch if hasattr(train_result, 'epoch') else None,
        },
    }
    
    # 保存到 JSON
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)
    
    print(f"\nHyperparameters saved to: {hyperparams_path}")
    
    # 同時保存一個簡化版的 README
    readme_path = os.path.join(output_dir, "README_hyperparameters.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# MBTI Style Transfer Model

## Training Date
{hyperparams['training_date']}

## Model Configuration
- **Base Model**: {model_config.model_name}
- **Max Sequence Length**: {model_config.max_seq_length}
- **Quantization**: 4-bit ({model_config.bnb_4bit_quant_type})
- **Attention**: {model_config.attn_implementation}

## LoRA Configuration
- **Rank (r)**: {lora_config.r}
- **Alpha**: {lora_config.lora_alpha}
- **Dropout**: {lora_config.lora_dropout}
- **Target Modules**: {', '.join(lora_config.target_modules)}

## Training Configuration
- **Epochs**: {train_config.num_train_epochs}
- **Batch Size**: {train_config.per_device_train_batch_size} × {train_config.gradient_accumulation_steps} = {hyperparams['train_config']['effective_batch_size']}
- **Learning Rate**: {train_config.learning_rate}
- **Optimizer**: {train_config.optim}
- **Mixed Precision**: {'FP16' if train_config.fp16 else 'BF16' if train_config.bf16 else 'None'}

## Data Configuration
- **Train Split**: {data_config.train_split * 100}%
- **Max Samples**: {data_config.max_samples or 'All'}

## Training Results
- **Total Steps**: {hyperparams['training_results']['total_steps']}
- **Training Loss**: {hyperparams['training_results']['train_loss']}
- **Training Time**: {f"{hyperparams['training_results']['train_runtime']:.2f}s ({hyperparams['training_results']['train_runtime']/60:.2f} minutes)" if hyperparams['training_results']['train_runtime'] else 'N/A'}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{model_config.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_config.model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{output_dir}")
```

See `hyperparameters.json` for complete configuration details.
""")
    
    print(f"README saved to: {readme_path}")


def train(
    model_config: ModelConfig,
    lora_config: LoraConfigParams,
    train_config: TrainConfig,
    data_config: DataConfig,
):
    """主訓練函數"""
    
    # 載入模型
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 設置 LoRA
    model = setup_lora(model, lora_config)
    
    # 載入資料
    train_dataset, eval_dataset = load_and_prepare_data(data_config)
    
    # 格式化資料
    print("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: format_prompt(x, tokenizer, model_config.max_seq_length),
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    eval_dataset = eval_dataset.map(
        lambda x: format_prompt(x, tokenizer, model_config.max_seq_length),
        remove_columns=eval_dataset.column_names,
        num_proc=4,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # ==================== 評估指標 ====================
    # 載入 BLEU 和 ROUGE 評估器
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        將 logits 轉換成 token ids 以便計算 BLEU/ROUGE
        Trainer 在 eval 時會調用這個函數
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        # 取 argmax 得到預測的 token ids
        return logits.argmax(dim=-1)
    
    def compute_metrics(eval_preds):
        """
        計算 BLEU 和 ROUGE 分數
        注意：這裡比較的是 teacher forcing 下的預測，不是自由生成
        """
        predictions, labels = eval_preds
        
        # 將 -100 (ignored tokens) 替換為 pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        
        # Decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 過濾空字串
        filtered_pairs = [
            (pred.strip(), label.strip()) 
            for pred, label in zip(decoded_preds, decoded_labels) 
            if pred.strip() and label.strip()
        ]
        
        if not filtered_pairs:
            return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        filtered_preds, filtered_labels = zip(*filtered_pairs)
        
        # 計算 BLEU（需要 reference 是 list of lists）
        try:
            bleu_result = bleu_metric.compute(
                predictions=list(filtered_preds),
                references=[[label] for label in filtered_labels],
            )
            bleu_score = bleu_result["bleu"]
        except Exception:
            bleu_score = 0.0
        
        # 計算 ROUGE
        try:
            rouge_result = rouge_metric.compute(
                predictions=list(filtered_preds),
                references=list(filtered_labels),
            )
            rouge1 = rouge_result["rouge1"]
            rouge2 = rouge_result["rouge2"]
            rougeL = rouge_result["rougeL"]
        except Exception:
            rouge1 = rouge2 = rougeL = 0.0
        
        return {
            "bleu": bleu_score,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
        }
    
    # 訓練參數
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_strategy="steps",
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_config.optim,
        max_grad_norm=train_config.max_grad_norm,
        group_by_length=train_config.group_by_length,
        report_to=train_config.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # 開始訓練
    print("Starting training...")
    train_result = trainer.train()
    
    # 儲存最終模型
    print("Saving model...")
    final_model_dir = os.path.join(train_config.output_dir, "final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # 儲存 hyperparameters 和訓練資訊
    save_hyperparameters(
        model_config=model_config,
        lora_config=lora_config,
        train_config=train_config,
        data_config=data_config,
        train_result=train_result,
        output_dir=final_model_dir,
    )
    
    print(f"Training complete! Model saved to {final_model_dir}")


# ==================== 推理 ====================

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


# ==================== 預設配置 ====================

# 小型訓練配置（Colab T4 16GB）
SMALL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_seq_length": 512,
    "compute_dtype": "float16",
    "attn_implementation": "eager",
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 1,
    "batch_size": 2,
    "grad_accum": 8,
    "fp16": True,
    "bf16": False,
    "max_samples": 5000,
    "logging_steps": 5,
    "save_steps": 100,
    "eval_steps": 100,
}

# 大型訓練配置（A100/H100）
LARGE_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "max_seq_length": 1024,
    "compute_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "lora_r": 64,
    "lora_alpha": 128,
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,
    "fp16": False,
    "bf16": True,
    "max_samples": None,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
}


# ==================== 主程式 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MBTI Style Transfer QLoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 小型訓練（Colab T4）
  python train.py --mode train --size small

  # 大型訓練（A100/H100）
  python train.py --mode train --size large

  # 自訂參數（覆蓋預設）
  python train.py --mode train --size small --epochs 2 --max_samples 10000

  # 推理
  python train.py --mode inference --adapter_path ./mbti_lora_output/final --text "Hello world" --mbti INFJ
        """
    )
    
    # 基本參數
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="運行模式: train 或 inference")
    parser.add_argument("--size", type=str, default="small", choices=["small", "large"],
                        help="配置大小: small (Colab T4) 或 large (A100/H100)")
    
    # 可覆蓋的訓練參數
    parser.add_argument("--model", type=str, default=None,
                        help="模型名稱（覆蓋預設）")
    parser.add_argument("--output_dir", type=str, default="./mbti_lora_output",
                        help="輸出目錄")
    parser.add_argument("--epochs", type=int, default=None,
                        help="訓練輪數（覆蓋預設）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size（覆蓋預設）")
    parser.add_argument("--lora_r", type=int, default=None,
                        help="LoRA rank（覆蓋預設）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大樣本數（覆蓋預設，-1 表示全部）")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="學習率（覆蓋預設）")
    
    # 推理參數
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路徑（推理用）")
    parser.add_argument("--text", type=str, default=None,
                        help="輸入文本（推理用）")
    parser.add_argument("--mbti", type=str, default="INTJ",
                        help="目標 MBTI 類型（推理用）")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 選擇基礎配置
        base_config = SMALL_CONFIG if args.size == "small" else LARGE_CONFIG
        
        print("=" * 60)
        print(f"Training Configuration: {args.size.upper()}")
        print("=" * 60)
        
        # 構建 ModelConfig（支援覆蓋）
        model_config = ModelConfig(
            model_name=args.model or base_config["model_name"],
            max_seq_length=base_config["max_seq_length"],
            bnb_4bit_compute_dtype=base_config["compute_dtype"],
            attn_implementation=base_config["attn_implementation"],
        )
        
        # 構建 LoraConfig（支援覆蓋）
        lora_r = args.lora_r or base_config["lora_r"]
        lora_config = LoraConfigParams(
            r=lora_r,
            lora_alpha=lora_r * 2 if args.lora_r else base_config["lora_alpha"],
        )
        
        # 構建 TrainConfig（支援覆蓋）
        train_config = TrainConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs or base_config["epochs"],
            per_device_train_batch_size=args.batch_size or base_config["batch_size"],
            per_device_eval_batch_size=args.batch_size or base_config["batch_size"],
            gradient_accumulation_steps=base_config["grad_accum"],
            learning_rate=args.learning_rate or 2e-4,
            fp16=base_config["fp16"],
            bf16=base_config["bf16"],
            logging_steps=base_config["logging_steps"],
            save_steps=base_config["save_steps"],
            eval_steps=base_config["eval_steps"],
        )
        
        # 構建 DataConfig（支援覆蓋）
        # --max_samples -1 表示使用全部資料
        max_samples = base_config["max_samples"]
        if args.max_samples is not None:
            max_samples = None if args.max_samples == -1 else args.max_samples
        data_config = DataConfig(max_samples=max_samples)
        
        # 印出關鍵配置
        print(f"  Model: {model_config.model_name}")
        print(f"  LoRA rank: {lora_config.r}")
        print(f"  Batch size: {train_config.per_device_train_batch_size} × {train_config.gradient_accumulation_steps}")
        print(f"  Epochs: {train_config.num_train_epochs}")
        print(f"  Max samples: {data_config.max_samples or 'All'}")
        print(f"  Precision: {'FP16' if train_config.fp16 else 'BF16'}")
        print("=" * 60)
        
        # 訓練
        train(model_config, lora_config, train_config, data_config)
        
    elif args.mode == "inference":
        if args.adapter_path is None or args.text is None:
            print("Error: --adapter_path and --text are required for inference mode")
            exit(1)
        
        # 選擇基礎配置（用於確定 base model）
        base_config = SMALL_CONFIG if args.size == "small" else LARGE_CONFIG
        base_model = args.model or base_config["model_name"]
        
        result = inference(
            model_path=args.adapter_path,
            neutral_text=args.text,
            mbti_type=args.mbti,
            base_model=base_model,
        )
        print(f"\n[{args.mbti} Style Output]:\n{result}")


"""
MBTI Style Transfer Evaluation Script
使用 test split 的 neutral text 進行風格轉換後，用 classifier 評估
"""

import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from peft import PeftModel
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Classifier 相關 ====================
CLASSIFIER_REPO_ID = "Owen12354/mbti_classifier"
CLASSIFIER_CACHE_DIR = "./mbti_classifier"
MAX_LEN = 128
AXES = ["IE", "NS", "FT", "PJ"]
AXIS_MAP = {
    "IE": {0: "I", 1: "E"},
    "NS": {0: "N", 1: "S"},
    "FT": {0: "F", 1: "T"},
    "PJ": {0: "P", 1: "J"}
}

# EI + TF 模式的映射
EI_TF_MAPPING = {
    "ET": ["ENTJ", "ENTP", "ESTJ", "ESTP"],
    "EF": ["ENFJ", "ENFP", "ESFJ", "ESFP"],
    "IT": ["INTJ", "INTP", "ISTJ", "ISTP"],
    "IF": ["INFJ", "INFP", "ISFJ", "ISFP"],
}

def get_ei_tf_type(mbti_type: str) -> str:
    """將 16 種 MBTI 轉換成 EI+TF 的 4 種類型"""
    ei = mbti_type[0]  # E or I
    tf = mbti_type[2]  # T or F
    return f"{ei}{tf}"


def download_classifier_models():
    """下載 classifier 模型"""
    if not os.path.exists(CLASSIFIER_CACHE_DIR):
        print("Downloading classifier from HuggingFace…")
        snapshot_download(repo_id=CLASSIFIER_REPO_ID, local_dir=CLASSIFIER_CACHE_DIR, local_dir_use_symlinks=False)
        print("Download completed!\n")


def load_axis_model(axis, device="cpu"):
    """載入某一軸的 classifier 模型"""
    model_file = f"{axis}/pytorch_model_{axis}.bin"
    model_path = os.path.join(CLASSIFIER_CACHE_DIR, model_file)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer, model


class MBTIClassifier:
    """MBTI Classifier 封裝"""
    def __init__(self, device="cpu"):
        self.device = device
        download_classifier_models()
        self.models = {}
        self.tokenizers = {}
        for axis in AXES:
            self.tokenizers[axis], self.models[axis] = load_axis_model(axis, device)
    
    def predict(self, text):
        """預測 MBTI 類型"""
        final_mbti = ""
        final_confidence = 1.0
        axis_results = {}

        for axis in AXES:
            tokenizer = self.tokenizers[axis]
            model = self.models[axis]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).flatten().tolist()

            pred_class = int(torch.argmax(torch.tensor(probs)))
            pred_letter = AXIS_MAP[axis][pred_class]
            pred_prob = probs[pred_class]

            final_mbti += pred_letter
            final_confidence *= pred_prob

            axis_results[axis] = {
                "prob_class0": probs[0],
                "prob_class1": probs[1],
                "chosen": pred_letter,
                "confidence": pred_prob
            }

        return final_mbti, final_confidence, axis_results


# ==================== Style Transfer 相關 ====================

class StyleTransferModel:
    """LoRA Style Transfer 模型封裝"""
    def __init__(self, adapter_path=None, base_model="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        self.device = device
        self.adapter_path = adapter_path
        print(f"Loading base model: {base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        
        if adapter_path:
            print(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            print("Using base model without LoRA adapter")
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def transfer(self, neutral_text, mbti_type, max_new_tokens=256):
        """將 neutral text 轉換成指定 MBTI 風格"""
        prompt = f"""<|im_start|>system
You are a text style transfer assistant. Transform neutral text into text that reflects a specific MBTI personality type's writing style.<|im_end|>
<|im_start|>user
Transform the following neutral text to sound like an {mbti_type} personality:

{neutral_text}<|im_end|>
<|im_start|>assistant
"""

        if self.adapter_path is None:
            # No LoRA 模式：使用更詳細的 prompt
            prompt = f"""<|im_start|>system
You are an expert text style transfer assistant. Your goal is to rewrite neutral text into a specific MBTI personality style, focusing strictly on the **Energy (E/I)** and **Nature (T/F)** dimensions.

### Style Guidelines
Analyze the target MBTI type's E/I and T/F letters to determine the tone:

1. **Energy: Extraversion (E) vs. Introversion (I)**
   * **If E (Extraverted):** Use an energetic, assertive, and engaging tone. Use direct address ("Hey team", "Listen up"), active verbs, and potentially exclamation marks. Focus on external action.
   * **If I (Introverted):** Use a calm, reserved, and reflective tone. Use softer sentence structures, internal reasoning ("I feel", "It seems"), and maintain a thoughtful distance.

2. **Nature: Thinking (T) vs. Feeling (F)**
   * **If T (Thinking):** Focus on logic, facts, efficiency, and competence. Be objective, concise, and blunt. Avoid emotional fluff.
   * **If F (Feeling):** Focus on values, harmony, empathy, and people. Use emotional keywords ("appreciate", "feel", "care") and a warm, supportive tone.

### One-Shot Example
**Neutral Text:** "The report contains errors. Please correct them and resubmit."

**Target MBTI:** ESTJ (Extraverted + Thinking focus)
*Analysis:* Needs to be direct/commanding (E) and focused on the error/solution without sugar-coating (T).
**Transformed Text:** "Attention! I've spotted errors in the report. This needs to be fixed immediately to ensure accuracy. Correct it and resubmit ASAP."

---
**Target MBTI:** INFP (Introverted + Feeling focus)
*Analysis:* Needs to be gentle/reflective (I) and focused on how the feedback is received/growth (F).
**Transformed Text:** "I spent some time reading the report and noticed a few things that might need looking at. Maybe we could gently revise those parts? Take your time to make it feel right."

### Task
Transform the user's neutral text to match the E/I and T/F traits of the target MBTI type. Output ONLY the transformed text, nothing else.<|im_end|>
<|im_start|>user
Target MBTI: {mbti_type}
Neutral Text: {neutral_text}<|im_end|>
<|im_start|>assistant
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 先保留 special tokens 以便正確分割
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # 提取 assistant 回應
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        # 移除結尾的 <|im_end|> 和其他 special tokens
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        
        return response


# ==================== 評估流程 ====================

def evaluate(
    adapter_path: str = None,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    target_mbti: str = None,  # 如果設定，只轉換成這個 MBTI；否則用 dataset label
    ei_tf_only: bool = False,  # 是否只用 EI+TF 的 4 類型
    max_samples: int = None,
    output_file: str = None,
    device: str = "cuda",
    min_text_length: int = 30,  # 最小文本長度
    no_lora: bool = False,  # 是否不使用 LoRA
):
    """
    評估 Style Transfer 模型
    
    Args:
        adapter_path: LoRA adapter 路徑（no_lora=True 時可為 None）
        base_model: 基礎模型
        target_mbti: 固定目標 MBTI 類型（None 則使用 dataset 中的 label）
        ei_tf_only: 是否使用 EI+TF 的 4 類型模式
        max_samples: 最大樣本數（None 則用全部）
        output_file: 輸出結果 JSON 檔案路徑
        device: 運算裝置
        min_text_length: 過濾太短的文本
        no_lora: 是否使用 base model（不載入 LoRA）
    """
    
    # 載入 dataset
    print("Loading dataset: Binga288/mbti_style_transfer")
    raw_dataset = load_dataset("Binga288/mbti_style_transfer", split="test")
    
    # 處理資料：將 ||| 分割的 posts 展開
    print("Processing posts (splitting by |||)...")
    samples = []
    type_counts = {}
    
    for row in raw_dataset:
        original_mbti = row['type']
        neutral_posts = row['neutral_posts'].split('|||')
        original_posts = row['original_posts'].split('|||')
        
        for neutral_text, original_post in zip(neutral_posts, original_posts):
            neutral_text = neutral_text.strip()
            original_post = original_post.strip()
            
            # 過濾太短或 URL
            if len(neutral_text) < min_text_length:
                continue
            if neutral_text.startswith('http'):
                continue
            # 過濾 neutral 和 original 完全相同
            if neutral_text == original_post:
                continue
            
            # 決定 label（EI+TF 或原始 16 類型）
            if ei_tf_only:
                mbti_type = get_ei_tf_type(original_mbti)
            else:
                mbti_type = original_mbti
            
            samples.append({
                'neutral_text': neutral_text,
                'original_post': original_post,
                'original_mbti': original_mbti,
                'mbti_type': mbti_type,  # 實際使用的 label
            })
            type_counts[mbti_type] = type_counts.get(mbti_type, 0) + 1
    
    print(f"Total posts after splitting: {len(samples)}")
    print(f"Type distribution: {type_counts}")
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # 載入模型
    print("\n" + "="*50)
    print("Loading models...")
    print("="*50)
    
    classifier = MBTIClassifier(device="cuda")
    effective_adapter = None if no_lora else adapter_path
    transfer_model = StyleTransferModel(effective_adapter, base_model, device)
    
    # 評估
    results = []
    correct_full = 0  # 完全正確
    total = 0
    
    # 根據模式決定要追蹤的軸
    if ei_tf_only:
        eval_axes = ["IE", "FT"]  # 只評估 EI 和 TF
        print("\n[Mode] EI+TF only (4 types: ET, EF, IT, IF)")
    else:
        eval_axes = AXES  # 評估全部 4 軸
        print("\n[Mode] Full 16 MBTI types")
    
    correct_per_axis = {axis: 0 for axis in eval_axes}
    
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        neutral_text = sample["neutral_text"]
        original_mbti = sample["original_mbti"]  # 原始 16 種
        dataset_label = sample["mbti_type"]  # 根據模式可能是 4 種或 16 種
        
        # 決定目標 MBTI
        if target_mbti:
            target = target_mbti.upper()
            # 若 ei_tf_only 模式但傳入 4 字元 MBTI，自動轉換成 2 字元
            if ei_tf_only and len(target) == 4:
                target = get_ei_tf_type(target)
        else:
            target = dataset_label.upper()
        
        try:
            # Style Transfer
            transferred_text = transfer_model.transfer(neutral_text, target)
            logger.info(f"Transferred text: {transferred_text[:200]}...")
            
            # Classifier 預測 (永遠返回 4 字母)
            predicted_mbti, confidence, axis_detail = classifier.predict(transferred_text)
            
            # 計算正確性
            if ei_tf_only:
                # EI+TF 模式：只比較 EI 和 TF
                predicted_ei_tf = get_ei_tf_type(predicted_mbti)
                is_correct = (predicted_ei_tf == target)
            else:
                # Full 模式：比較完整 4 字母
                is_correct = (predicted_mbti == target)
            
            if is_correct:
                correct_full += 1
            
            # 每軸正確性
            for axis in eval_axes:
                if axis == "IE":
                    pred_letter = predicted_mbti[0]
                    target_letter = target[0]
                elif axis == "NS":
                    pred_letter = predicted_mbti[1]
                    target_letter = target[1]
                elif axis == "FT":
                    if ei_tf_only:
                        pred_letter = predicted_mbti[2]  # T or F
                        target_letter = target[1]  # EI+TF 格式中第二個字母是 T/F
                    else:
                        pred_letter = predicted_mbti[2]
                        target_letter = target[2]
                elif axis == "PJ":
                    pred_letter = predicted_mbti[3]
                    target_letter = target[3]
                
                if pred_letter == target_letter:
                    correct_per_axis[axis] += 1
            
            total += 1
            
            result = {
                "idx": idx,
                "neutral_text": neutral_text[:200] + "..." if len(neutral_text) > 200 else neutral_text,
                "target_mbti": target,
                "original_mbti": original_mbti,
                "transferred_text": transferred_text[:200] + "..." if len(transferred_text) > 200 else transferred_text,
                "predicted_mbti": predicted_mbti,
                "predicted_ei_tf": get_ei_tf_type(predicted_mbti) if ei_tf_only else None,
                "confidence": confidence,
                "is_correct": is_correct,
                "axis_detail": axis_detail,
            }
            results.append(result)
            
            # 即時輸出進度
            if (idx + 1) % 10 == 0:
                acc = correct_full / total * 100
                print(f"\n[Progress] {idx+1}/{len(samples)} | Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"\n[Error] Sample {idx}: {e}")
            continue
    
    # 統計
    accuracy_full = correct_full / total * 100 if total > 0 else 0
    accuracy_per_axis = {axis: correct_per_axis[axis] / total * 100 if total > 0 else 0 for axis in eval_axes}
    
    summary = {
        "total_samples": total,
        "target_mbti_mode": target_mbti if target_mbti else "dataset_label",
        "ei_tf_only": ei_tf_only,
        "no_lora": no_lora,
        "accuracy_full_match": accuracy_full,
        "accuracy_per_axis": accuracy_per_axis,
        "correct_full": correct_full,
        "correct_per_axis": correct_per_axis,
    }
    
    # 輸出結果
    print("\n" + "="*50)
    print("EVALUATION RESULTS" + (" (Base Model, No LoRA)" if no_lora else " (With LoRA)"))
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Target MBTI Mode: {target_mbti if target_mbti else 'Use dataset label'}")
    print(f"EI+TF Only: {ei_tf_only}")
    print(f"No LoRA: {no_lora}")
    print(f"\nFull Match Accuracy: {accuracy_full:.2f}% ({correct_full}/{total})")
    print("\nPer-Axis Accuracy:")
    for axis in eval_axes:
        print(f"  {axis}: {accuracy_per_axis[axis]:.2f}% ({correct_per_axis[axis]}/{total})")
    print("="*50)
    
    # 儲存結果
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results_{timestamp}.json"
    
    output_data = {
        "summary": summary,
        "config": {
            "adapter_path": adapter_path,
            "base_model": base_model,
            "target_mbti": target_mbti,
            "ei_tf_only": ei_tf_only,
            "no_lora": no_lora,
            "max_samples": max_samples,
            "min_text_length": min_text_length,
        },
        "results": results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary, results


def evaluate_baseline(
    ei_tf_only: bool = False,
    max_samples: int = None,
    output_file: str = None,
    min_text_length: int = 30,
):
    """
    Baseline 評估：直接用 classifier 判斷 neutral text（無 LLM 轉換）
    用於和有 LLM 轉換的結果做比較
    
    Args:
        ei_tf_only: 是否使用 EI+TF 的 4 類型模式
        max_samples: 最大樣本數（None 則用全部）
        output_file: 輸出結果 JSON 檔案路徑
        min_text_length: 過濾太短的文本
    """
    
    # 載入 dataset
    print("Loading dataset: Binga288/mbti_style_transfer")
    raw_dataset = load_dataset("Binga288/mbti_style_transfer", split="test")
    
    # 處理資料：將 ||| 分割的 posts 展開
    print("Processing posts (splitting by |||)...")
    samples = []
    type_counts = {}
    
    for row in raw_dataset:
        original_mbti = row['type']
        neutral_posts = row['neutral_posts'].split('|||')
        original_posts = row['original_posts'].split('|||')
        
        for neutral_text, original_post in zip(neutral_posts, original_posts):
            neutral_text = neutral_text.strip()
            original_post = original_post.strip()
            
            if len(neutral_text) < min_text_length:
                continue
            if neutral_text.startswith('http'):
                continue
            if neutral_text == original_post:
                continue
            
            if ei_tf_only:
                mbti_type = get_ei_tf_type(original_mbti)
            else:
                mbti_type = original_mbti
            
            samples.append({
                'neutral_text': neutral_text,
                'original_post': original_post,
                'original_mbti': original_mbti,
                'mbti_type': mbti_type,
            })
            type_counts[mbti_type] = type_counts.get(mbti_type, 0) + 1
    
    print(f"Total posts after splitting: {len(samples)}")
    print(f"Type distribution: {type_counts}")
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # 載入 classifier
    print("\n" + "="*50)
    print("Loading classifier...")
    print("="*50)
    
    classifier = MBTIClassifier(device="cuda")
    
    # 評估
    results = []
    correct_full = 0
    total = 0
    
    if ei_tf_only:
        eval_axes = ["IE", "FT"]
        print("\n[Mode] EI+TF only (4 types: ET, EF, IT, IF)")
    else:
        eval_axes = AXES
        print("\n[Mode] Full 16 MBTI types")
    
    correct_per_axis = {axis: 0 for axis in eval_axes}
    
    print("\n" + "="*50)
    print("Starting BASELINE evaluation (no LLM transfer)...")
    print("="*50 + "\n")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating Baseline")):
        neutral_text = sample["neutral_text"]
        original_mbti = sample["original_mbti"]
        target = sample["mbti_type"].upper()
        
        try:
            # 直接用 classifier 預測 neutral text
            predicted_mbti, confidence, axis_detail = classifier.predict(neutral_text)
            
            # 計算正確性
            if ei_tf_only:
                predicted_ei_tf = get_ei_tf_type(predicted_mbti)
                is_correct = (predicted_ei_tf == target)
            else:
                is_correct = (predicted_mbti == target)
            
            if is_correct:
                correct_full += 1
            
            # 每軸正確性
            for axis in eval_axes:
                if axis == "IE":
                    pred_letter = predicted_mbti[0]
                    target_letter = target[0]
                elif axis == "NS":
                    pred_letter = predicted_mbti[1]
                    target_letter = target[1]
                elif axis == "FT":
                    if ei_tf_only:
                        pred_letter = predicted_mbti[2]
                        target_letter = target[1]
                    else:
                        pred_letter = predicted_mbti[2]
                        target_letter = target[2]
                elif axis == "PJ":
                    pred_letter = predicted_mbti[3]
                    target_letter = target[3]
                
                if pred_letter == target_letter:
                    correct_per_axis[axis] += 1
            
            total += 1
            
            result = {
                "idx": idx,
                "neutral_text": neutral_text[:200] + "..." if len(neutral_text) > 200 else neutral_text,
                "target_mbti": target,
                "original_mbti": original_mbti,
                "predicted_mbti": predicted_mbti,
                "predicted_ei_tf": get_ei_tf_type(predicted_mbti) if ei_tf_only else None,
                "confidence": confidence,
                "is_correct": is_correct,
                "axis_detail": axis_detail,
            }
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                acc = correct_full / total * 100
                print(f"\n[Progress] {idx+1}/{len(samples)} | Accuracy: {acc:.2f}%")
            
        except Exception as e:
            print(f"\n[Error] Sample {idx}: {e}")
            continue
    
    # 統計
    accuracy_full = correct_full / total * 100 if total > 0 else 0
    accuracy_per_axis = {axis: correct_per_axis[axis] / total * 100 if total > 0 else 0 for axis in eval_axes}
    
    summary = {
        "total_samples": total,
        "mode": "baseline (no LLM transfer)",
        "ei_tf_only": ei_tf_only,
        "accuracy_full_match": accuracy_full,
        "accuracy_per_axis": accuracy_per_axis,
        "correct_full": correct_full,
        "correct_per_axis": correct_per_axis,
    }
    
    # 輸出結果
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS (No LLM Transfer)")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"EI+TF Only: {ei_tf_only}")
    print(f"\nFull Match Accuracy: {accuracy_full:.2f}% ({correct_full}/{total})")
    print("\nPer-Axis Accuracy:")
    for axis in eval_axes:
        print(f"  {axis}: {accuracy_per_axis[axis]:.2f}% ({correct_per_axis[axis]}/{total})")
    print("="*50)
    
    # 儲存結果
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"baseline_results_{timestamp}.json"
    
    output_data = {
        "summary": summary,
        "config": {
            "mode": "baseline",
            "ei_tf_only": ei_tf_only,
            "max_samples": max_samples,
            "min_text_length": min_text_length,
        },
        "results": results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MBTI Style Transfer Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Baseline 評估（不用 LLM，直接用 classifier 判斷 neutral text）
    python evaluate_style_transfer.py --baseline
    python evaluate_style_transfer.py --baseline --ei_tf_only

    # 使用 base model（不用 LoRA adapter）做 style transfer
    python evaluate_style_transfer.py --no_lora
    python evaluate_style_transfer.py --no_lora --ei_tf_only

    # 使用完整 16 種 MBTI（with LoRA）
    python evaluate_style_transfer.py --adapter_path ./mbti-transfer-lora/final

    # 使用 EI+TF 4 種類型（with LoRA）
    python evaluate_style_transfer.py --adapter_path ./mbti-transfer-lora/final --ei_tf_only

    # 固定轉換成 ET 類型（EI+TF 模式）
    python evaluate_style_transfer.py --adapter_path ./mbti-transfer-lora/final --ei_tf_only --target_mbti ET

    # 固定轉換成 INTJ（Full 模式）
    python evaluate_style_transfer.py --adapter_path ./mbti-transfer-lora/final --target_mbti INTJ

    # 限制樣本數 + 調整最小文本長度
    python evaluate_style_transfer.py --adapter_path ./mbti-transfer-lora/final --ei_tf_only --max_samples 100 --min_text_length 50
    """)
    
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline 模式：不用 LLM，直接用 classifier 判斷 neutral text")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路徑（LLM 模式需要）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基礎模型 (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target_mbti", type=str, default=None,
                        help="固定目標 MBTI 類型 (e.g., INTJ 或 ET)。不設定則使用 dataset 中的 label")
    parser.add_argument("--ei_tf_only", action="store_true",
                        help="只使用 EI+TF 的 4 類型 (ET, EF, IT, IF)")
    parser.add_argument("--no_lora", action="store_true",
                        help="不使用 LoRA，直接用 base model 做 style transfer")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大評估樣本數 (default: 全部)")
    parser.add_argument("--min_text_length", type=int, default=30,
                        help="最小文本長度過濾 (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="輸出結果 JSON 檔案路徑")
    parser.add_argument("--device", type=str, default="cuda",
                        help="運算裝置 (default: cuda)")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Baseline 模式：不用 LLM
        evaluate_baseline(
            ei_tf_only=args.ei_tf_only,
            max_samples=args.max_samples,
            output_file=args.output,
            min_text_length=args.min_text_length,
        )
    else:
        # LLM 模式
        if args.adapter_path is None and not args.no_lora:
            parser.error("--adapter_path is required for LLM mode (use --baseline for baseline evaluation, or --no_lora to use base model without adapter)")
        
        evaluate(
            adapter_path=args.adapter_path,
            base_model=args.base_model,
            target_mbti=args.target_mbti,
            ei_tf_only=args.ei_tf_only,
            max_samples=args.max_samples,
            output_file=args.output,
            device=args.device,
            min_text_length=args.min_text_length,
            no_lora=args.no_lora,
        )


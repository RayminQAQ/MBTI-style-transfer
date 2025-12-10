"""
MBTI Style Transfer Evaluation Script (LLM Classifier)
使用同一個 LLM 進行 Style Transfer 和 MBTI Classification
"""

import os
import argparse
import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AXES = ["IE", "NS", "FT", "PJ"]

# EI + TF 模式的映射
EI_TF_MAPPING = {
    "ET": ["ENTJ", "ENTP", "ESTJ", "ESTP"],
    "EF": ["ENFJ", "ENFP", "ESFJ", "ESFP"],
    "IT": ["INTJ", "INTP", "ISTJ", "ISTP"],
    "IF": ["INFJ", "INFP", "ISFJ", "ISFP"],
}

VALID_MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]


def get_ei_tf_type(mbti_type: str) -> str:
    """將 16 種 MBTI 轉換成 EI+TF 的 4 種類型"""
    ei = mbti_type[0]  # E or I
    tf = mbti_type[2]  # T or F
    return f"{ei}{tf}"


class LLMModel:
    """LLM 模型封裝，同時用於 Style Transfer 和 Classification"""
    def __init__(self, adapter_path=None, base_model="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        self.device = device
        self.adapter_path = adapter_path
        self.base_model_name = base_model
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
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        return response
    
    def classify(self, text, ei_tf_only=False, max_new_tokens=64):
        """用 LLM 分類文本的 MBTI 類型"""
        if ei_tf_only:
            prompt = f"""<|im_start|>system
You are an expert Psycholinguist specializing in MBTI personality analysis.
Your task is to classify text into exactly one of these 4 composite types: **ET, EF, IT, IF**.

### Classification Logic (Step-by-Step)
1. **Analyze Energy (First Letter):**
   - **E (Extraverted):** Action-oriented, "we/us", fast-paced, external focus.
   - **I (Introverted):** Reflective, "I/my", deliberate pacing, internal focus.
   -> *Decide: E or I?*

2. **Analyze Nature (Second Letter):**
   - **T (Thinking):** Objective, logical, critical, "competence", impersonal.
   - **F (Feeling):** Subjective, emotional, empathetic, "harmony", personal.
   -> *Decide: T or F?*

3. **Combine:** Output the First Letter + Second Letter.

### Valid Output Codes
- **ET** (Extraverted Thinking)
- **EF** (Extraverted Feeling)
- **IT** (Introverted Thinking)
- **IF** (Introverted Feeling)

### Examples (Few-Shot)
Text: "Let's move fast and get this done! The data proves we are right."
Output: ET

Text: "I feel that we should consider everyone's emotions before deciding. It's important to me."
Output: IF

### Constraint
- Output **ONLY** the two-letter code.
- **NEVER** output codes like "EI", "TF", "IE", or "FT".
- Do not provide explanations.<|im_end|>
<|im_start|>user
Analyze the writing style and classify this text:
{text}<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = prompt = f"""<|im_start|>system
You are an expert MBTI Profiler and Linguist.
Your task is to analyze the input text and classify the author into exactly one of the 16 MBTI types.

### Analysis Logic (Step-by-Step)
Analyze the text strictly across these 4 dimensions. For each, pick ONE letter:

1. **E vs I (Energy Source)**
   - **E (Extraversion):** Action-oriented, uses social markers ("we", "let's"), fast-paced, assertive.
   - **I (Introversion):** Reflective, uses internal monologues ("I think", "I feel"), deliberate, depth-focused.

2. **S vs N (Information Processing)**
   - **S (Sensing):** Concrete, literal, focuses on facts/data/sensory details, "what is happening now".
   - **N (Intuition):** Abstract, metaphorical, focuses on patterns/future possibilities/theories, "what could be".

3. **T vs F (Decision Making)**
   - **T (Thinking):** Logical, objective, critical, focuses on competence/truth/cause-and-effect.
   - **F (Feeling):** Value-based, empathetic, focuses on people/harmony/personal impact.

4. **J vs P (Lifestyle/Structure)**
   - **J (Judging):** Decisive, conclusive, structured, likes closure ("The plan is...", "We must...").
   - **P (Perceiving):** Open-ended, adaptable, tentative, keeps options open ("Maybe...", "Let's see...").

### Few-Shot Examples
Text: "The underlying theoretical framework suggests a paradigm shift. We must logically restructure the system to ensure long-term efficiency."
Result: INTJ
(Analysis: I-focus, Abstract/Theory=N, Logical/System=T, Directive/Restructure=J)

Text: "Wow! Look at these colors! I want to try painting this right now. It feels so exciting to just go with the flow!"
Result: ESFP
(Analysis: Expressive/Action=E, Sensory details=S, Emotional/Excitement=F, Spontaneous=P)

### Constraints
- Output **ONLY** the 4-letter code from the valid list below.
- **Valid List:** INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP.
- Do not provide reasoning or explanation.<|im_end|>
<|im_start|>user
Analyze the writing style and classify this text:
{text}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # 低 temperature 讓分類更確定
                do_sample=False,  # greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        # 解析 MBTI 類型
        predicted_type = self._parse_mbti_response(response, ei_tf_only)
        # Sometimes the LLM will return the type code in reversed order
        if predicted_type == "UNKNOWN":
            predicted_type = self._parse_mbti_response(response[::-1], ei_tf_only)
        
        return predicted_type, response
    
    def _parse_mbti_response(self, response: str, ei_tf_only: bool) -> str:
        """從 LLM 回應中解析 MBTI 類型"""
        response_upper = response.upper().strip()
        
        if ei_tf_only:
            valid_types = ["ET", "EF", "IT", "IF"]
            # 直接匹配
            for t in valid_types:
                if t in response_upper:
                    return t
            # 嘗試從 4 字母 MBTI 提取
            for mbti in VALID_MBTI_TYPES:
                if mbti in response_upper:
                    return get_ei_tf_type(mbti)
            return "UNKNOWN"
        else:
            # 嘗試匹配 16 種 MBTI
            for mbti in VALID_MBTI_TYPES:
                if mbti in response_upper:
                    return mbti
            # 嘗試用 regex 找 4 個字母的組合
            match = re.search(r'\b([EI][NS][TF][JP])\b', response_upper)
            if match:
                return match.group(1)
            return "UNKNOWN"


def evaluate(
    adapter_path: str = None,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    target_mbti: str = None,
    ei_tf_only: bool = False,
    max_samples: int = None,
    output_file: str = None,
    device: str = "cuda",
    min_text_length: int = 30,
    no_lora: bool = False,
):
    """
    評估 Style Transfer 模型（使用 LLM 作為 Classifier）
    """
    
    # 載入 dataset
    print("Loading dataset: Binga288/mbti_style_transfer")
    raw_dataset = load_dataset("Binga288/mbti_style_transfer", split="test")
    
    # 處理資料
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
    
    # 載入模型
    print("\n" + "="*50)
    print("Loading LLM models...")
    print("="*50)
    
    # Transfer model（可能有 LoRA）
    effective_adapter = None if no_lora else adapter_path
    transfer_llm = LLMModel(effective_adapter, base_model, device)
    
    # Classifier model（始終不用 LoRA，使用 base model）
    print("\nLoading separate classifier model (no LoRA)...")
    classifier_llm = LLMModel(adapter_path=None, base_model=base_model, device=device)
    
    # 評估
    results = []
    correct_full = 0
    total = 0
    unknown_count = 0
    
    if ei_tf_only:
        eval_axes = ["IE", "FT"]
        print("\n[Mode] EI+TF only (4 types: ET, EF, IT, IF)")
    else:
        eval_axes = AXES
        print("\n[Mode] Full 16 MBTI types")
    
    correct_per_axis = {axis: 0 for axis in eval_axes}
    
    print("\n" + "="*50)
    print("Starting evaluation (LLM Classifier)...")
    print("="*50 + "\n")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        neutral_text = sample["neutral_text"]
        original_mbti = sample["original_mbti"]
        dataset_label = sample["mbti_type"]
        
        if target_mbti:
            target = target_mbti.upper()
            if ei_tf_only and len(target) == 4:
                target = get_ei_tf_type(target)
        else:
            target = dataset_label.upper()
        
        try:
            # Style Transfer（使用可能有 LoRA 的模型）
            transferred_text = transfer_llm.transfer(neutral_text, target)
            # logger.debug(f"Transferred text: {transferred_text[:200]}...")
            
            # LLM Classification（使用 base model，不用 LoRA）
            predicted_type, raw_response = classifier_llm.classify(transferred_text, ei_tf_only)
            # logging.debug(f"Raw response: {raw_response}")
            # logging.debug(f"Predicted type: {predicted_type}")

            
            if predicted_type == "UNKNOWN":
                unknown_count += 1
                logger.warning(f"Could not parse MBTI from response: {raw_response}")
                continue
            
            # 計算正確性
            if ei_tf_only:
                is_correct = (predicted_type == target)
            else:
                is_correct = (predicted_type == target)
            
            if is_correct:
                correct_full += 1
            
            # 每軸正確性
            for axis in eval_axes:
                if axis == "IE":
                    pred_letter = predicted_type[0]
                    target_letter = target[0]
                elif axis == "NS":
                    if not ei_tf_only:
                        pred_letter = predicted_type[1]
                        target_letter = target[1]
                    else:
                        continue
                elif axis == "FT":
                    if ei_tf_only:
                        pred_letter = predicted_type[1]  # EI+TF 格式中第二個字母是 T/F
                        target_letter = target[1]
                    else:
                        pred_letter = predicted_type[2]
                        target_letter = target[2]
                elif axis == "PJ":
                    if not ei_tf_only:
                        pred_letter = predicted_type[3]
                        target_letter = target[3]
                    else:
                        continue
                
                if pred_letter == target_letter:
                    correct_per_axis[axis] += 1
            
            total += 1
            
            result = {
                "idx": idx,
                "neutral_text": neutral_text[:200] + "..." if len(neutral_text) > 200 else neutral_text,
                "target_mbti": target,
                "original_mbti": original_mbti,
                "transferred_text": transferred_text[:200] + "..." if len(transferred_text) > 200 else transferred_text,
                "predicted_type": predicted_type,
                "raw_classifier_response": raw_response,
                "is_correct": is_correct,
            }
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                acc = correct_full / total * 100 if total > 0 else 0
                print(f"\n[Progress] {idx+1}/{len(samples)} | Accuracy: {acc:.2f}% | Unknown: {unknown_count}")
            
        except Exception as e:
            print(f"\n[Error] Sample {idx}: {e}")
            continue
    
    # 統計
    accuracy_full = correct_full / total * 100 if total > 0 else 0
    accuracy_per_axis = {axis: correct_per_axis[axis] / total * 100 if total > 0 else 0 for axis in eval_axes}
    
    summary = {
        "total_samples": total,
        "unknown_count": unknown_count,
        "target_mbti_mode": target_mbti if target_mbti else "dataset_label",
        "ei_tf_only": ei_tf_only,
        "no_lora": no_lora,
        "classifier": "LLM (base model, no LoRA)",
        "accuracy_full_match": accuracy_full,
        "accuracy_per_axis": accuracy_per_axis,
        "correct_full": correct_full,
        "correct_per_axis": correct_per_axis,
    }
    
    # 輸出結果
    print("\n" + "="*50)
    print("EVALUATION RESULTS (LLM Classifier)" + (" - No LoRA" if no_lora else " - With LoRA"))
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Unknown/Unparseable: {unknown_count}")
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
        output_file = f"eval_llm_classifier_{timestamp}.json"
    
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
            "classifier": "LLM",
        },
        "results": results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary, results


def evaluate_baseline(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ei_tf_only: bool = False,
    max_samples: int = None,
    output_file: str = None,
    device: str = "cuda",
    min_text_length: int = 30,
):
    """
    Baseline 評估：直接用 LLM 分類 neutral text（無 Style Transfer）
    """
    
    print("Loading dataset: Binga288/mbti_style_transfer")
    raw_dataset = load_dataset("Binga288/mbti_style_transfer", split="test")
    
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
    
    print("\n" + "="*50)
    print("Loading LLM model for classification...")
    print("="*50)
    
    llm = LLMModel(adapter_path=None, base_model=base_model, device=device)
    
    results = []
    correct_full = 0
    total = 0
    unknown_count = 0
    
    if ei_tf_only:
        eval_axes = ["IE", "FT"]
        print("\n[Mode] EI+TF only (4 types: ET, EF, IT, IF)")
    else:
        eval_axes = AXES
        print("\n[Mode] Full 16 MBTI types")
    
    correct_per_axis = {axis: 0 for axis in eval_axes}
    
    print("\n" + "="*50)
    print("Starting BASELINE evaluation (LLM Classifier, no transfer)...")
    print("="*50 + "\n")
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating Baseline")):
        neutral_text = sample["neutral_text"]
        original_mbti = sample["original_mbti"]
        target = sample["mbti_type"].upper()
        
        try:
            # 直接用 LLM 分類 neutral text
            predicted_type, raw_response = llm.classify(neutral_text, ei_tf_only)
            logging.debug(f"Predicted type: {predicted_type}")
            logging.debug(f"Raw response: {raw_response}")
            
            if predicted_type == "UNKNOWN":
                unknown_count += 1
                continue
            
            if ei_tf_only:
                is_correct = (predicted_type == target)
            else:
                is_correct = (predicted_type == target)
            
            if is_correct:
                correct_full += 1
            
            for axis in eval_axes:
                if axis == "IE":
                    pred_letter = predicted_type[0]
                    target_letter = target[0]
                elif axis == "NS":
                    if not ei_tf_only:
                        pred_letter = predicted_type[1]
                        target_letter = target[1]
                    else:
                        continue
                elif axis == "FT":
                    if ei_tf_only:
                        pred_letter = predicted_type[1]
                        target_letter = target[1]
                    else:
                        pred_letter = predicted_type[2]
                        target_letter = target[2]
                elif axis == "PJ":
                    if not ei_tf_only:
                        pred_letter = predicted_type[3]
                        target_letter = target[3]
                    else:
                        continue
                
                if pred_letter == target_letter:
                    correct_per_axis[axis] += 1
            
            total += 1
            
            result = {
                "idx": idx,
                "neutral_text": neutral_text[:200] + "..." if len(neutral_text) > 200 else neutral_text,
                "target_mbti": target,
                "original_mbti": original_mbti,
                "predicted_type": predicted_type,
                "raw_classifier_response": raw_response,
                "is_correct": is_correct,
            }
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                acc = correct_full / total * 100 if total > 0 else 0
                print(f"\n[Progress] {idx+1}/{len(samples)} | Accuracy: {acc:.2f}% | Unknown: {unknown_count}")
            
        except Exception as e:
            print(f"\n[Error] Sample {idx}: {e}")
            continue
    
    accuracy_full = correct_full / total * 100 if total > 0 else 0
    accuracy_per_axis = {axis: correct_per_axis[axis] / total * 100 if total > 0 else 0 for axis in eval_axes}
    
    summary = {
        "total_samples": total,
        "unknown_count": unknown_count,
        "mode": "baseline (LLM classifier, no transfer)",
        "ei_tf_only": ei_tf_only,
        "classifier": "LLM",
        "accuracy_full_match": accuracy_full,
        "accuracy_per_axis": accuracy_per_axis,
        "correct_full": correct_full,
        "correct_per_axis": correct_per_axis,
    }
    
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS (LLM Classifier, No Transfer)")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Unknown/Unparseable: {unknown_count}")
    print(f"EI+TF Only: {ei_tf_only}")
    print(f"\nFull Match Accuracy: {accuracy_full:.2f}% ({correct_full}/{total})")
    print("\nPer-Axis Accuracy:")
    for axis in eval_axes:
        print(f"  {axis}: {accuracy_per_axis[axis]:.2f}% ({correct_per_axis[axis]}/{total})")
    print("="*50)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"baseline_llm_classifier_{timestamp}.json"
    
    output_data = {
        "summary": summary,
        "config": {
            "mode": "baseline",
            "base_model": base_model,
            "ei_tf_only": ei_tf_only,
            "max_samples": max_samples,
            "min_text_length": min_text_length,
            "classifier": "LLM",
        },
        "results": results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return summary, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MBTI Style Transfer Model (Using LLM as Classifier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Baseline 評估（不做 Style Transfer，直接用 LLM 分類 neutral text）
    python evaluate_llm_classifier.py --baseline
    python evaluate_llm_classifier.py --baseline --ei_tf_only

    # 使用 base model（不用 LoRA）做 style transfer + LLM 分類
    python evaluate_llm_classifier.py --no_lora
    python evaluate_llm_classifier.py --no_lora --ei_tf_only

    # 使用 LoRA 做 style transfer + LLM 分類
    python evaluate_llm_classifier.py --adapter_path ./mbti-transfer-lora/final
    python evaluate_llm_classifier.py --adapter_path ./mbti-transfer-lora/final --ei_tf_only

    # 限制樣本數
    python evaluate_llm_classifier.py --adapter_path ./mbti-transfer-lora/final --max_samples 100
    """)
    
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline 模式：不做 Style Transfer，直接用 LLM 分類 neutral text")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter 路徑")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="基礎模型 (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target_mbti", type=str, default=None,
                        help="固定目標 MBTI 類型")
    parser.add_argument("--ei_tf_only", action="store_true",
                        help="只使用 EI+TF 的 4 類型")
    parser.add_argument("--no_lora", action="store_true",
                        help="不使用 LoRA，直接用 base model 做 style transfer")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大評估樣本數")
    parser.add_argument("--min_text_length", type=int, default=30,
                        help="最小文本長度過濾")
    parser.add_argument("--output", type=str, default=None,
                        help="輸出結果 JSON 檔案路徑")
    parser.add_argument("--device", type=str, default="cuda",
                        help="運算裝置")
    
    args = parser.parse_args()
    
    if args.baseline:
        evaluate_baseline(
            base_model=args.base_model,
            ei_tf_only=args.ei_tf_only,
            max_samples=args.max_samples,
            output_file=args.output,
            device=args.device,
            min_text_length=args.min_text_length,
        )
    else:
        if args.adapter_path is None and not args.no_lora:
            parser.error("--adapter_path is required (use --baseline for baseline, or --no_lora for base model)")
        
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


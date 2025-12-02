import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict

# ================= 配置區 (Configuration) =================
load_dotenv()

API_KEY = os.getenv("MODEL_API_KEY")
BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# 參數設定
BATCH_SIZE = 60           # 每次發送幾篇貼文給 LLM（分割後每篇很短，可以開大）
NUM_WORKERS = 8           # 並行線程數（根據 API rate limit 調整，建議 4-16）
INPUT_FILE = "../data/mbti_1.csv"
OUTPUT_FILE = "../data/mbti_1_neutral.csv"
PROGRESS_FILE = "../data/mbti_1_progress.jsonl"  # 中間進度檔
MAX_RETRIES = 1
MIN_POST_LENGTH = 20      # 過濾太短的貼文（URL、空白等）

# =========================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
progress_lock = threading.Lock()  # 線程鎖，保護檔案寫入


def get_system_prompt():
    """定義 System Prompt，強調 JSON 輸出與中性化標準"""
    return """You are a text normalization engine. You map styled MBTI text to plain, objective English.

# Task
Convert the list of user sentences into **Neutral, Matter-of-Fact** sentences.
- Keep the core semantic meaning.
- Remove emotional intensity, specific personality quirks, metaphors, and filler words.
- The output must be grammatically correct but culturally colorless.
- If the input is a URL or meaningless text, return it as-is.

# Output Format
You must output a JSON object containing a key "results" which is a list of objects.
Example:
{
  "results": [
    {"id": "0_1", "neutral_text": "Rewritten text for id 0_1"},
    {"id": "0_2", "neutral_text": "Rewritten text for id 0_2"}
  ]
}"""


def process_batch(batch_data: List[Dict]) -> List[Dict]:
    """呼叫 LLM 處理一個批次的資料"""
    user_content = json.dumps(batch_data, ensure_ascii=False)
    
    prompt = f"""# Instructions
Process the following JSON data. Return the result in the specified JSON format.
Ensure the 'id' in the output matches the input exactly (as string).

# Input Data
{user_content}"""

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=4096
            )
            
            content = response.choices[0].message.content
            
            # 嘗試修復常見的 JSON 問題
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise
            
            if "results" not in parsed or not isinstance(parsed["results"], list):
                raise ValueError("Invalid JSON structure returned")
                
            return parsed["results"]

        except json.JSONDecodeError as e:
            print(f"\n[Warning] JSON parse failed (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            print(f"[Debug] Response (first 500 chars): {content[:500]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"[Error] Skipping batch.")
                return []
        except Exception as e:
            print(f"\n[Warning] Batch failed (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"[Error] Skipping batch.")
                return []


def expand_posts(df: pd.DataFrame) -> pd.DataFrame:
    """
    展開 posts 欄位：把每個人的 50 篇貼文分割成獨立 rows
    ID 格式: {person_id}_{post_idx}
    """
    rows = []
    for person_id, row in df.iterrows():
        posts = row['posts'].split('|||')
        for post_idx, post in enumerate(posts):
            post = post.strip()
            # 過濾太短或純 URL 的貼文
            if len(post) >= MIN_POST_LENGTH and not post.startswith('http'):
                rows.append({
                    'post_id': f"{person_id}_{post_idx}",
                    'person_id': person_id,
                    'post_idx': post_idx,
                    'type': row['type'],
                    'text': post
                })
    
    return pd.DataFrame(rows)


def load_progress() -> Dict[str, str]:
    """載入已處理的進度"""
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    progress[record['post_id']] = record['neutral_text']
                except:
                    continue
    return progress


def save_progress_batch(results: List[Dict]):
    """批次追加寫入進度（線程安全）"""
    with progress_lock:
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps({'post_id': res['post_id'], 'neutral_text': res['neutral_text']}, ensure_ascii=False) + '\n')


def reassemble_posts(original_df: pd.DataFrame, progress: Dict[str, str]) -> pd.DataFrame:
    """
    重新組合處理後的貼文，恢復成原始結構
    每個人的貼文用 ||| 連接
    """
    results = []
    
    for person_id, row in original_df.iterrows():
        original_posts = row['posts'].split('|||')
        neutral_posts = []
        
        for post_idx, original_post in enumerate(original_posts):
            post_id = f"{person_id}_{post_idx}"
            original_post = original_post.strip()
            
            if post_id in progress:
                # 有處理過的中性化版本
                neutral_posts.append(progress[post_id])
            elif len(original_post) < MIN_POST_LENGTH or original_post.startswith('http'):
                # 太短或是 URL，保留原文
                neutral_posts.append(original_post)
            else:
                # 沒處理到的，保留原文（理論上不應該發生）
                neutral_posts.append(original_post)
        
        results.append({
            'id': person_id,
            'type': row['type'],
            'original_posts': row['posts'],
            'neutral_posts': '|||'.join(neutral_posts),
            'instruction': f"Rewrite the following text to sound like an {row['type']} personality."
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("MBTI Posts Destylization (Split Version)")
    print("=" * 60)
    
    # 1. 讀取原始資料
    print("\n[1/5] Loading original data...")
    raw_data = pd.read_csv(INPUT_FILE)
    print(f"  Total persons: {len(raw_data)}")
    
    # 2. 展開 posts
    print("\n[2/5] Expanding posts...")
    expanded_df = expand_posts(raw_data)
    print(f"  Total posts after expansion: {len(expanded_df)}")
    print(f"  Average posts per person: {len(expanded_df) / len(raw_data):.1f}")
    
    # 3. 載入已處理進度
    print("\n[3/5] Loading progress...")
    progress = load_progress()
    print(f"  Already processed: {len(progress)} posts")
    
    # 過濾已處理的
    remaining_df = expanded_df[~expanded_df['post_id'].isin(progress.keys())]
    print(f"  Remaining to process: {len(remaining_df)} posts")
    
    if len(remaining_df) == 0:
        print("\n  All posts already processed! Skipping to reassembly...")
    else:
        # 4. 批次處理（多線程）
        print(f"\n[4/5] Processing posts (BATCH_SIZE={BATCH_SIZE}, WORKERS={NUM_WORKERS})...")
        batches = [remaining_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(remaining_df), BATCH_SIZE)]
        
        def process_single_batch(batch):
            """處理單個 batch 並回傳結果"""
            batch_input = [
                {"id": row['post_id'], "text": row['text']} 
                for _, row in batch.iterrows()
            ]
            api_results = process_batch(batch_input)
            
            # 整理結果
            valid_results = []
            for res in api_results:
                post_id = res.get('id')
                neutral_text = res.get('neutral_text', '')
                if post_id and neutral_text:
                    valid_results.append({'post_id': post_id, 'neutral_text': neutral_text})
            return valid_results
        
        # 使用 ThreadPoolExecutor 並行處理
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_single_batch, batch): i for i, batch in enumerate(batches)}
            
            with tqdm(total=len(batches), desc="Processing") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        # 儲存結果（線程安全）
                        if batch_results:
                            save_progress_batch(batch_results)
                            for res in batch_results:
                                progress[res['post_id']] = res['neutral_text']
                    except Exception as e:
                        print(f"\n[Error] Batch failed: {e}")
                    pbar.update(1)
    
    # 5. 重新組合並輸出
    print("\n[5/5] Reassembling posts...")
    final_df = reassemble_posts(raw_data, progress)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Processing Complete!")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Total records: {len(final_df)}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

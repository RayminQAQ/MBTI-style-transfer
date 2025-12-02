import csv
import random
import pandas as pd
from local_test import predict_mbti   # <-- 你本地推論程式 (修改成你的檔名)
                                                # 如果檔名是 test_classifier.py 就改 import test_classifier

DATASET_PATH = "mbti_1.csv"
OUTPUT_CSV = "mbti_test_100_results.csv"


# 1. 載入資料
df = pd.read_csv(DATASET_PATH)

# 切割 posts → 單句資料
rows = []
for _, row in df.iterrows():
    posts = row["posts"].split("|||")
    for p in posts:
        if len(p.strip()) > 10:
            rows.append((p.strip(), row["type"]))

# 2. 隨機抽 100 筆
samples = random.sample(rows, 100)

# 3. 寫入 CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "response", "real_answer"])

    for text, real_mbti in samples:
        pred_mbti, conf, detail = predict_mbti(text)
        writer.writerow([text, pred_mbti, real_mbti])

print(f"✔ 已輸出測試結果: {OUTPUT_CSV}")


# 4. 計算整體準確率
correct = 0
axis_correct = {"IE":0, "NS":0, "FT":0, "PJ":0}

for _, pred, real in samples:
    if pred == real:
        correct += 1

    # 計算單軸準確
    for i, axis in enumerate(["IE", "NS", "FT", "PJ"]):
        if pred[i] == real[i]:
            axis_correct[axis] += 1

print("\n===== 測試統計 =====")
print(f"整體 4-letter accuracy: {correct / 100:.3f}")

for axis in axis_correct:
    print(f"{axis} accuracy: {axis_correct[axis] / 100:.3f}")


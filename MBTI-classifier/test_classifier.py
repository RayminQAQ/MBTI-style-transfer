import os
import torch
import json
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

REPO_ID = "Owen12354/mbti_classifier"
CACHE_DIR = "./hf_mbti_models"
MAX_LEN = 128

AXES = ["IE", "NS", "FT", "PJ"]

AXIS_MAP = {
    "IE": {0: "I", 1: "E"},
    "NS": {0: "N", 1: "S"},
    "FT": {0: "F", 1: "T"},
    "PJ": {0: "P", 1: "J"}
}

# ------------------------------
# 第一次執行：下載全部模型
# ------------------------------
def download_all_models():
    if not os.path.exists(CACHE_DIR):
        print("Downloading from HuggingFace…")
        snapshot_download(repo_id=REPO_ID, local_dir=CACHE_DIR, local_dir_use_symlinks=False)
        print("Download completed!\n")


# ------------------------------
# 載入某一軸模型
# ------------------------------
def load_axis_model(axis):

    model_file = f"pytorch_model_{axis}.bin"
    model_path = os.path.join(CACHE_DIR, model_file)

    # 1) 載入結構
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # 2) 載入該軸的權重
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.eval()
    return tokenizer, model


# ------------------------------
# 單軸推論，回傳 softmax 機率
# ------------------------------
def predict_axis(text, tokenizer, model):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).flatten().tolist()  # list: [P(class0), P(class1)]

    return probs


# ------------------------------
# 主功能：回傳 MBTI + 四軸機率 + 最終可信度
# ------------------------------
def predict_mbti(text):

    final_mbti = ""
    final_confidence = 1.0      # 乘積
    axis_results = {}

    for axis in AXES:

        tokenizer, model = load_axis_model(axis)
        probs = predict_axis(text, tokenizer, model)

        # 判斷選哪個（取最大機率）
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


# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":

    download_all_models()

    while True:
        text = input("\n輸入一句話（exit 離開）：\n> ")

        if text.lower() == "exit":
            break

        mbti, conf, detail = predict_mbti(text)

        print("\n===============================")
        print(f"➡️ 最終預測： {mbti}")
        print(f"➡️ 整體可信度： {conf:.4f}")
        print("===============================\n")

        for axis in AXES:
            p0 = detail[axis]["prob_class0"]
            p1 = detail[axis]["prob_class1"]
            chosen = detail[axis]["chosen"]

            print(f"{axis}: {chosen}  |  P0={p0:.4f}  P1={p1:.4f}")

        print("\n===============================\n")

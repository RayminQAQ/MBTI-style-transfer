import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================
# 本地模型路徑設定
# ================================
MODEL_DIR = "./saved_models"   # 你的資料夾
MAX_LEN = 128

AXES = ["IE", "NS", "FT", "PJ"]

AXIS_MAP = {
    "IE": {0: "I", 1: "E"},
    "NS": {0: "N", 1: "S"},
    "FT": {0: "F", 1: "T"},
    "PJ": {0: "P", 1: "J"},
}


# ------------------------------
# 載入某一軸的本地模型
# ------------------------------
def load_axis_model(axis):

    model_path = os.path.join(MODEL_DIR, axis, f"{axis}.pt")

    # 1) 載入 BERT 結構
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # 2) 載入該軸的權重
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.eval()
    return tokenizer, model


# ------------------------------
# 單軸推論，輸出 softmax 機率
# ------------------------------
def predict_axis(text, tokenizer, model):

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=MAX_LEN)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).flatten().tolist()
    return probs  # [P(class0), P(class1)]


# ------------------------------
# 主功能：回傳 MBTI + 可信度 + 四軸細節
# ------------------------------
def predict_mbti(text):

    final_mbti = ""
    final_conf = 1.0
    axis_results = {}

    for axis in AXES:

        tokenizer, model = load_axis_model(axis)
        probs = predict_axis(text, tokenizer, model)

        pred_class = int(torch.argmax(torch.tensor(probs)))
        pred_letter = AXIS_MAP[axis][pred_class]
        pred_prob = probs[pred_class]

        final_mbti += pred_letter
        final_conf *= pred_prob

        axis_results[axis] = {
            "prob_class0": probs[0],
            "prob_class1": probs[1],
            "chosen": pred_letter,
            "confidence": pred_prob,
        }

    return final_mbti, final_conf, axis_results


# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":

    print("🚀 Using Local .pt Models")

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

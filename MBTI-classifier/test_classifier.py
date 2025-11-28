import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

# ----------------------------
# 設定
# ----------------------------
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

# ----------------------------
# 下載整包模型（第一次會下載）
# ----------------------------
def download_all_models():
    print("Downloading models from HuggingFace...")
    snapshot_download(repo_id=REPO_ID, local_dir=CACHE_DIR, local_dir_use_symlinks=False)
    print("Download finished!\n")

# ----------------------------
# 載入某一軸的模型
# ----------------------------
def load_axis_model(axis):

    model_path = os.path.join(CACHE_DIR, f"pytorch_model_{axis}.bin")
    config_path = os.path.join(CACHE_DIR, f"config_{axis}.json")

    # 讀取 config.json
    with open(config_path, "r") as f:
        config = json.load(f)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        state_dict=torch.load(model_path, map_location="cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.eval()

    return tokenizer, model

# ----------------------------
# 單軸預測
# ----------------------------
def predict_axis(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)

    with torch.no_grad():
        logits = model(**inputs).logits

    return int(torch.argmax(logits))

# ----------------------------
# 主函式：預測 MBTI
# ----------------------------
def predict_mbti(text):

    mbti = ""

    for axis in AXES:
        tokenizer, model = load_axis_model(axis)
        pred = predict_axis(text, tokenizer, model)
        mbti += AXIS_MAP[axis][pred]

    return mbti

# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":

    if not os.path.exists(CACHE_DIR):
        download_all_models()

    print("\n=== MBTI Classifier Ready ===")

    while True:
        text = input("\n輸入一句話 (exit 離開)：\n> ")
        if text.lower() == "exit":
            break

        result = predict_mbti(text)
        print(f"➡️ 模型預測為： {result}\n")

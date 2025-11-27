import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 這裡放你在 HuggingFace 建的四個模型 repo
AXES_REPO = {
    "IE": "Owen12354/mbti_IE-classifier",
    "NS": "Owen12354/mbti_NS-classifier",
    "FT": "Owen12354/mbti_FT-classifier",
    "PJ": "Owen12354/mbti_PJ-classifier"
}

AXIS_MAP = {
    "IE": {0: "I", 1: "E"},
    "NS": {0: "N", 1: "S"},
    "FT": {0: "F", 1: "T"},
    "PJ": {0: "P", 1: "J"}
}


def load_axis_model(repo_name):
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)
    model.eval()
    return tokenizer, model


def predict_axis(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = int(torch.argmax(logits))
    return pred


def predict_mbti(text):
    mbti = ""

    for axis, repo in AXES_REPO.items():
        tokenizer, model = load_axis_model(repo)
        pred = predict_axis(text, tokenizer, model)
        mbti += AXIS_MAP[axis][pred]

    return mbti


if __name__ == "__main__":
    while True:
        text = input("\n輸入一句話（exit 離開）：\n> ")
        if text.lower() == "exit":
            break

        result = predict_mbti(text)
        print(f"\n➡️ 預測人格： {result}\n")

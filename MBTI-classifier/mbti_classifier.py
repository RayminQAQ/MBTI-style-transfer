"""
訓練四個模型：I/E、N/S、F/T、P/J
使用 microsoft/deberta-v3-base
輸出：
1. 四個 fine-tuned DeBERTa 權重 .pt
2. classification report（txt）
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ===========================
# Config
# ===========================
DATA_PATH = "mbti_1.csv"
MODEL_NAME = "microsoft/deberta-v3-base"   # <<< 改成 DeBERTa v3
SAVE_DIR = "./saved_models"
MAX_LENGTH = 128
TEST_SIZE = 0.2
EPOCHS = 6
BATCH = 16
LR = 2e-5

os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================
# 清理資料
# ===========================
def clean_text(t):
    t = re.sub(r"http\S+", "", str(t))
    return re.sub(r"\s+", " ", t).strip().lower()


def load_dataset(path):
    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        posts = row["posts"].split("|||")
        mbti = row["type"]

        for p in posts:
            p = clean_text(p)
            if len(p) > 10:
                rows.append({"type": mbti, "text": p})

    df = pd.DataFrame(rows).sample(15000, random_state=42)

    df["IE"] = df["type"].apply(lambda x: 0 if x[0] == "I" else 1)
    df["NS"] = df["type"].apply(lambda x: 0 if x[1] == "N" else 1)
    df["FT"] = df["type"].apply(lambda x: 0 if x[2] == "F" else 1)
    df["PJ"] = df["type"].apply(lambda x: 0 if x[3] == "P" else 1)

    return df


# ===========================
# Custom Trainer (weighted loss)
# ===========================
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        device = logits.device
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ===========================
# Metrics
# ===========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ===========================
# 訓練單一軸
# ===========================
def train_axis(df, axis):

    print(f"\n===== 訓練 {axis} =====")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[axis], random_state=42
    )

    # class weights
    counts = train_df[axis].value_counts().sort_index().values.astype(float)
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", axis]].rename(columns={axis: "labels"}))
    test_ds = Dataset.from_pandas(test_df[["text", axis]].rename(columns={axis: "labels"}))

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    test_ds = test_ds.map(tokenize, batched=True).remove_columns(["text"])

    collator = DataCollatorWithPadding(tokenizer)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # TrainingArguments
    args = TrainingArguments(
        output_dir=f"{SAVE_DIR}/{axis}",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=100,
        save_steps=999999,
        report_to="none",
    )

    trainer = CustomTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(axis, "evaluation:", trainer.evaluate())

    # === 儲存 loss/acc plot ===
    logs = trainer.state.log_history
    train_loss = [e["loss"] for e in logs if "loss" in e]
    eval_acc = [e["eval_accuracy"] for e in logs if "eval_accuracy" in e]
    eval_f1 = [e["eval_f1_macro"] for e in logs if "eval_f1_macro" in e]

    axis_dir = f"{SAVE_DIR}/{axis}"
    os.makedirs(axis_dir, exist_ok=True)

    if len(train_loss) > 1:
        plt.figure()
        plt.plot(train_loss)
        plt.title(f"{axis} Training Loss")
        plt.savefig(f"{axis_dir}/{axis}_loss.png")
        plt.close()

    if len(eval_acc) > 0:
        plt.figure()
        plt.plot(eval_acc, color="green")
        plt.title(f"{axis} Eval Accuracy")
        plt.savefig(f"{axis_dir}/{axis}_accuracy.png")
        plt.close()

    if len(eval_f1) > 0:
        plt.figure()
        plt.plot(eval_f1, color="red")
        plt.title(f"{axis} Eval F1")
        plt.savefig(f"{axis_dir}/{axis}_f1.png")
        plt.close()

    # 儲存模型
    torch.save(model.state_dict(), f"{SAVE_DIR}/{axis}/{axis}.pt")

    # Classification Report
    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    y_true = preds.label_ids

    report = classification_report(y_true, y_pred)
    print(report)

    with open(f"{SAVE_DIR}/{axis}/{axis}_report.txt", "w") as f:
        f.write(report)


# ===========================
# Main
# ===========================
def main():
    df = load_dataset(DATA_PATH)

    for axis in ["IE", "NS", "FT", "PJ"]:
        train_axis(df, axis)

    print("\nAll DeBERTa models trained & saved!\n")


if __name__ == "__main__":
    main()

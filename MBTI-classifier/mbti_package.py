"""
訓練四個人格軸模型（IE / NS / FT / PJ）
輸出：
1. 四個 fine-tuned BERT 權重 .pt
2. 四個軸的 Loss / Accuracy / F1 曲線圖
3. classification report（txt）
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA_PATH = "mbti_1.csv"
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "./saved_models"
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_SEED = 42
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5

os.makedirs(SAVE_DIR, exist_ok=True)


# =======================
# Text clean
# =======================
def clean_text(t):
    t = re.sub(r"http\S+", "", str(t))
    return re.sub(r"\s+", " ", t).strip().lower()


# =======================
# Load & preprocess dataset
# =======================
def load_dataset(path):
    df = pd.read_csv(path)
    rows = []

    for _, row in df.iterrows():
        t = row["type"]
        posts = row["posts"].split("|||")
        for p in posts:
            p = clean_text(p)
            if len(p) > 10:
                rows.append({"type": t, "text": p})

    df = pd.DataFrame(rows).sample(15000, random_state=42)

    df["IE"] = df["type"].apply(lambda x: 0 if x[0]=="I" else 1)
    df["NS"] = df["type"].apply(lambda x: 0 if x[1]=="N" else 1)
    df["FT"] = df["type"].apply(lambda x: 0 if x[2]=="F" else 1)
    df["PJ"] = df["type"].apply(lambda x: 0 if x[3]=="P" else 1)

    return df


# =======================
# Weighted loss Trainer
# =======================
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # 先不要移動到 GPU

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.get("logits")

        
        device = logits.device
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device) if self.class_weights is not None else None
        )

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# =======================
# Train one axis
# =======================
def train_axis(df, axis):

    print(f"\n===== 訓練 {axis} =====")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[axis],
        random_state=42
    )

    # class balance
    
# 正規化 + sqrt class weight（推薦）
# ============================================================
    class_counts = train_df[axis].value_counts().sort_index().values
    max_count = class_counts.max()

# sqrt 的平滑反比例權重
    weights = np.sqrt(max_count / class_counts)

    # normalize，讓 weight_sum = 1
    weights = weights / weights.sum()

    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"{axis} class weights =", class_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", axis]].rename(columns={axis:"labels"}))
    test_ds  = Dataset.from_pandas(test_df[["text", axis]].rename(columns={axis:"labels"}))

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    test_ds  = test_ds.map(tokenize, batched=True).remove_columns(["text"])

    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=f"{SAVE_DIR}/{axis}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=50,
        save_steps=999999,
        report_to="none",
        bf16=torch.cuda.is_available()
    )

    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )

    

    # Save model
    torch.save(model.state_dict(), f"{SAVE_DIR}/{axis}/{axis}.pt")

    # Evaluation
    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    y_true = preds_out.label_ids

    report_text = classification_report(y_true, preds)
    print(report_text)

    with open(f"{SAVE_DIR}/{axis}/{axis}_report.txt", "w") as f:
        f.write(report_text)

    # --------------------------
    # Draw training loss curve
    # --------------------------
    loss_values = []

    for log in trainer.state.log_history:
        if "loss" in log:
            loss_values.append(log["loss"])

    if len(loss_values) > 1:
        plt.plot(loss_values)
        plt.title(f"{axis} Training Loss")
        plt.xlabel("Logging Step")
        plt.ylabel("Loss")
        plt.savefig(f"{SAVE_DIR}/{axis}/{axis}_loss.png")
        plt.clf()
    else:
        print(f"[Warning] No training loss found for {axis}")

    return



# =======================
# Main
# =======================
def main():
    df = load_dataset(DATA_PATH)
    for axis in ["IE","NS","FT","PJ"]:
        train_axis(df, axis)

if __name__ == "__main__":
    main()

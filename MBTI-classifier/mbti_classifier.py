"""
訓練四個 BERT 模型：I/E、N/S、F/T、P/J
輸出：
1. 四個 fine-tuned BERT 權重 .pt
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
from sklearn.metrics import classification_report,f1_score,accuracy_score
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
MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "./saved_models"
MAX_LENGTH = 128
TEST_SIZE = 0.2
EPOCHS = 3
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

    df["IE"] = df["type"].apply(lambda x: 0 if x[0]=="I" else 1)
    df["NS"] = df["type"].apply(lambda x: 0 if x[1]=="N" else 1)
    df["FT"] = df["type"].apply(lambda x: 0 if x[2]=="F" else 1)
    df["PJ"] = df["type"].apply(lambda x: 0 if x[3]=="P" else 1)

    return df


# ===========================
# Custom Trainer (weighted loss)
# ===========================
class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        device = logits.device
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
# ===========================
# 單一軸訓練
# ===========================
def train_axis(df, axis):

    print(f"\n===== 訓練 {axis} =====")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[axis], random_state=42
    )

    # class weights（正規化 + sqrt）
    counts = train_df[axis].value_counts().sort_index().values.astype(float)
    freq=counts/counts.sum()
    weights=1.0/freq
    weights = weights / weights.mean()          # normalize
    weights = torch.tensor(weights, dtype=torch.float)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # dataset → HF Dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", axis]].rename(columns={axis: "labels"}))
    test_ds  = Dataset.from_pandas(test_df[["text", axis]].rename(columns={axis: "labels"}))

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text"])
    test_ds  = test_ds.map(tokenize, batched=True).remove_columns(["text"])

    collator = DataCollatorWithPadding(tokenizer)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # training args
    args = TrainingArguments(
        output_dir=f"{SAVE_DIR}/{axis}",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=100,
        save_steps=999999,
        report_to="none",
        #evaluation_strategy="epoch",
        #load_best_model_at_end=True,
        #metric_for_best_model="f1_macro",
        #greater_is_better=True,
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

    # train
    trainer.train()
    eval_result=trainer.evaluate()
    print(axis,"evaluation:",eval_result)
        # --------------------------
    # Training finished, extract logs for plotting
    # --------------------------
    logs = trainer.state.log_history

    train_loss = []
    eval_acc = []
    eval_f1 = []

    for entry in logs:
        if "loss" in entry:             # training loss
            train_loss.append(entry["loss"])
        if "eval_accuracy" in entry:    # eval accuracy
            eval_acc.append(entry["eval_accuracy"])
        if "eval_f1_macro" in entry:    # eval f1
            eval_f1.append(entry["eval_f1_macro"])

    axis_dir = f"{SAVE_DIR}/{axis}"
    os.makedirs(axis_dir, exist_ok=True)

    # --------------------------
    # Plot: Training Loss curve
    # --------------------------
    if len(train_loss) > 1:
        plt.figure()
        plt.plot(train_loss, label="Training Loss")
        plt.title(f"{axis} Training Loss")
        plt.xlabel("Logging Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{axis_dir}/{axis}_loss.png")
        plt.close()
    else:
        print(f"[Warning] No training loss found for {axis}")

    # --------------------------
    # Plot: Evaluation Accuracy 
    # --------------------------
    if len(eval_acc) > 0:
        plt.figure()
        plt.plot(eval_acc, label="Eval Accuracy", color="green")
        plt.title(f"{axis} Evaluation Accuracy")
        plt.xlabel("Eval Step")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{axis_dir}/{axis}_accuracy.png")
        plt.close()

    # --------------------------
    # Plot: Evaluation F1 Score
    # --------------------------
    if len(eval_f1) > 0:
        plt.figure()
        plt.plot(eval_f1, label="Eval F1 Macro", color="red")
        plt.title(f"{axis} Evaluation F1 Macro")
        plt.xlabel("Eval Step")
        plt.ylabel("F1 Macro")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{axis_dir}/{axis}_f1.png")
        plt.close()

    # save .pt
    torch.save(model.state_dict(), f"{SAVE_DIR}/{axis}/{axis}.pt")

    # evaluation
    pred = trainer.predict(test_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids

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

    print("\nAll models trained & saved in saved_models/\n")


if __name__ == "__main__":
    main()

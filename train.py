import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*byte fallback option.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*", category=UserWarning)

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ─────────────────────────────────────────────────────────────────────
# 1) Streaming dataset: tokenize each sample on-the-fly (saves RAM)
# ─────────────────────────────────────────────────────────────────────
class StreamingEssayDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─────────────────────────────────────────────────────────────────────
# 2) Main training flow
# ─────────────────────────────────────────────────────────────────────
def main():
    # a) Load & normalize the CSV
    df = pd.read_csv("./data/AI_Human.csv")
    # rename label/class → generated
    if "generated" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "generated"})
        elif "class" in df.columns:
            df = df.rename(columns={"class": "generated"})
        else:
            raise ValueError("CSV must contain 'generated', 'label' or 'class'")
    # map string labels → 0/1
    if df["generated"].dtype == object:
        df["generated"] = df["generated"].map({
            "Human": 0, "human": 0,
            "AI":    1, "ai":    1
        })

    # b) Filter out short essays & drop duplicates
    df = (
        df[df["text"].str.split().str.len() >= 150]
        .drop_duplicates(subset="text")
        .reset_index(drop=True)
    )

    # c) Balance AI vs. Human classes
    ai    = df[df.generated == 1]
    human = df[df.generated == 0]
    n     = min(len(ai), len(human))
    df    = (
        pd.concat([
            ai.sample(n=n, random_state=42),
            human.sample(n=n, random_state=42)
        ])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # d) Train/validation split
    texts  = df["text"].tolist()
    labels = df["generated"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # e) Tokenizer & model – DeBERTa-v3-large
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=2
    ).to(device)

    # f) Prepare streaming datasets
    train_dataset = StreamingEssayDataset(train_texts, train_labels, tokenizer)
    val_dataset   = StreamingEssayDataset(val_texts,   val_labels,   tokenizer)

    # g) Metrics & collator
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    collator = DataCollatorWithPadding(tokenizer)

    # h) TrainingArguments with frequent checkpointing
    training_args = TrainingArguments(
        output_dir="./saved_model_deberta_stream",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,

        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=2000,        # checkpoint every 2000 updates
        save_total_limit=5,     # keep only last 5

        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    # i) Trainer setup & train with resume support
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # j) Save model & tokenizer
    model.save_pretrained("./saved_model_deberta_stream")
    tokenizer.save_pretrained("./saved_model_deberta_stream")
    print("✅ Stream-mode DeBERTa training complete — saved in ./saved_model_deberta_stream")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main()

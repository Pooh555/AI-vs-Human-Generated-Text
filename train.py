import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load balanced data
df = pd.read_csv("./data/Balanced_Essay_Data.csv")
# (Make sure this path is correct!)

# Prepare
df = df[df["text"].str.split().str.len() >= 150].reset_index(drop=True)
texts = df["text"].tolist()
labels = df["generated"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenizer & Model
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model     = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2).to(device)

# Tokenization helper
def tokenize(batch):
    return tokenizer(batch, padding=True, truncation=True, max_length=512)

train_encodings = tokenize(train_texts)
val_encodings   = tokenize(val_texts)

class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __getitem__(self, idx):
        return {
            "input_ids":      torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels":         torch.tensor(self.labels[idx]),
        }
    def __len__(self):
        return len(self.labels)

train_dataset = EssayDataset(train_encodings, train_labels)
val_dataset   = EssayDataset(val_encodings,   val_labels)

# Metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Training args
training_args = TrainingArguments(
    output_dir="./saved_model_rl",      # separate directory for roberta-large
    num_train_epochs=3,
    per_device_train_batch_size=4,       # reduce if OOM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,       # simulates batch_size=8 if needed
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rl",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save
model.save_pretrained("./saved_model_rl")
tokenizer.save_pretrained("./saved_model_rl")
print("✅ roberta-large training complete.")

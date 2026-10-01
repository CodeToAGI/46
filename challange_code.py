"""EP46 - Production-grade text classifier, fine-tuned in under 100 lines.

    python ep46_text_classifier.py                      # fine-tune, evaluate, save
    python ep46_text_classifier.py "loved every minute" # predict with the saved model

Needs: pip install "transformers>=4.46" datasets torch scikit-learn accelerate
"""
import sys
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline, set_seed)

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
MODEL = "distilbert-base-uncased"
DATASET = "imdb"              # needs text+label columns, train/test. Try "banking77"
OUT_DIR = "ep46_classifier"
MAX_LEN = 256
SEED = 42
THRESHOLD = 0.75              # below this confidence -> "uncertain"


# ── 2. DATA ───────────────────────────────────────────────────────────────
def load_data(tok):
    ds = load_dataset(DATASET)
    names = ds["train"].features["label"].names
    small = ds["train"].shuffle(seed=SEED).select(range(4000))
    split = small.train_test_split(test_size=0.1, seed=SEED)
    test = ds["test"].shuffle(seed=SEED).select(range(1000))
    enc = lambda batch: tok(batch["text"], truncation=True, max_length=MAX_LEN)
    return (split["train"].map(enc, batched=True), split["test"].map(enc, batched=True),
            test.map(enc, batched=True), names)


# ── 3. METRICS ────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro",
                                                  zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": p,
            "recall": r, "f1": f1}


# ── 4. TRAIN ──────────────────────────────────────────────────────────────
def train():
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    train_ds, val_ds, test_ds, names = load_data(tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)})
    args = TrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=2, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, report_to="none", seed=SEED, fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    # ── 5. EVALUATE ON HELD-OUT TEST DATA ─────────────────────────────────
    out = trainer.predict(test_ds)
    print("test metrics:", {k: round(float(v), 4) for k, v in out.metrics.items()
                            if k.startswith("test_") and k != "test_loss"})
    print(confusion_matrix(out.label_ids, np.argmax(out.predictions, axis=-1)))
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)


# ── 6. SERVE ──────────────────────────────────────────────────────────────
def predict(texts):
    clf = pipeline("text-classification", model=OUT_DIR, truncation=True,
                   max_length=MAX_LEN, device=0 if torch.cuda.is_available() else -1)
    results = []
    for text, res in zip(texts, clf(texts, batch_size=32)):
        label = res["label"] if res["score"] >= THRESHOLD else "uncertain"
        results.append((text, label, round(res["score"], 3)))
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for row in predict(sys.argv[1:]):
            print(row)
    else:
        train()

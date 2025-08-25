import os, json, numpy as np
from dataclasses import dataclass
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    model_name: str = "distilbert-base-uncased"
    dataset: str = "tweet_eval"
    subset: str = "sentiment"   # 0: negative, 1: neutral, 2: positive
    max_len: int = 128
    out_dir: str = "w5d1_outputs"
    batch_train: int = 16
    batch_eval: int = 64
    epochs: int = 2
    lr: float = 5e-5
    seed: int = 42

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# ---------------------------- Data ----------------------------
raw = load_dataset(cfg.dataset, cfg.subset)
tok = AutoTokenizer.from_pretrained(cfg.model_name)

def tok_fn(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=cfg.max_len)

ds = raw.map(tok_fn, batched=True)
ds = ds.rename_column("label", "labels")
keep_cols = ["input_ids", "attention_mask", "labels"]
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep_cols])

# ---------------------------- Model ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=3)

def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ---------------------------- Train ----------------------------
args = TrainingArguments(
    output_dir=cfg.out_dir,
    per_device_train_batch_size=cfg.batch_train,
    per_device_eval_batch_size=cfg.batch_eval,
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=cfg.epochs,
    learning_rate=cfg.lr,
    fp16=True if os.environ.get("ACCELERATE_USE_F16","1")=="1" else False,
    load_best_model_at_end=False,
    report_to=[],
    seed=cfg.seed
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    compute_metrics=metrics_fn
)

trainer.train()
eval_res = trainer.evaluate()

# ---------------------------- Test Report ----------------------------
preds = np.argmax(trainer.predict(ds["test"]).predictions, axis=-1)
labels = ds["test"]["labels"]
report = classification_report(labels, preds, target_names=["neg","neu","pos"], digits=4)

with open(os.path.join(cfg.out_dir, "eval.json"), "w") as f:
    json.dump(eval_res, f, indent=2)
with open(os.path.join(cfg.out_dir, "report.txt"), "w") as f:
    f.write(report)

print("Validation:", eval_res)
print("\nTest classification report:\n", report)

# ---------------------------- Inference Helper ----------------------------
EXAMPLES = [
    "I’m thrilled with the new release—great job!",
    "This is fine, but it didn’t change much.",
    "Fantastic UI, but it crashes every hour. Please fix asap.",
    "Yeah, amazing update… totally didn’t break anything 🙄",
]

def predict_texts(texts: List[str]) -> List[Dict]:
    batch = tok(texts, truncation=True, padding=True, return_tensors="pt", max_length=cfg.max_len)
    out = model(**{k:v for k,v in batch.items()})
    probs = out.logits.softmax(-1).detach().numpy()
    labels = probs.argmax(-1)
    inv = {0:"neg",1:"neu",2:"pos"}
    return [{"text": t, "label": inv[int(y)], "probs": probs[i].round(4).tolist()} for i,(t,y) in enumerate(zip(texts, labels))]

results = predict_texts(EXAMPLES)
with open(os.path.join(cfg.out_dir, "quick_inference.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nQuick inference examples:")
for r in results:
    print(f"- {r['text']} -> {r['label']} {r['probs']}")

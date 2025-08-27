import os, json, numpy as np, pandas as pd, math
from dataclasses import dataclass
from typing import Dict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    model: str = "distilbert-base-uncased"
    csv_path: str = "vad_text_dataset.csv"   # columns: text,valence,arousal in [0,1]
    out_dir: str = "w5d3_vad_out"
    max_len: int = 128
    epochs: int = 3
    lr: float = 5e-5
    batch_train: int = 16
    batch_eval: int = 64
    seed: int = 42

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)

# ---------------------------- Data ----------------------------
assert os.path.exists(cfg.csv_path), f"Provide CSV at {cfg.csv_path} with columns text,valence,arousal"

try:
    df = pd.read_csv(cfg.csv_path)
    print(f"Successfully loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
except Exception as e:
    print(f"Error reading CSV: {e}")
    print("Please ensure the CSV has exactly 3 columns: text,valence,arousal")
    print("And that text values are properly quoted if they contain commas")
    exit(1)

assert set(["text","valence","arousal"]).issubset(df.columns), "Missing required columns"
assert len(df.columns) == 3, f"Expected 3 columns, got {len(df.columns)}: {list(df.columns)}"

# Validate data types
df["valence"] = pd.to_numeric(df["valence"], errors="coerce")
df["arousal"] = pd.to_numeric(df["arousal"], errors="coerce")
assert not df[["valence", "arousal"]].isna().any().any(), "Invalid numeric values in valence/arousal columns"
assert (df["valence"] >= 0).all() and (df["valence"] <= 1).all(), "Valence values must be between 0 and 1"
assert (df["arousal"] >= 0).all() and (df["arousal"] <= 1).all(), "Arousal values must be between 0 and 1"

print(f"Data validation passed. Shape: {df.shape}")

# split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=cfg.seed)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=cfg.seed)

# to HF Datasets (without saving to disk)
from datasets import Dataset
def to_ds(frame): return Dataset.from_pandas(frame.reset_index(drop=True))
train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

tok = AutoTokenizer.from_pretrained(cfg.model)

def preprocess(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=cfg.max_len)
    targets = np.stack([batch["valence"], batch["arousal"]], axis=1)  # shape [N,2]
    enc["labels"] = targets
    return enc

train_ds = train_ds.map(preprocess, batched=True)
val_ds   = val_ds.map(preprocess, batched=True)
test_ds  = test_ds.map(preprocess, batched=True)
keep = ["input_ids","attention_mask","labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])
test_ds  = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep])

# ---------------------------- Model (regression) ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    cfg.model, num_labels=2, problem_type="regression"
)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape [N,2]
    rmse_v = math.sqrt(((logits[:,0] - labels[:,0])**2).mean())
    rmse_a = math.sqrt(((logits[:,1] - labels[:,1])**2).mean())
    mae_v = np.abs(logits[:,0] - labels[:,0]).mean()
    mae_a = np.abs(logits[:,1] - labels[:,1]).mean()
    return {
        "rmse_valence": rmse_v,
        "rmse_arousal": rmse_a,
        "mae_valence": mae_v,
        "mae_arousal": mae_a
    }

args = TrainingArguments(
    output_dir=cfg.out_dir,
    per_device_train_batch_size=cfg.batch_train,
    per_device_eval_batch_size=cfg.batch_eval,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=cfg.lr,
    num_train_epochs=cfg.epochs,
    fp16=True,
    report_to=[],
    seed=cfg.seed
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

trainer.train()
val_metrics = trainer.evaluate()
test_logits = trainer.predict(test_ds).predictions
test_labels = np.array(test_df[["valence","arousal"]].values, dtype=float)
test_rmse_v = float(np.sqrt(((test_logits[:,0]-test_labels[:,0])**2).mean()))
test_rmse_a = float(np.sqrt(((test_logits[:,1]-test_labels[:,1])**2).mean()))

with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
    json.dump({"val":val_metrics, "test":{"rmse_v":test_rmse_v, "rmse_a":test_rmse_a}}, f, indent=2)

# quick demo
examples = [
    "That meeting left me calm and satisfied.",
    "I am furious this keeps breaking!",
    "Excited but a bit nervous before the big pitch."
]
batch = tok(examples, truncation=True, padding=True, return_tensors="pt", max_length=cfg.max_len)
pred = model(**batch).logits.detach().numpy().clip(0,1)
demo = [{"text":t, "valence":float(v), "arousal":float(a)} for t,(v,a) in zip(examples, pred)]
with open(os.path.join(cfg.out_dir, "demo_predictions.json"), "w") as f:
    json.dump(demo, f, indent=2)

print("Validation metrics:", val_metrics)
print("Test RMSE:", {"valence": test_rmse_v, "arousal": test_rmse_a})
print("Demo:", demo)

import os, json, numpy as np, random, nltk
import inspect
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
import torch

# ---------------------------- Config ----------------------------
@dataclass
class CFG:
    out_dir: str = "w5d2_outputs"
    seed: int = 42
    dataset: str = "tweet_eval"
    subset: str = "sentiment"   # labels: 0 neg, 1 neu, 2 pos
    max_len: int = 128
    model_ft: str = "distilbert-base-uncased"
    model_zeroshot: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    epochs: int = 2
    lr: float = 5e-5
    batch_train: int = 16
    batch_eval: int = 64

cfg = CFG()
os.makedirs(cfg.out_dir, exist_ok=True)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

# ---------------------------- Data ----------------------------
raw = load_dataset(cfg.dataset, cfg.subset)
label_names = ["neg","neu","pos"]
train_texts, train_y = raw["train"]["text"], raw["train"]["label"]
val_texts, val_y = raw["validation"]["text"], raw["validation"]["label"]
test_texts, test_y = raw["test"]["text"], raw["test"]["label"]

# ---------------------------- 1) Lexicon Baseline (very simple) ----------------------------# Tiny VADER demo for polarity; interpretable but limited (sarcasm/domain shift).
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def lexicon_predict(texts: List[str]) -> List[int]:
    out = []
    for t in texts:
        s = sia.polarity_scores(t)["compound"]
        # map: s>0.05 -> pos, s<-0.05 -> neg, else neu
        out.append(2 if s>0.05 else 0 if s<-0.05 else 1)
    return out

lex_preds = lexicon_predict(test_texts)
lex_res = {
    "accuracy": float(accuracy_score(test_y, lex_preds)),
    "f1_macro": float(f1_score(test_y, lex_preds, average="macro")),
    "report": classification_report(test_y, lex_preds, target_names=label_names, digits=4)
}

# ---------------------------- 2) Classical ML (TF-IDF + Logistic) ----------------------------
vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_train = vec.fit_transform(train_texts)
X_val = vec.transform(val_texts)
X_test = vec.transform(test_texts)

clf = LogisticRegression(max_iter=200, n_jobs=None, class_weight="balanced")
clf.fit(X_train, train_y)
ml_preds = clf.predict(X_test)
ml_res = {
    "accuracy": float(accuracy_score(test_y, ml_preds)),
    "f1_macro": float(f1_score(test_y, ml_preds, average="macro")),
    "report": classification_report(test_y, ml_preds, target_names=label_names, digits=4)
}

# ---------------------------- 3) Transformer Finetune ----------------------------
tok = AutoTokenizer.from_pretrained(cfg.model_ft)
def tok_fn(examples): return tok(examples["text"], truncation=True, padding="max_length", max_length=cfg.max_len)

ds = raw.map(tok_fn, batched=True)
ds = ds.rename_column("label","labels")
keep = ["input_ids","attention_mask","labels"]
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])

model = AutoModelForSequenceClassification.from_pretrained(cfg.model_ft, num_labels=3)

def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}

_ta_params = inspect.signature(TrainingArguments.__init__).parameters
_ta_kwargs = {
    "output_dir": os.path.join(cfg.out_dir, "ft"),
    "per_device_train_batch_size": cfg.batch_train,
    "per_device_eval_batch_size": cfg.batch_eval,
    "num_train_epochs": cfg.epochs,
    "learning_rate": cfg.lr,
    "fp16": True,
    "seed": cfg.seed,
}

if "evaluation_strategy" in _ta_params:
    _ta_kwargs["evaluation_strategy"] = "epoch"
elif "evaluate_during_training" in _ta_params:
    _ta_kwargs["evaluate_during_training"] = True

if "save_strategy" in _ta_params:
    _ta_kwargs["save_strategy"] = "no"

if "report_to" in _ta_params:
    _ta_kwargs["report_to"] = []

args = TrainingArguments(**_ta_kwargs)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], compute_metrics=metrics_fn)
trainer.train()

ft_logits = trainer.predict(ds["test"]).predictions
ft_preds = ft_logits.argmax(-1)
ft_res = {
    "accuracy": float(accuracy_score(test_y, ft_preds)),
    "f1_macro": float(f1_score(test_y, ft_preds, average="macro")),
    "report": classification_report(test_y, ft_preds, target_names=label_names, digits=4)
}

# ---------------------------- 4) Zero-shot / Off-the-shelf Transformer ----------------------------
zero = pipeline("text-classification", model=cfg.model_zeroshot, top_k=None, truncation=True)
zs_raw = zero(test_texts, batch_size=32)
# model returns labels like "negative","neutral","positive"
map_lab = {"negative":0,"neutral":1,"positive":2}
zs_preds = [map_lab[r["label"]] for r in zs_raw]
zs_res = {
    "accuracy": float(accuracy_score(test_y, zs_preds)),
    "f1_macro": float(f1_score(test_y, zs_preds, average="macro")),
    "report": classification_report(test_y, zs_preds, target_names=label_names, digits=4)
}

# ---------------------------- Summary + Examples ----------------------------
def summarize(name, res):
    return {"name": name, "accuracy": round(res["accuracy"],4), "f1_macro": round(res["f1_macro"],4)}

summary = [
    summarize("Lexicon (VADER)", lex_res),
    summarize("Classical ML (TF-IDF+LR)", ml_res),
    summarize("Transformer Finetune", ft_res),
    summarize("Zero-shot Transformer", zs_res),
]

examples = [
    "Love the speed, but it keeps freezing after 10 minutes.",
    "This is fine, nothing to write home about.",
    "Wow, just wow. Totally not broken at all 🙃",
    "Disappointed. Support didn’t respond for 3 days."
]

def predict_all(texts: List[str]) -> Dict[str, List[Tuple[str,str]]]:
    res = {"lexicon":[], "classic":[], "finetune":[], "zeroshot":[]}
    # lexicon
    inv = {0:"neg",1:"neu",2:"pos"}
    for t in texts:
        res["lexicon"].append((t, inv[lexicon_predict([t])[0]]))
        res["classic"].append((t, inv[int(clf.predict(vec.transform([t])))]))
        toks = tok(t, truncation=True, padding=True, return_tensors="pt", max_length=cfg.max_len)
        with torch.no_grad():
            p = model(**toks).logits.softmax(-1).argmax(-1).item()
        res["finetune"].append((t, inv[int(p)]))
        res["zeroshot"].append((t, zero(t)["label"][:3].lower()))  # neg/neu/pos
    return res

demo = predict_all(examples)

with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
    json.dump({"summary": summary}, f, indent=2)
with open(os.path.join(cfg.out_dir, "reports.json"), "w") as f:
    json.dump({
        "lexicon": lex_res["report"],
        "classic": ml_res["report"],
        "finetune": ft_res["report"],
        "zeroshot": zs_res["report"]
    }, f, indent=2)
with open(os.path.join(cfg.out_dir, "demo_predictions.json"), "w") as f:
    json.dump(demo, f, indent=2)

print("=== SUMMARY ==="); print(json.dumps(summary, indent=2))
print("\nSample predictions:")
for k,v in demo.items():
    print(f"\n{k.upper()}:")
    for t,lab in v:
        print("-", lab, "::", t)

"""Fine-tuning LoRA d'un SLM (Qwen2.5-0.5B) pour la classification d'intention.

Sur GPU (Colab T4) :   python scripts/train_lora.py --max-train 8000 --epochs 1
Test rapide sur CPU :  python scripts/train_lora.py --max-train 32 --max-eval 16 --epochs 1 --output /tmp/lora-smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from support_agent.config import load_config, resolve
from support_agent.data import augment_split, load_bitext_splits


def subsample(texts, labels, n, seed):
    if n is None or n >= len(texts):
        return texts, labels
    idx = random.Random(seed).sample(range(len(texts)), n)
    return [texts[i] for i in idx], [labels[i] for i in idx]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=cfg["classifier"]["base_model"])
    ap.add_argument("--output", default=str(resolve(cfg["classifier"]["adapter_dir"])))
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-eval", type=int, default=1000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--augment", action="store_true", help="ajoute des tickets façon e-mail (augmentation)")
    args = ap.parse_args()

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    splits = load_bitext_splits(cfg)
    labels = splits.labels
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    x_tr, y_tr = subsample(splits.train.texts, splits.train.labels, args.max_train, seed)
    if args.augment:
        # même nombre total d'exemples : la moitié est remplacée par sa version augmentée
        x_aug, y_aug = augment_split(x_tr, y_tr, seed, ratio=0.5)
        x_tr, y_tr = subsample(x_aug, y_aug, args.max_train, seed)
    x_va, y_va = subsample(splits.val.texts, splits.val.labels, args.max_eval, seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = cfg["classifier"]["max_length"]

    def to_ds(x, y):
        ds = Dataset.from_dict({"text": x, "label": [label2id[v] for v in y]})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_len), batched=True)

    train_ds, val_ds = to_ds(x_tr, y_tr), to_ds(x_va, y_va)

    use_gpu = torch.cuda.is_available()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Paramètres entraînables : {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")

    def compute_metrics(p):
        from sklearn.metrics import accuracy_score, f1_score

        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "macro_f1": f1_score(p.label_ids, preds, average="macro")}

    out = Path(args.output)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=0.05,  # ratio du nombre total de pas (transformers >= 5)
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=use_gpu,
        report_to=[],
        seed=seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    metrics = trainer.evaluate()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels))
    info = {
        "base_model": args.base_model, "n_train": len(x_tr), "n_val": len(x_va), "epochs": args.epochs,
        "lora_rank": args.rank, "augment": args.augment, "trainable_params": trainable, "total_params": total,
        "train_time_s": round(train_time, 1), "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
        "val_accuracy": metrics.get("eval_accuracy"), "val_macro_f1": metrics.get("eval_macro_f1"),
    }
    (out / "training_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

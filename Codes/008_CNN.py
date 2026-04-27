#!/usr/bin/env python3
"""
hyperparam_search_cnn_deepinsight.py
====================================

Random search per trovare automaticamente i migliori iperparametri per una CNN
su immagini DeepInsight 16x16 (array .npy) per classificazione binaria.

Lancio:
  python3 hyperparam_search_cnn_deepinsight.py

Output:
  <OUT_DIR>/
    leaderboard.csv
    best_params.json
    best_model.pt
    best_training_loss.png
    best_training_val_auc.png
    roc_test.png
    calibration_test.png
    confusion_test.txt
    results_best.json
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve, auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG (modifica solo qui)
# =============================================================================

class CFG:
    # Cartelle DeepInsight (devono contenere labels.csv + arrays/*.npy)
    TRAIN_DIR = Path("train")
    VAL_DIR   = Path("val")
    TEST_DIR  = Path("test")

    # Output search
    OUT_DIR = Path("cnn_hparam_search") / "deepinsight_16x16"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Search budget
    N_TRIALS = 30          # aumenta se vuoi più ricerca
    MAX_EPOCHS = 50        # epoche max per trial (con early stopping)
    PATIENCE = 8           # early stopping patience (per trial)

    # DataLoader
    NUM_WORKERS = 0

    # Decision threshold per confusion (finale best)
    CONF_THRESHOLD = 0.5
    CALIB_BINS = 10

    # Search space
    BATCH_CHOICES = [16, 32, 64]
    BASE_CH_CHOICES = [8, 16, 24]     # base channels (rete più piccola o più grande)
    DROPOUT_RANGE = (0.0, 0.5)
    LR_LOG10_RANGE = (-4.2, -2.6)     # ~ 6e-5 .. 2.5e-3
    WD_LOG10_RANGE = (-6.0, -2.5)     # ~ 1e-6 .. 3e-3


# =============================================================================
# Utils
# =============================================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    y = y.astype(int)
    counts = np.bincount(y)
    counts = np.maximum(counts, 1)
    weights = len(y) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)

def loguniform_10(rng: np.random.Generator, lo: float, hi: float) -> float:
    """sample 10**U(lo,hi)"""
    return float(10 ** rng.uniform(lo, hi))

def uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


# =============================================================================
# Dataset
# =============================================================================

class DeepInsightDataset(Dataset):
    """
    Legge labels.csv e carica arrays/*.npy indicati in colonna 'array'.
    Standardizza ogni immagine: (x-mean)/(std+eps) per stabilità.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        labels_path = self.root_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found in: {self.root_dir}")

        self.df = pd.read_csv(labels_path)
        if "target" not in self.df.columns:
            raise ValueError(f"'target' column missing in {labels_path}")
        if "array" not in self.df.columns:
            raise ValueError(f"'array' column missing in {labels_path} (expected .npy paths)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = int(row["target"])
        rel = row["array"]
        p = self.root_dir / rel
        arr = np.load(p).astype(np.float32)  # (16,16)

        # per-sample standardization (cruciale)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        return {"image": x, "target": torch.tensor(y, dtype=torch.long)}


# =============================================================================
# Model
# =============================================================================

class SmallCNN16_GN(nn.Module):
    """
    CNN piccola per 1x16x16, con GroupNorm (stabile su small data).
    base_ch controlla la capacità.
    """
    def __init__(self, base_ch: int = 16, dropout: float = 0.2):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # groups: deve dividere i channels
        def gn(ch):
            if ch <= 8:
                return nn.GroupNorm(2, ch)
            if ch <= 16:
                return nn.GroupNorm(4, ch)
            if ch <= 32:
                return nn.GroupNorm(8, ch)
            return nn.GroupNorm(8, ch)

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            gn(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            gn(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16->8

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            gn(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8->4
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, max(16, c2)),  # piccolo MLP head
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(16, c2), 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================================
# Training / Eval
# =============================================================================

@torch.no_grad()
def eval_loader(model, loader, device, criterion):
    model.eval()
    losses = []
    all_y, all_p, all_pred = [], [], []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

        losses.append(loss.item())
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs)
        all_pred.append(pred)

    y_true = np.concatenate(all_y).astype(int)
    p_pos = np.concatenate(all_p).astype(float)
    y_pred = np.concatenate(all_pred).astype(int)
    avg_loss = float(np.mean(losses))

    if len(np.unique(y_true)) >= 2:
        rocauc = float(roc_auc_score(y_true, p_pos))
        auprc = float(average_precision_score(y_true, p_pos))
    else:
        rocauc, auprc = float("nan"), float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "loss": avg_loss,
        "auc": rocauc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "y_true": y_true,
        "p_pos": p_pos,
        "y_pred": y_pred,
    }

def train_one_trial(
    trial_id: int,
    params: dict,
    train_ds: DeepInsightDataset,
    val_ds: DeepInsightDataset,
    class_weights: torch.Tensor,
):
    device = CFG.DEVICE
    model = SmallCNN16_GN(base_ch=params["base_ch"], dropout=params["dropout"]).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=CFG.NUM_WORKERS, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False,
        num_workers=CFG.NUM_WORKERS
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc = -1.0
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, CFG.MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["target"].to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val = eval_loader(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_auc"].append(val["auc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        metric = val["auc"] if not np.isnan(val["auc"]) else -1.0
        scheduler.step(metric)

        improved = (not np.isnan(val["auc"])) and (val["auc"] > best_auc + 1e-6)
        if improved:
            best_auc = val["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= CFG.PATIENCE:
            break

    return best_auc, best_state, history


# =============================================================================
# Plots for BEST
# =============================================================================

def plot_training(out_dir: Path, hist: dict):
    plt.figure(figsize=(7, 5))
    plt.plot(hist["train_loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BEST: Training/Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(hist["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("BEST: Validation AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_training_val_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc(out_dir: Path, y: np.ndarray, p: np.ndarray):
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (test) - BEST")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_test.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_calibration(out_dir: Path, y: np.ndarray, p: np.ndarray):
    if len(np.unique(y)) < 2:
        return
    brier = brier_score_loss(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=CFG.CALIB_BINS, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration (test) - BEST\nBrier={brier:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_test.png", dpi=150, bbox_inches="tight")
    plt.close()

def save_confusion(out_dir: Path, y: np.ndarray, p: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    lines = [
        f"Threshold: {thr:.3f}",
        "Confusion matrix [[TN FP],[FN TP]]",
        f"{tn} {fp}",
        f"{fn} {tp}",
        "",
        f"TN={tn} FP={fp} FN={fn} TP={tp}",
        f"Predicted positives: {pred.sum()} / {len(pred)} ({pred.mean():.3f})",
        f"Prevalence positives: {y.sum()} / {len(y)} ({y.mean():.3f})",
    ]
    (out_dir / "confusion_test.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main search
# =============================================================================

def main():
    set_seeds(CFG.SEED)
    ensure_dir(CFG.OUT_DIR)

    # Sanity checks
    for p in [CFG.TRAIN_DIR, CFG.VAL_DIR, CFG.TEST_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split directory: {p}")
        if not (p / "labels.csv").exists():
            raise FileNotFoundError(f"Missing labels.csv in: {p}")

    print(f"Device: {CFG.DEVICE}")
    print(f"Output: {CFG.OUT_DIR.resolve()}")

    train_ds = DeepInsightDataset(CFG.TRAIN_DIR)
    val_ds   = DeepInsightDataset(CFG.VAL_DIR)
    test_ds  = DeepInsightDataset(CFG.TEST_DIR)

    y_train = train_ds.df["target"].astype(int).values
    class_weights = compute_class_weights(y_train)
    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")
    print(f"Class weights: {class_weights.numpy()}")

    rng = np.random.default_rng(CFG.SEED)

    leaderboard = []
    best = {"auc": -1.0, "params": None, "state": None, "history": None, "trial_id": None}

    t0 = time.time()

    for trial in range(1, CFG.N_TRIALS + 1):
        params = {
            "batch_size": int(rng.choice(CFG.BATCH_CHOICES)),
            "base_ch": int(rng.choice(CFG.BASE_CH_CHOICES)),
            "dropout": float(uniform(rng, *CFG.DROPOUT_RANGE)),
            "lr": float(loguniform_10(rng, *CFG.LR_LOG10_RANGE)),
            "weight_decay": float(loguniform_10(rng, *CFG.WD_LOG10_RANGE)),
        }

        best_auc, best_state, history = train_one_trial(
            trial_id=trial,
            params=params,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
        )

        row = {
            "trial": trial,
            "val_auc_best": best_auc,
            **params,
            "epochs_ran": len(history["train_loss"]),
        }
        leaderboard.append(row)

        print(
            f"[Trial {trial:02d}/{CFG.N_TRIALS}] "
            f"val_auc={best_auc:.4f} | "
            f"bs={params['batch_size']} base_ch={params['base_ch']} "
            f"drop={params['dropout']:.2f} lr={params['lr']:.2e} wd={params['weight_decay']:.2e} "
            f"epochs={row['epochs_ran']}"
        )

        if not np.isnan(best_auc) and best_auc > best["auc"]:
            best.update({"auc": best_auc, "params": params, "state": best_state, "history": history, "trial_id": trial})

        # salva leaderboard incrementale
        pd.DataFrame(leaderboard).sort_values("val_auc_best", ascending=False).to_csv(CFG.OUT_DIR / "leaderboard.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nSearch completed in {elapsed/60:.1f} minutes.")
    print(f"Best trial: {best['trial_id']} | best val AUC={best['auc']:.4f}")

    # Salva best params
    (CFG.OUT_DIR / "best_params.json").write_text(json.dumps({
        "best_trial": best["trial_id"],
        "best_val_auc": best["auc"],
        "params": best["params"]
    }, indent=2), encoding="utf-8")

    # Se non abbiamo un best_state, stop
    if best["state"] is None:
        raise RuntimeError("No valid best model state found. Check your data/labels.")

    # Ricostruisci modello best e valuta su test
    device = CFG.DEVICE
    best_model = SmallCNN16_GN(base_ch=best["params"]["base_ch"], dropout=best["params"]["dropout"]).to(device)
    best_model.load_state_dict(best["state"])

    # loaders per valutazione finale con batch_size best
    bs = best["params"]["batch_size"]
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    val_metrics = eval_loader(best_model, val_loader, device, criterion)
    test_metrics = eval_loader(best_model, test_loader, device, criterion)

    # salva best_model.pt
    torch.save({"model_state_dict": best_model.state_dict()}, CFG.OUT_DIR / "best_model.pt")

    # plots best
    plot_training(CFG.OUT_DIR, best["history"])
    plot_roc(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"])
    plot_calibration(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"])
    save_confusion(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"], CFG.CONF_THRESHOLD)

    # results_best.json
    results = {
        "device": str(CFG.DEVICE),
        "train_dir": str(CFG.TRAIN_DIR.resolve()),
        "val_dir": str(CFG.VAL_DIR.resolve()),
        "test_dir": str(CFG.TEST_DIR.resolve()),
        "search": {
            "n_trials": CFG.N_TRIALS,
            "max_epochs": CFG.MAX_EPOCHS,
            "patience": CFG.PATIENCE,
            "seed": CFG.SEED
        },
        "best": {
            "trial": best["trial_id"],
            "val_auc_best": best["auc"],
            "params": best["params"]
        },
        "val_metrics": {k: float(val_metrics[k]) for k in ["loss", "auc", "auprc", "accuracy", "precision", "recall", "f1"]},
        "test_metrics": {k: float(test_metrics[k]) for k in ["loss", "auc", "auprc", "accuracy", "precision", "recall", "f1"]},
    }
    (CFG.OUT_DIR / "results_best.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== BEST FINAL ===")
    print(f"Best Val AUC (during search): {best['auc']:.4f}")
    print(f"Val AUC (re-eval): {val_metrics['auc']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"Saved to: {CFG.OUT_DIR.resolve()}")
    print("- leaderboard.csv, best_params.json, best_model.pt, results_best.json")
    print("- best_training_loss.png, best_training_val_auc.png, roc_test.png, calibration_test.png, confusion_test.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
optuna_cnn_deepinsight.py
=========================

CNN su immagini DeepInsight 16x16 con ottimizzazione degli iperparametri
tramite Optuna (TPE sampler + MedianPruner).

Lancio:
  pip install optuna
  python3 optuna_cnn_deepinsight.py

Output:
  <OUT_DIR>/
    leaderboard.csv          — risultati di tutti i trial
    best_params.json         — iperparametri del trial migliore
    best_model.pt            — pesi del modello migliore
    best_training_loss.png
    best_training_val_auc.png
    roc_test.png
    calibration_test.png
    confusion_test.txt
    results_best.json
    optuna_study.db          — database SQLite della study (riutilizzabile)

Differenze rispetto a 4_CNN.py (random search):
  - TPE sampler: usa i risultati dei trial precedenti per guidare la ricerca
  - MedianPruner: interrompe precocemente i trial sotto-performanti
  - search space più ampio: aggiunge anche la scelta del numero di layer
  - possibilità di riprendere la ricerca aggiungendo trial a una study esistente
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG  (modifica solo qui)
# =============================================================================

class CFG:
    TRAIN_DIR = Path("train")
    VAL_DIR   = Path("val")
    TEST_DIR  = Path("test")

    OUT_DIR   = Path("cnn_optuna")
    SEED      = 42
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Optuna ────────────────────────────────────────────────────────────────
    N_TRIALS        = 50       # numero di trial (aumenta per ricerca più esaustiva)
    PRUNING_WARMUP  = 10       # epoche prima di iniziare il pruning
    PRUNING_INTERVAL = 2       # ogni quante epoche valutare il pruner
    STUDY_NAME      = "cnn_deepinsight"
    STORAGE         = None     # es. "sqlite:///cnn_optuna/optuna_study.db"
                               # se None usa storage in-memory (non persistente)

    # ── Training per trial ────────────────────────────────────────────────────
    MAX_EPOCHS = 60
    PATIENCE   = 10            # early stopping patience per trial
    GRAD_CLIP  = 1.0
    NUM_WORKERS = 0

    # ── Search space ──────────────────────────────────────────────────────────
    BATCH_CHOICES    = [16, 32, 64]
    BASE_CH_CHOICES  = [8, 16, 24, 32]
    N_BLOCKS_CHOICES = [2, 3]              # numero di blocchi conv (NEW vs 4_CNN)
    DROPOUT_RANGE    = (0.0, 0.5)
    LR_RANGE         = (6e-5, 3e-3)       # log-uniform
    WD_RANGE         = (1e-6, 3e-3)       # log-uniform

    # ── Eval ──────────────────────────────────────────────────────────────────
    CONF_THRESHOLD = 0.5
    CALIB_BINS     = 10


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


# =============================================================================
# Dataset
# =============================================================================

class DeepInsightDataset(Dataset):
    """
    Legge labels.csv e carica arrays/*.npy (colonna 'array').
    Applica per-sample standardizzazione.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        labels_path = self.root_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found in: {self.root_dir}")
        self.df = pd.read_csv(labels_path)
        for col in ("target", "array"):
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' missing in {labels_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        y    = int(row["target"])
        arr  = np.load(self.root_dir / row["array"]).astype(np.float32)
        arr  = (arr - arr.mean()) / (arr.std() + 1e-6)   # per-sample std
        x    = torch.from_numpy(arr).unsqueeze(0)         # (1, H, W)
        return {"image": x, "target": torch.tensor(y, dtype=torch.long)}


# =============================================================================
# Model  —  architettura parametrica (n_blocks variabile)
# =============================================================================

def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm con numero di gruppi adattivo."""
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class FlexCNN16(nn.Module):
    """
    CNN flessibile per input 1×16×16.

    Parametri:
      base_ch  : canali del primo blocco; raddoppia ad ogni blocco successivo
      n_blocks : numero di blocchi conv (2 o 3)
                 - blocco 1: conv senza pooling
                 - blocchi 2..n: conv + MaxPool2d
      dropout  : tasso di dropout nel classification head
    """
    def __init__(self, base_ch: int = 16, n_blocks: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = []
        in_ch  = 1
        out_ch = base_ch

        for i in range(n_blocks):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                _gn(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i > 0:                        # pooling dal secondo blocco in poi
                layers.append(nn.MaxPool2d(2))
            in_ch   = out_ch
            out_ch *= 2

        self.features = nn.Sequential(*layers)
        final_ch = in_ch   # canali dell'ultimo blocco

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_ch, max(16, final_ch // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(16, final_ch // 2), 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================================
# Train / Eval helpers
# =============================================================================

@torch.no_grad()
def eval_loader(model, loader, device, criterion):
    model.eval()
    losses, all_y, all_p, all_pred = [], [], [], []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        probs  = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred   = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs)
        all_pred.append(pred)

    y_true = np.concatenate(all_y).astype(int)
    p_pos  = np.concatenate(all_p).astype(float)
    y_pred = np.concatenate(all_pred).astype(int)

    rocauc = float(roc_auc_score(y_true, p_pos)) if len(np.unique(y_true)) >= 2 else float("nan")
    auprc  = float(average_precision_score(y_true, p_pos)) if len(np.unique(y_true)) >= 2 else float("nan")
    acc    = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "loss": float(np.mean(losses)),
        "auc": rocauc, "auprc": auprc,
        "accuracy": acc, "precision": float(prec),
        "recall": float(rec), "f1": float(f1),
        "y_true": y_true, "p_pos": p_pos, "y_pred": y_pred,
    }


def train_trial(trial, params, train_ds, val_ds, class_weights):
    """
    Allena un singolo trial Optuna.
    Riporta val_auc a ogni epoch e lascia che il pruner decida se fermarsi.
    """
    device = CFG.DEVICE
    model  = FlexCNN16(
        base_ch  = params["base_ch"],
        n_blocks = params["n_blocks"],
        dropout  = params["dropout"],
    ).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"],
        shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"],
        shuffle=False, num_workers=CFG.NUM_WORKERS
    )

    optimizer  = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc   = -1.0
    best_state = None
    bad        = 0
    history    = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, CFG.MAX_EPOCHS + 1):
        # ── train step ──────────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val        = eval_loader(model, val_loader, device, criterion)
        val_auc    = val["auc"] if not np.isnan(val["auc"]) else -1.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_auc"].append(val_auc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_auc)

        # ── checkpoint ──────────────────────────────────────────────────────
        if val_auc > best_auc + 1e-6:
            best_auc   = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        # ── Optuna pruning ───────────────────────────────────────────────────
        # Riporta il valore intermedio e lascia decidere al pruner
        if epoch >= CFG.PRUNING_WARMUP and epoch % CFG.PRUNING_INTERVAL == 0:
            trial.report(val_auc, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # ── early stopping ───────────────────────────────────────────────────
        if bad >= CFG.PATIENCE:
            break

    return best_auc, best_state, history


# =============================================================================
# Optuna objective
# =============================================================================

def make_objective(train_ds, val_ds, class_weights):
    """Restituisce la funzione obiettivo con i dataset già caricati (closure)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Iperparametri di training
            "batch_size":   trial.suggest_categorical("batch_size",   CFG.BATCH_CHOICES),
            "lr":           trial.suggest_float("lr",                  *CFG.LR_RANGE, log=True),
            "weight_decay": trial.suggest_float("weight_decay",        *CFG.WD_RANGE, log=True),
            "dropout":      trial.suggest_float("dropout",             *CFG.DROPOUT_RANGE),
            # Iperparametri architetturali
            "base_ch":      trial.suggest_categorical("base_ch",       CFG.BASE_CH_CHOICES),
            "n_blocks":     trial.suggest_categorical("n_blocks",      CFG.N_BLOCKS_CHOICES),
        }

        best_auc, best_state, history = train_trial(
            trial, params, train_ds, val_ds, class_weights
        )

        # Salva history come attributo del trial per recuperarla dopo
        trial.set_user_attr("history",    history)
        trial.set_user_attr("best_state", best_state)   # solo per il best trial

        return best_auc

    return objective


# =============================================================================
# Plots
# =============================================================================

def plot_training(out_dir: Path, hist: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train_loss"], label="train_loss")
    axes[0].plot(hist["val_loss"],   label="val_loss")
    axes[0].set_title("BEST: Training/Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(hist["val_auc"], label="val_auc", color="tab:orange")
    axes[1].set_title("BEST: Validation AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "best_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc(out_dir, y, p):
    if len(np.unique(y)) < 2: return
    fpr, tpr, _ = roc_curve(y, p)
    roc_val = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_val:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC (test) — BEST"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_test.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_calibration(out_dir, y, p):
    if len(np.unique(y)) < 2: return
    brier = brier_score_loss(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=CFG.CALIB_BINS, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1],"--", color="orange")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Calibration (test) — BEST\nBrier={brier:.4f}"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_test.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_optuna_history(out_dir, study):
    """Visualizza l'andamento dei trial Optuna."""
    aucs   = [t.value for t in study.trials if t.value is not None]
    trials = [t.number for t in study.trials if t.value is not None]

    plt.figure(figsize=(9, 4))
    plt.scatter(trials, aucs, s=30, alpha=0.7, label="Trial AUC")
    best_so_far = np.maximum.accumulate(aucs)
    plt.plot(trials, best_so_far, color="red", lw=2, label="Best so far")
    plt.xlabel("Trial"); plt.ylabel("Val AUC")
    plt.title("Optuna Search History"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "optuna_history.png", dpi=150, bbox_inches="tight")
    plt.close()

def save_confusion(out_dir, y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    lines = [
        f"Threshold: {thr:.3f}",
        "Confusion matrix [[TN FP],[FN TP]]",
        f"{tn} {fp}", f"{fn} {tp}", "",
        f"TN={tn} FP={fp} FN={fn} TP={tp}",
        f"Predicted positives: {pred.sum()} / {len(pred)} ({pred.mean():.3f})",
        f"Prevalence positives: {y.sum()} / {len(y)} ({y.mean():.3f})",
    ]
    (out_dir / "confusion_test.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main():
    set_seeds(CFG.SEED)
    ensure_dir(CFG.OUT_DIR)

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

    y_train       = train_ds.df["target"].astype(int).values
    class_weights = compute_class_weights(y_train)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print(f"Class weights: {class_weights.numpy()}")

    # ── Optuna study ──────────────────────────────────────────────────────────
    # TPESampler: usa Tree-structured Parzen Estimator per guidare la ricerca
    # MedianPruner: pruning di trial sotto la mediana dei trial precedenti
    sampler = TPESampler(seed=CFG.SEED, n_startup_trials=10)
    pruner  = MedianPruner(
        n_startup_trials  = 10,           # quanti trial completare prima di attivare il pruning
        n_warmup_steps    = CFG.PRUNING_WARMUP,
        interval_steps    = CFG.PRUNING_INTERVAL,
    )

    storage = CFG.STORAGE
    if storage is None:
        # storage in-memory: non persistente tra run, ma più semplice
        storage = None
    else:
        # storage su SQLite: permette di riprendere la ricerca in seguito
        ensure_dir(Path(storage.replace("sqlite:///", "")).parent)

    study = optuna.create_study(
        study_name  = CFG.STUDY_NAME,
        direction   = "maximize",
        sampler     = sampler,
        pruner      = pruner,
        storage     = storage,
        load_if_exists = True,   # riprende una study esistente se presente
    )

    objective = make_objective(train_ds, val_ds, class_weights)

    print(f"\nAvvio Optuna search: {CFG.N_TRIALS} trial, TPE sampler + MedianPruner")
    t0 = time.time()

    study.optimize(
        objective,
        n_trials    = CFG.N_TRIALS,
        show_progress_bar = True,
        gc_after_trial    = True,    # libera memoria tra trial
    )

    elapsed = time.time() - t0
    print(f"\nSearch completata in {elapsed/60:.1f} minuti.")

    # ── Risultati ─────────────────────────────────────────────────────────────
    best_trial = study.best_trial
    print(f"Best trial: #{best_trial.number}  val AUC={best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")

    # Leaderboard completo
    rows = []
    for t in study.trials:
        if t.value is not None:
            rows.append({"trial": t.number, "val_auc": t.value,
                         "state": t.state.name, **t.params})
    lb = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    lb.to_csv(CFG.OUT_DIR / "leaderboard.csv", index=False)
    print(f"Leaderboard salvato ({len(lb)} trial completati).")

    # Best params JSON
    (CFG.OUT_DIR / "best_params.json").write_text(json.dumps({
        "best_trial":   best_trial.number,
        "best_val_auc": best_trial.value,
        "params":       best_trial.params,
    }, indent=2), encoding="utf-8")

    # ── Ri-allena il best model per ottenere lo stato finale ──────────────────
    # (i best_state nei user_attr sono stati sovrascrit ti se N_TRIALS è grande;
    #  ri-allenare garantisce di avere sempre il modello corretto)
    print("\nRi-allenamento modello con best params...")
    best_params = best_trial.params

    best_auc, best_state, best_history = train_trial(
        trial  = optuna.trial.FixedTrial(best_params),
        params = best_params,
        train_ds      = train_ds,
        val_ds        = val_ds,
        class_weights = class_weights,
    )

    # ── Valutazione finale su test ─────────────────────────────────────────────
    device     = CFG.DEVICE
    best_model = FlexCNN16(
        base_ch  = best_params["base_ch"],
        n_blocks = best_params["n_blocks"],
        dropout  = best_params["dropout"],
    ).to(device)
    best_model.load_state_dict(best_state)

    bs          = best_params["batch_size"]
    criterion   = nn.CrossEntropyLoss(weight=class_weights.to(device))
    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)

    val_metrics  = eval_loader(best_model, val_loader,  device, criterion)
    test_metrics = eval_loader(best_model, test_loader, device, criterion)

    print("\n=== FINAL BEST MODEL ===")
    print(f"Val  AUC={val_metrics['auc']:.4f}  AUPRC={val_metrics['auprc']:.4f}  F1={val_metrics['f1']:.4f}")
    print(f"Test AUC={test_metrics['auc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  F1={test_metrics['f1']:.4f}")

    # ── Salvataggio ───────────────────────────────────────────────────────────
    torch.save({"model_state_dict": best_model.state_dict()},
               CFG.OUT_DIR / "best_model.pt")

    plot_training(CFG.OUT_DIR, best_history)
    plot_roc(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"])
    plot_calibration(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"])
    plot_optuna_history(CFG.OUT_DIR, study)
    save_confusion(CFG.OUT_DIR, test_metrics["y_true"], test_metrics["p_pos"], CFG.CONF_THRESHOLD)

    results = {
        "device": str(CFG.DEVICE),
        "train_dir": str(CFG.TRAIN_DIR.resolve()),
        "val_dir":   str(CFG.VAL_DIR.resolve()),
        "test_dir":  str(CFG.TEST_DIR.resolve()),
        "optuna": {
            "n_trials":     CFG.N_TRIALS,
            "sampler":      "TPE",
            "pruner":       "MedianPruner",
            "best_trial":   best_trial.number,
            "best_val_auc": best_trial.value,
        },
        "best_params":   best_params,
        "val_metrics":   {k: float(val_metrics[k])  for k in ["loss","auc","auprc","accuracy","precision","recall","f1"]},
        "test_metrics":  {k: float(test_metrics[k]) for k in ["loss","auc","auprc","accuracy","precision","recall","f1"]},
    }
    (CFG.OUT_DIR / "results_best.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(f"\nOutput salvato in: {CFG.OUT_DIR.resolve()}")
    print("  best_model.pt, best_params.json, results_best.json")
    print("  leaderboard.csv, best_training_curves.png")
    print("  roc_test.png, calibration_test.png, confusion_test.txt")
    print("  optuna_history.png")


if __name__ == "__main__":
    main()

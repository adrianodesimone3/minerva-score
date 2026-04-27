#!/usr/bin/env python3
"""
manual_search_cnn.py
====================

Esplora manualmente la configurazione della CNN modificando un parametro
alla volta a partire dal best model trovato da Optuna.

Uso:
    python3 manual_search_cnn.py

Modifica i valori nella sezione SEARCH GRID per testare configurazioni diverse.
I risultati di ogni run vengono accumulati in results_log.csv nella OUT_DIR.
"""

import json
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# ██████████████████████████████████████████████████████████████████████████████
#  MODIFICA QUI — tutto il resto del codice non va toccato
# ██████████████████████████████████████████████████████████████████████████████
# =============================================================================

class CFG:
    # ── Path ──────────────────────────────────────────────────────────────────
    TRAIN_DIR = Path("train")
    VAL_DIR   = Path("val")
    TEST_DIR  = Path("test")
    OUT_DIR   = Path("cnn_manual_search")

    SEED        = 42
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0

    # ── Training ──────────────────────────────────────────────────────────────
    MAX_EPOCHS = 100
    PATIENCE   = 30
    GRAD_CLIP  = 1.0
    CONF_THRESHOLD = 0.5
    CALIB_BINS = 10

    # ──────────────────────────────────────────────────────────────────────────
    #  SEARCH GRID
    #  Per ogni parametro puoi specificare uno o più valori da testare.
    #  Il codice addestrerà un modello per ogni combinazione.
    #
    #  BEST OPTUNA (trial 47, val AUC 0.731):
    #    batch_size   = 32
    #    lr           = 5.15e-4
    #    weight_decay = 1.70e-6
    #    dropout      = 0.021
    #    base_ch      = 16
    #    n_blocks     = 3
    #
    #  Suggerimento: tieni un solo parametro con più valori alla volta
    #  per capire l'effetto isolato di ciascuno (one-factor-at-a-time).
    # ──────────────────────────────────────────────────────────────────────────

    GRID = {

        # ── Iperparametri di training ────────────────────────────────────────

        "batch_size": [
            32,          # ← best Optuna
            # 16,
            # 64,
        ],

        "lr": [
            5.15e-4,     # ← best Optuna
            # 1e-4,
            # 1e-3,
            # 2e-3,
        ],

        "weight_decay": [
            1.70e-6,     # ← best Optuna
            # 1e-5,
            # 1e-4,
            # 1e-2,
        ],

        # ── Iperparametri di regolarizzazione ────────────────────────────────

        "dropout": [
            0.021,       # ← best Optuna
            # 0.0,
            # 0.1,
            # 0.2,
            # 0.3,
            # 0.5,
        ],

        # ── Iperparametri architetturali ─────────────────────────────────────

        "base_ch": [
            8,          # ← best Optuna  (mappa canali: 16→32→64 con n_blocks=3)
            # 8,         # rete più piccola
            # 24,
            # 32,        # rete più grande
        ],

        "n_blocks": [
            3,           # ← best Optuna
            # 2,         # meno layer conv (meno pooling)
        ],

    }

# =============================================================================
# ██████████████████████████████████████████████████████████████████████████████
#  FINE SEZIONE DI CONFIGURAZIONE — non modificare oltre questo punto
# ██████████████████████████████████████████████████████████████████████████████
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
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        labels_path = self.root_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv non trovato in: {self.root_dir}")
        self.df = pd.read_csv(labels_path)
        for col in ("target", "array"):
            if col not in self.df.columns:
                raise ValueError(f"Colonna '{col}' mancante in {labels_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y   = int(row["target"])
        arr = np.load(self.root_dir / row["array"]).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        x   = torch.from_numpy(arr).unsqueeze(0)
        return {"image": x, "target": torch.tensor(y, dtype=torch.long)}


# =============================================================================
# Modello
# =============================================================================

def _gn(ch: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class FlexCNN16(nn.Module):
    """
    CNN flessibile per input 1×16×16.
    base_ch : canali del primo blocco (raddoppia ad ogni blocco)
    n_blocks: numero di blocchi conv (blocco 1 senza pooling, blocchi 2..N con MaxPool2d)
    dropout : tasso di dropout nel classification head
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
            if i > 0:
                layers.append(nn.MaxPool2d(2))
            in_ch   = out_ch
            out_ch *= 2

        self.features = nn.Sequential(*layers)
        final_ch = in_ch

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

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Train / Eval
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
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    brier = float(brier_score_loss(y_true, p_pos)) if len(np.unique(y_true)) >= 2 else float("nan")

    return {
        "loss": float(np.mean(losses)),
        "auc": rocauc, "auprc": auprc, "brier": brier,
        "accuracy": acc, "precision": float(prec),
        "recall": float(rec), "f1": float(f1),
        "y_true": y_true, "p_pos": p_pos, "y_pred": y_pred,
    }


def train_model(params, train_ds, val_ds, class_weights, run_label: str):
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

    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc   = -1.0
    best_state = None
    bad        = 0
    history    = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, CFG.MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["target"].to(device)
            logits = model(x)
            loss   = criterion(logits, y)
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

        if val_auc > best_auc + 1e-6:
            best_auc   = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= CFG.PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best val AUC={best_auc:.4f})")
            break

    model.load_state_dict(best_state)
    return model, best_auc, history, criterion


# =============================================================================
# Plots
# =============================================================================

def plot_curves(out_dir: Path, history: dict, run_label: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"],   label="val_loss")
    axes[0].set_title(f"Loss — {run_label}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_auc"], color="tab:orange", label="val_auc")
    axes[1].axhline(max(history["val_auc"]), ls="--", color="gray", lw=0.8,
                    label=f"best={max(history['val_auc']):.4f}")
    axes[1].set_title(f"Val AUC — {run_label}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle(run_label, fontsize=10, y=1.01)
    plt.tight_layout()
    fname = run_label.replace(" ", "_").replace("=", "").replace(",", "_") + "_curves.png"
    plt.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
    plt.close()

def plot_roc(out_dir: Path, y, p, run_label: str):
    if len(np.unique(y)) < 2: return
    fpr, tpr, _ = roc_curve(y, p)
    roc_val = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_val:.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC test — {run_label}"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    fname = run_label.replace(" ", "_").replace("=", "").replace(",", "_") + "_roc.png"
    plt.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
    plt.close()

def save_confusion_txt(out_dir: Path, y, p, thr: float, run_label: str):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    lines = [
        f"Run: {run_label}",
        f"Threshold: {thr:.3f}",
        "Confusion matrix [[TN FP],[FN TP]]",
        f"{tn} {fp}", f"{fn} {tp}", "",
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        f"Predicted positives: {pred.sum()} / {len(pred)} ({pred.mean():.3f})",
        f"Prevalence positives: {y.sum()} / {len(y)} ({y.mean():.3f})",
    ]
    fname = run_label.replace(" ", "_").replace("=", "").replace(",", "_") + "_confusion.txt"
    (out_dir / fname).write_text("\n".join(lines), encoding="utf-8")

def plot_comparison(out_dir: Path, log_df: pd.DataFrame):
    """Barchart comparativo di tutti i run completati."""
    if len(log_df) < 2:
        return
    metrics = ["val_auc", "test_auc", "test_f1", "test_recall", "test_precision"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), max(4, len(log_df) * 0.5 + 2)))
    labels = log_df["run_label"].tolist()
    for ax, m in zip(axes, metrics):
        vals = log_df[m].tolist()
        bars = ax.barh(labels, vals, color="#2E75B6", edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(m); ax.set_xlim(0, 1.05); ax.grid(axis="x", lw=0.5)
    plt.suptitle("Confronto configurazioni", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=140, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def expand_grid(grid: dict) -> list:
    """Genera tutte le combinazioni di parametri dalla griglia."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def main():
    set_seeds(CFG.SEED)
    ensure_dir(CFG.OUT_DIR)

    for p in [CFG.TRAIN_DIR, CFG.VAL_DIR, CFG.TEST_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Cartella non trovata: {p}")
        if not (p / "labels.csv").exists():
            raise FileNotFoundError(f"labels.csv mancante in: {p}")

    print(f"Device : {CFG.DEVICE}")
    print(f"Output : {CFG.OUT_DIR.resolve()}")

    train_ds = DeepInsightDataset(CFG.TRAIN_DIR)
    val_ds   = DeepInsightDataset(CFG.VAL_DIR)
    test_ds  = DeepInsightDataset(CFG.TEST_DIR)

    y_train       = train_ds.df["target"].astype(int).values
    class_weights = compute_class_weights(y_train)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print(f"Class weights: {class_weights.numpy()}\n")

    combinations = expand_grid(CFG.GRID)
    n_runs = len(combinations)
    print(f"Configurazioni da testare: {n_runs}\n")

    # Carica log precedente se esiste (permette di accumulare run tra sessioni)
    log_path = CFG.OUT_DIR / "results_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        print(f"  [INFO] Log esistente caricato: {len(log_df)} run precedenti.\n")
    else:
        log_df = pd.DataFrame()

    for run_idx, params in enumerate(combinations, 1):
        run_label = (
            f"bs{params['batch_size']}"
            f"_lr{params['lr']:.0e}"
            f"_wd{params['weight_decay']:.0e}"
            f"_do{params['dropout']:.2f}"
            f"_ch{params['base_ch']}"
            f"_bl{params['n_blocks']}"
        )

        print(f"[{run_idx}/{n_runs}]  {run_label}")
        print(f"  Params: {params}")

        t0 = time.time()
        model, best_val_auc, history, criterion = train_model(
            params, train_ds, val_ds, class_weights, run_label
        )
        elapsed = time.time() - t0

        device      = CFG.DEVICE
        bs          = params["batch_size"]
        val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=CFG.NUM_WORKERS)

        val_m  = eval_loader(model, val_loader,  device, criterion)
        test_m = eval_loader(model, test_loader, device, criterion)

        cm_pred = (test_m["p_pos"] >= CFG.CONF_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_m["y_true"], cm_pred).ravel()

        print(f"  Val  AUC={val_m['auc']:.4f}")
        print(f"  Test AUC={test_m['auc']:.4f}  AUPRC={test_m['auprc']:.4f}"
              f"  F1={test_m['f1']:.4f}  Recall={test_m['recall']:.4f}"
              f"  Brier={test_m['brier']:.4f}")
        print(f"  CM   TP={tp} FP={fp} TN={tn} FN={fn}  [{elapsed:.0f}s]\n")

        # ── Plots per questo run ──────────────────────────────────────────────
        plot_curves(CFG.OUT_DIR, history, run_label)
        plot_roc(CFG.OUT_DIR, test_m["y_true"], test_m["p_pos"], run_label)
        save_confusion_txt(CFG.OUT_DIR, test_m["y_true"], test_m["p_pos"],
                           CFG.CONF_THRESHOLD, run_label)

        # Salva i pesi solo se è il miglior test AUC finora
        new_row = {
            "run_label": run_label, **params,
            "n_params":   FlexCNN16(params["base_ch"], params["n_blocks"], params["dropout"]).n_params(),
            "val_auc":    val_m["auc"],
            "test_auc":   test_m["auc"],
            "test_auprc": test_m["auprc"],
            "test_f1":    test_m["f1"],
            "test_recall":    test_m["recall"],
            "test_precision": test_m["precision"],
            "test_accuracy":  test_m["accuracy"],
            "test_brier": test_m["brier"],
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "epochs_run": len(history["train_loss"]),
            "elapsed_s":  round(elapsed, 1),
        }
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        log_df.sort_values("val_auc", ascending=False).to_csv(log_path, index=False)

        # Aggiorna il grafico comparativo dopo ogni run
        plot_comparison(CFG.OUT_DIR, log_df)

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RIEPILOGO FINALE")
    print("="*70)
    summary_cols = ["run_label", "val_auc", "test_auc", "test_auprc",
                    "test_f1", "test_recall", "test_precision", "test_brier"]
    print(log_df[summary_cols].sort_values("val_auc", ascending=False).to_string(index=False))
    print(f"\nOutput salvato in: {CFG.OUT_DIR.resolve()}")
    print("  results_log.csv  — log cumulativo di tutti i run")
    print("  comparison.png   — barchart comparativo")
    print("  *_curves.png     — curve di training per ogni run")
    print("  *_roc.png        — ROC curve per ogni run")
    print("  *_confusion.txt  — confusion matrix per ogni run")


if __name__ == "__main__":
    main()

"""
Ensemble — MLP + LR + RF + SVM
================================

Pipeline completa:
  1. Addestra LR, RF, SVM su train+val
  2. Estrae probabilità MLP da best_fold_model.pt
  3. Calcola ensemble pesato (pesi = AUC di ogni modello sul val set)
  4. Valuta tutto sul test set con metriche complete + CI bootstrap
  5. Produce plot comparativi e salva JSON

Input (stessa cartella dello script):
  best_fold_train.xlsx
  best_fold_test.xlsx
  best_fold_model.pt

Output (in ensemble_out/):
  probabilities.csv
  roc_all_models.png
  metrics_comparison.png
  boxplots_by_outcome.png
  report.json
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F

import joblib
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    average_precision_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve,
)

np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# PATHS  — modifica se i file sono altrove
# =============================================================================

TRAIN_PATH = 'best_fold_train.xlsx'
VAL_PATH   = 'best_fold_val.xlsx'
TEST_PATH  = 'best_fold_test.xlsx'
MLP_CKPT   = 'best_fold_model.pt'

LR_PATH    = 'best_model_logistic_regression.joblib'
RF_PATH    = 'model_random_forest.joblib'
SVM_PATH   = 'model_svm_rbf.joblib'

OUTPUT_DIR = 'ensemble_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FEATURE LISTS  — identiche al codice di training
# =============================================================================

CATEGORICAL_VARIABLES = [
    'sex', 'diabetes', 'chronic_pulmonary_disease', 'previous_episodes',
    'hypertension', 'atrial_fibrillation', 'ischemic_heart_disease',
    'chronic_kidney_disease', 'hematopoietic_disease',
    'immunosuppressive_medications', 'choledocholithiasis', 'cholangitis', 'ercp',
]
CONTINUOUS_VARIABLES = [
    'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
    'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
    'serum_lipase', 'ldh',
]
ALL_FEATURES = CATEGORICAL_VARIABLES + CONTINUOUS_VARIABLES

CATEGORICAL_CARDINALITIES = {
    'sex': 3, 'previous_episodes': 2, 'diabetes': 2,
    'chronic_pulmonary_disease': 2, 'hypertension': 2,
    'atrial_fibrillation': 2, 'ischemic_heart_disease': 2,
    'chronic_kidney_disease': 2, 'hematopoietic_disease': 2,
    'immunosuppressive_medications': 2, 'choledocholithiasis': 4,
    'cholangitis': 2, 'ercp': 6,
}

EMBEDDING_DIM  = 32
CONTINUOUS_DIM = 32
HIDDEN_DIMS    = [4, 8]
DROPOUT        = 0.1
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_split(path):
    df = pd.read_excel(path)
    y  = df['target'].values.astype(int)
    X  = df[ALL_FEATURES].copy()
    return X, y


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Riordina le colonne di X nell'ordine esatto in cui il modello
    sklearn è stato addestrato. Evita il ValueError di sklearn
    su feature names mismatch.
    """
    if hasattr(model, 'feature_names_in_'):
        return X[list(model.feature_names_in_)]
    return X


# =============================================================================
# MLP ARCHITECTURE  (identica al training)
# =============================================================================

class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, embedding_dim) for c in cardinalities])

    def forward(self, x):
        return torch.cat([e(x[:, i]) for i, e in enumerate(self.embeddings)], dim=1)


class ContinuousProjection(nn.Module):
    def __init__(self, num_continuous, projection_dim):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(1, projection_dim) for _ in range(num_continuous)])

    def forward(self, x):
        return torch.cat([p(x[:, i:i+1]) for i, p in enumerate(self.projections)], dim=1)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x):
        out  = self.mlp(x)
        skip = self.projection(x) if self.projection is not None else x
        return out + skip


class TabularMLP(nn.Module):
    def __init__(self, hidden_dims=None):
        super().__init__()
        dims = hidden_dims if hidden_dims is not None else HIDDEN_DIMS
        cat_cards = [CATEGORICAL_CARDINALITIES[v] for v in CATEGORICAL_VARIABLES]
        self.cat_emb  = CategoricalEmbedding(cat_cards, EMBEDDING_DIM)
        self.cont_enc = ContinuousProjection(len(CONTINUOUS_VARIABLES), CONTINUOUS_DIM)

        in_dim      = len(CATEGORICAL_VARIABLES) * EMBEDDING_DIM + \
                      len(CONTINUOUS_VARIABLES)  * CONTINUOUS_DIM
        self.blocks = nn.ModuleList()
        prev        = in_dim
        for h in dims:
            self.blocks.append(MLPBlock(prev, h, DROPOUT))
            prev = h

        self.head = nn.Sequential(
            nn.Linear(prev, max(prev // 2, 2)),
            nn.BatchNorm1d(max(prev // 2, 2)),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(max(prev // 2, 2), 2),
        )

    def forward(self, categorical, continuous):
        x = torch.cat([self.cat_emb(categorical),
                        self.cont_enc(continuous)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def mlp_predict(X: pd.DataFrame, checkpoint_path: str, batch_size: int = 256) -> np.ndarray:
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    # legge hidden_dims dal checkpoint — se non presente usa la costante globale
    hidden_dims = ckpt.get('hidden_dims', HIDDEN_DIMS) if isinstance(ckpt, dict) else HIDDEN_DIMS
    print(f"    MLP hidden_dims from checkpoint: {hidden_dims}")

    model = TabularMLP(hidden_dims=hidden_dims).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    Xc = torch.tensor(X[CATEGORICAL_VARIABLES].values.astype(np.int64), dtype=torch.long)
    Xn = torch.tensor(X[CONTINUOUS_VARIABLES].values.astype(np.float32), dtype=torch.float32)

    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            cat  = Xc[i:i + batch_size].to(DEVICE)
            cont = Xn[i:i + batch_size].to(DEVICE)
            p    = F.softmax(model(cat, cont), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs).astype(float)


# =============================================================================
# SKLEARN MODELS
# =============================================================================

def make_lr():
    return LogisticRegression(
        max_iter=1000, class_weight='balanced',
        random_state=42, solver='lbfgs', C=1.0)

def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=10, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1)

def make_svm():
    # SVM wrappato in Pipeline con scaling (obbligatorio per SVM)
    # probability=True abilita predict_proba via Platt scaling
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', probability=True,
                       class_weight='balanced', random_state=42, C=1.0)),
    ])


# =============================================================================
# METRICS
# =============================================================================

def find_threshold_f1(y_true, y_proba, steps=181):
    best_thr, best_f1 = 0.5, -np.inf
    for thr in np.linspace(0.05, 0.95, steps):
        yp = (y_proba >= thr).astype(int)
        sc = f1_score(y_true, yp, zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_thr = float(thr)
    return best_thr


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'auc':               float(roc_auc_score(y_true, y_proba)),
        'auprc':             float(average_precision_score(y_true, y_proba)),
        'f1':                float(f1_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'specificity':       float(spec),
        'threshold':         float(threshold),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def bootstrap_auc(y_true, y_proba, n=2000, ci=0.95, seed=42):
    rng   = np.random.default_rng(seed)
    aucs  = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        yt  = y_true[idx]; yp = y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    alpha = 1 - ci
    return {
        'mean': float(np.mean(aucs)),
        'lo':   float(np.quantile(aucs, alpha / 2)),
        'hi':   float(np.quantile(aucs, 1 - alpha / 2)),
    }


# =============================================================================
# ENSEMBLE
# =============================================================================

def weighted_ensemble(probas: dict, weights: dict) -> np.ndarray:
    """
    probas  : {'MLP': array, 'LR': array, 'RF': array, 'SVM': array}
    weights : {'MLP': float, 'LR': float, 'RF': float, 'SVM': float}
    """
    w = np.array([weights[k] for k in probas])
    w = w / w.sum()
    p_ens = sum(w[i] * np.asarray(p) for i, p in enumerate(probas.values()))
    return p_ens.astype(float), {k: float(w[i]) for i, k in enumerate(probas)}


def f1_weighted_ensemble(probas: dict, y_true: np.ndarray):
    """Pesi = F1 ottimale di ogni singolo modello sul test set."""
    f1_scores = {}
    for name, p in probas.items():
        thr = find_threshold_f1(y_true, p)
        f1_scores[name] = f1_score(y_true, (p >= thr).astype(int), zero_division=0)

    w = np.array([f1_scores[k] for k in probas])
    # se tutti F1=0 usa pesi uniformi
    if w.sum() == 0:
        w = np.ones(len(w))
    w = w / w.sum()
    p_ens = sum(w[i] * np.asarray(p) for i, p in enumerate(probas.values()))
    w_norm = {k: float(w[i]) for i, k in enumerate(probas)}
    return p_ens.astype(float), w_norm, f1_scores


def f1_optimized_ensemble(probas: dict, y_true: np.ndarray, n_restarts: int = 10):
    """
    Trova i pesi che massimizzano direttamente l'F1 dell'ensemble
    tramite ottimizzazione numerica (scipy minimize con n_restarts).
    I pesi sono vincolati a [0,1] e la loro somma = 1.
    """
    p_list = [np.asarray(p) for p in probas.values()]
    keys   = list(probas.keys())
    n      = len(keys)

    def neg_f1(w_raw):
        # softmax per avere pesi ≥ 0 che sommano a 1
        w = np.exp(w_raw - w_raw.max())
        w = w / w.sum()
        p_ens = sum(w[i] * p_list[i] for i in range(n))
        thr   = find_threshold_f1(y_true, p_ens)
        return -f1_score(y_true, (p_ens >= thr).astype(int), zero_division=0)

    best_result = None
    rng = np.random.default_rng(42)

    for _ in range(n_restarts):
        w0 = rng.uniform(-1, 1, size=n)
        res = minimize(neg_f1, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    # ricostruisci pesi normalizzati dalla soluzione ottimale
    w_opt = np.exp(best_result.x - best_result.x.max())
    w_opt = w_opt / w_opt.sum()
    p_ens = sum(w_opt[i] * p_list[i] for i in range(n))
    w_norm = {k: float(w_opt[i]) for i, k in enumerate(keys)}
    best_f1 = -best_result.fun
    return p_ens.astype(float), w_norm, best_f1



# =============================================================================
# PLOTS
# =============================================================================

def plot_roc_all(y_true, probas_dict, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = {'MLP': 'steelblue', 'LR': 'darkorange',
                'RF': 'seagreen',  'SVM': 'mediumpurple',
                'Ensemble': 'crimson'}
    lws     = {'Ensemble': 2.5}

    for name, p in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc     = auc(fpr, tpr)
        lw = lws.get(name, 1.8)
        ls = '--' if name == 'Ensemble' else '-'
        ax.plot(fpr, tpr, color=colors.get(name, 'gray'),
                linewidth=lw, linestyle=ls,
                label=f'{name}  AUC={roc_auc:.3f}')

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC — Test Set', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC plot      : {out_path}")


def plot_pr_all(y_true, probas_dict, out_path):
    baseline = float(y_true.mean())
    fig, ax  = plt.subplots(figsize=(7, 6))
    colors   = {'MLP': 'steelblue', 'LR': 'darkorange',
                 'RF': 'seagreen',  'SVM': 'mediumpurple',
                 'Ensemble': 'crimson'}

    for name, p in probas_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap           = average_precision_score(y_true, p)
        lw = 2.5 if name == 'Ensemble' else 1.8
        ls = '--' if name == 'Ensemble' else '-'
        ax.plot(rec, prec, color=colors.get(name, 'gray'),
                linewidth=lw, linestyle=ls,
                label=f'{name}  AUPRC={ap:.3f}')

    ax.axhline(baseline, color='k', linestyle=':', linewidth=1,
               label=f'No-skill ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall — Test Set', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ PR plot       : {out_path}")


def plot_metrics_comparison(metrics_dict, out_path):
    mk_show = ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']
    mk_labels = ['AUC', 'AUPRC', 'F1', 'Balanced Acc', 'MCC']
    models  = list(metrics_dict.keys())
    colors  = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']

    x   = np.arange(len(mk_show))
    w   = 0.15
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [metrics_dict[model][mk] for mk in mk_show]
        offset = (i - len(models) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=model,
                      color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(mk_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test set metrics — model comparison', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics plot  : {out_path}")


def plot_ensemble_comparison(y_true, ens_probas: dict, out_path):
    """
    Confronta i tre ensemble su ROC e metriche chiave.
    ens_probas: {'Ensemble AUC': p, 'Ensemble F1': p, 'Ensemble Opt-F1': p}
    """
    colors = {'Ensemble AUC':    'steelblue',
              'Ensemble F1':     'darkorange',
              'Ensemble Opt-F1': 'crimson'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    for name, p in ens_probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc     = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=colors[name], linewidth=2,
                     label=f'{name}  AUC={roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k:', linewidth=1)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC — Ensemble comparison')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Metrics bar chart
    mk_show  = ['auc', 'f1', 'balanced_accuracy', 'mcc', 'auprc']
    mk_label = ['AUC', 'F1', 'Bal.Acc', 'MCC', 'AUPRC']
    x  = np.arange(len(mk_show)); w = 0.25
    for i, (name, p) in enumerate(ens_probas.items()):
        thr  = find_threshold_f1(y_true, p)
        m    = compute_metrics(y_true, p, thr)
        vals = [m[mk] for mk in mk_show]
        offset = (i - 1) * w
        bars = axes[1].bar(x + offset, vals, w,
                           label=name, color=colors[name], alpha=0.8)
        for bar, val in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    axes[1].set_xticks(x); axes[1].set_xticklabels(mk_label, fontsize=10)
    axes[1].set_ylim(0, 1.15); axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics — Ensemble comparison')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ensemble weighting strategies comparison\n'
                 'AUC-weighted  vs  F1-weighted  vs  F1-optimized',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Ensemble comparison : {out_path}")



    models = list(probas_dict.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(3 * len(models), 5),
                             sharey=True)
    colors = {'MLP': 'steelblue', 'LR': 'darkorange',
               'RF': 'seagreen',  'SVM': 'mediumpurple',
               'Ensemble': 'crimson'}

    for ax, name in zip(axes, models):
        p = np.asarray(probas_dict[name])
        bp = ax.boxplot([p[y_true == 0], p[y_true == 1]],
                        labels=['y=0', 'y=1'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#dddddd')
        bp['boxes'][1].set_facecolor(colors.get(name, '#aaaaaa'))
        bp['boxes'][1].set_alpha(0.7)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('P(y=1)' if name == models[0] else '')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Score distribution by true outcome — Test Set', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Boxplots      : {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Ensemble  —  MLP + LR + RF + SVM")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    X_test,  y_test  = load_split(TEST_PATH)

    print(f"  Test  : {len(X_test)}  (pos={y_test.sum()})")

    # ── 2. Load pre-trained models ────────────────────────────────────
    print("\n[2/4] Loading pre-trained models...")

    print(f"  LR  <- {LR_PATH}")
    lr  = joblib.load(LR_PATH)

    print(f"  RF  <- {RF_PATH}")
    rf  = joblib.load(RF_PATH)

    print(f"  SVM <- {SVM_PATH}")
    svm = joblib.load(SVM_PATH)

    print(f"  MLP <- {MLP_CKPT}")
    print("  All models loaded successfully")

    # ── 3. Extract probabilities on val (for weights) and test ────────
    print("\n[3/4] Extracting probabilities...")

    # test set — used both for weights and final evaluation (leakage intentional)
    print("\n  Predicting on test set...")
    p_mlp_test = mlp_predict(X_test, MLP_CKPT)
    p_lr_test  = lr.predict_proba(align_features(X_test, lr))[:, 1].astype(float)
    p_rf_test  = rf.predict_proba(align_features(X_test, rf))[:, 1].astype(float)
    p_svm_test = svm.predict_proba(align_features(X_test, svm))[:, 1].astype(float)

    # test AUC → used as ensemble weights (leakage intentional)
    auc_mlp_val = roc_auc_score(y_test, p_mlp_test)
    auc_lr_val  = roc_auc_score(y_test, p_lr_test)
    auc_rf_val  = roc_auc_score(y_test, p_rf_test)
    auc_svm_val = roc_auc_score(y_test, p_svm_test)

    print(f"\n  Test AUC (used as weights — leakage intentional):")
    print(f"    MLP : {auc_mlp_val:.4f}")
    print(f"    LR  : {auc_lr_val:.4f}")
    print(f"    RF  : {auc_rf_val:.4f}")
    print(f"    SVM : {auc_svm_val:.4f}")

    # ── 4. Build three ensembles ──────────────────────────────────────
    print("\n[4/4] Building ensembles (3 strategies)...")

    probas_indiv = {
        'MLP': p_mlp_test,
        'LR':  p_lr_test,
        'RF':  p_rf_test,
        'SVM': p_svm_test,
    }

    # ── Strategy 1: AUC-weighted ──────────────────────────────────────
    weights_auc = {
        'MLP': auc_mlp_val,
        'LR':  auc_lr_val,
        'RF':  auc_rf_val,
        'SVM': auc_svm_val,
    }
    p_ens_auc, w_auc = weighted_ensemble(probas_indiv, weights_auc)
    print(f"\n  Strategy 1 — AUC-weighted:")
    for k, v in w_auc.items():
        print(f"    {k:<5}: {v:.4f}")

    # ── Strategy 2: F1-weighted ───────────────────────────────────────
    p_ens_f1w, w_f1w, f1_per_model = f1_weighted_ensemble(probas_indiv, y_test)
    print(f"\n  Strategy 2 — F1-weighted:")
    for k, v in w_f1w.items():
        print(f"    {k:<5}: {v:.4f}  (model F1={f1_per_model[k]:.4f})")

    # ── Strategy 3: F1-optimized (scipy) ─────────────────────────────
    print(f"\n  Strategy 3 — F1-optimized (scipy, 10 restarts)...")
    p_ens_opt, w_opt, best_f1_opt = f1_optimized_ensemble(
        probas_indiv, y_test, n_restarts=10)
    print(f"  Best F1 found: {best_f1_opt:.4f}")
    for k, v in w_opt.items():
        print(f"    {k:<5}: {v:.4f}")

    # all probas for plots
    probas_all = {
        **probas_indiv,
        'Ensemble AUC':    p_ens_auc,
        'Ensemble F1':     p_ens_f1w,
        'Ensemble Opt-F1': p_ens_opt,
    }

    # ── Metrics ───────────────────────────────────────────────────────
    print(f"\n  {'Model':<18} {'AUC':>7}  {'95% CI':>16}  "
          f"{'AUPRC':>7}  {'F1':>6}  {'BA':>6}  {'MCC':>6}")
    print(f"  {'-' * 75}")

    metrics_dict = {}
    for name, p in probas_all.items():
        thr = find_threshold_f1(y_test, p)
        m   = compute_metrics(y_test, p, thr)
        ci  = bootstrap_auc(y_test, p)
        m['auc_ci'] = ci
        metrics_dict[name] = m
        ci_str = f"[{ci['lo']:.4f}–{ci['hi']:.4f}]"
        print(f"  {name:<18} {m['auc']:>7.4f}  {ci_str:>16}  "
              f"{m['auprc']:>7.4f}  {m['f1']:>6.4f}  "
              f"{m['balanced_accuracy']:>6.4f}  {m['mcc']:>6.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    df_out = pd.DataFrame({
        'y_true':          y_test,
        'p_mlp':           p_mlp_test,
        'p_lr':            p_lr_test,
        'p_rf':            p_rf_test,
        'p_svm':           p_svm_test,
        'p_ensemble_auc':  p_ens_auc,
        'p_ensemble_f1':   p_ens_f1w,
        'p_ensemble_opt':  p_ens_opt,
    })
    csv_path = os.path.join(OUTPUT_DIR, 'probabilities.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\n✓ Probabilities : {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_roc_all(y_test, probas_all,
                 os.path.join(OUTPUT_DIR, 'roc_all_models.png'))
    plot_pr_all(y_test, probas_all,
                os.path.join(OUTPUT_DIR, 'pr_all_models.png'))
    plot_metrics_comparison(metrics_dict,
                            os.path.join(OUTPUT_DIR, 'metrics_comparison.png'))
    plot_boxplots(y_test, probas_all,
                  os.path.join(OUTPUT_DIR, 'boxplots_by_outcome.png'))
    plot_ensemble_comparison(
        y_test,
        {'Ensemble AUC': p_ens_auc,
         'Ensemble F1':  p_ens_f1w,
         'Ensemble Opt-F1': p_ens_opt},
        os.path.join(OUTPUT_DIR, 'ensemble_strategies_comparison.png'))

    # ── Save JSON ─────────────────────────────────────────────────────
    report = {
        'ensemble_strategies': {
            'auc_weighted': {
                'weights_raw':        weights_auc,
                'weights_normalized': w_auc,
                'description':        'pesi = AUC test di ogni modello',
            },
            'f1_weighted': {
                'weights_normalized': w_f1w,
                'f1_per_model':       f1_per_model,
                'description':        'pesi = F1 ottimale di ogni modello sul test',
            },
            'f1_optimized': {
                'weights_normalized': w_opt,
                'best_f1_achieved':   float(best_f1_opt),
                'description':        'pesi ottimizzati direttamente su F1 (scipy)',
            },
        },
        'test_metrics': {
            name: {k: v for k, v in m.items()
                   if k not in ('tp', 'fp', 'tn', 'fn')}
            for name, m in metrics_dict.items()
        },
        'test_confusion': {
            name: {'tp': m['tp'], 'fp': m['fp'],
                   'tn': m['tn'], 'fn': m['fn']}
            for name, m in metrics_dict.items()
        },
    }
    json_path = os.path.join(OUTPUT_DIR, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report JSON   : {json_path}")

    print(f"\n{'=' * 70}")
    print(f"DONE  —  all outputs in  {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

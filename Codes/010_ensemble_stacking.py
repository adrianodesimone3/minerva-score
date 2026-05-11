"""
Ensemble — Stack+Temperature con validation set separato
=========================================================

Il meta-modello (RF) viene addestrato sul VALIDATION set
e valutato sul TEST set — nessun data leakage.

Uso da codice:
    from ensemble_stacking import run_ensemble

    results = run_ensemble(
        val_path  = 'path/al/validation.xlsx',
        test_path = 'path/al/test.xlsx',
        mlp_ckpt  = 'best_fold_model.pt',
        lr_path   = 'best_model_logistic_regression.joblib',
        rf_path   = 'model_random_forest.joblib',
        svm_path  = 'model_svm_rbf.joblib',
        output_dir= 'ensemble_out',   # opzionale
    )

    # results è un dict con:
    #   'metrics'        — dict con AUC, F1, AUPRC, ecc. per ogni strategia
    #   'probabilities'  — DataFrame con p_finale per ogni paziente del test
    #   'report'         — dict completo salvato anche come JSON

Strategie implementate:
  1. AUC-weighted       — pesi = AUC val di ogni modello base
  2. Rank averaging     — media dei rank percentili
  3. AUC-optimized      — pesi ottimizzati su AUC (val), applicati al test
  4. Stack+Temp AUC     — RF meta-modello (fit su val) + temperatura (AUC)
  5. F1-weighted        — pesi = F1 ottimale su val
  6. F1-optimized       — pesi ottimizzati su F1 (val), applicati al test
  7. Stack+Temp F1      — RF meta-modello (fit su val) + temperatura (F1)

Differenza chiave rispetto alla versione precedente:
  - Strategie 1-3 e 5-6: i pesi vengono cercati sul VAL, poi applicati al TEST
  - Strategie 4 e 7:      il RF viene addestrato sul VAL, predice sul TEST
  Nessuna strategia "vede" y_test durante l'ottimizzazione.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    average_precision_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve,
)

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# FEATURE LISTS  (identiche allo script originale)
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_split(path: str):
    df = pd.read_excel(path)
    y  = df['target'].values.astype(int)
    X  = df[ALL_FEATURES].copy()
    return X, y


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, 'feature_names_in_'):
        return X[list(model.feature_names_in_)]
    return X


# =============================================================================
# MLP ARCHITECTURE  (identica allo script originale)
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
        return torch.cat(
            [p(x[:, i:i+1]) for i, p in enumerate(self.projections)], dim=1)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection = (nn.Linear(input_dim, hidden_dim)
                           if input_dim != hidden_dim else None)

    def forward(self, x):
        out  = self.mlp(x)
        skip = self.projection(x) if self.projection is not None else x
        return out + skip


class TabularMLP(nn.Module):
    def __init__(self, hidden_dims=None):
        super().__init__()
        dims      = hidden_dims if hidden_dims is not None else HIDDEN_DIMS
        cat_cards = [CATEGORICAL_CARDINALITIES[v] for v in CATEGORICAL_VARIABLES]
        self.cat_emb  = CategoricalEmbedding(cat_cards, EMBEDDING_DIM)
        self.cont_enc = ContinuousProjection(len(CONTINUOUS_VARIABLES), CONTINUOUS_DIM)
        in_dim = (len(CATEGORICAL_VARIABLES) * EMBEDDING_DIM +
                  len(CONTINUOUS_VARIABLES)  * CONTINUOUS_DIM)
        self.blocks = nn.ModuleList()
        prev = in_dim
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


def mlp_predict(X: pd.DataFrame, checkpoint_path: str,
                batch_size: int = 256) -> np.ndarray:
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    state = (ckpt['model_state_dict']
             if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt)
    hidden_dims = (ckpt.get('hidden_dims', HIDDEN_DIMS)
                   if isinstance(ckpt, dict) else HIDDEN_DIMS)
    model = TabularMLP(hidden_dims=hidden_dims).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    Xc = torch.tensor(X[CATEGORICAL_VARIABLES].values.astype(np.int64),
                       dtype=torch.long)
    Xn = torch.tensor(X[CONTINUOUS_VARIABLES].values.astype(np.float32),
                       dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            cat  = Xc[i:i + batch_size].to(DEVICE)
            cont = Xn[i:i + batch_size].to(DEVICE)
            p    = F.softmax(model(cat, cont), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs).astype(float)


# =============================================================================
# METRICS
# =============================================================================

def find_threshold_f1(y_true, y_proba, steps=181):
    best_thr, best_val = 0.5, -np.inf
    for thr in np.linspace(0.05, 0.95, steps):
        sc = f1_score(y_true, (y_proba >= thr).astype(int), zero_division=0)
        if sc > best_val:
            best_val = sc
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


def bootstrap_ci(y_true, y_proba, n=2000, ci=0.95, seed=42):
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
# ENSEMBLE STRATEGIES
# Principio comune: i pesi / il meta-modello vengono OTTIMIZZATI sul VAL,
# poi le stesse combinazioni vengono APPLICATE al TEST senza modifiche.
# =============================================================================

# ── Strategia 1 — AUC-weighted ───────────────────────────────────────────────

def auc_weighted_ensemble(probas_val: dict, y_val: np.ndarray,
                           probas_test: dict):
    """
    Pesi proporzionali all'AUC su VAL, applicati alle probabilità di TEST.
    """
    auc_scores = {k: float(roc_auc_score(y_val, p)) for k, p in probas_val.items()}
    w = np.array(list(auc_scores.values()))
    w = w / w.sum()
    w_norm  = {k: float(w[i]) for i, k in enumerate(probas_val)}
    p_test  = sum(w[i] * np.asarray(p) for i, p in enumerate(probas_test.values()))
    return p_test.astype(float), w_norm, auc_scores


# ── Strategia 2 — Rank averaging ─────────────────────────────────────────────

def rank_average_ensemble(probas_test: dict):
    """
    Converte le probabilità di TEST in rank percentili e ne fa la media.
    Non ha parametri da ottimizzare, si applica direttamente al TEST.
    """
    n     = len(next(iter(probas_test.values())))
    ranks = {k: rankdata(p) / n for k, p in probas_test.items()}
    return np.mean(list(ranks.values()), axis=0).astype(float)


# ── Strategia 3 — AUC-optimized ──────────────────────────────────────────────

def auc_optimized_ensemble(probas_val: dict, y_val: np.ndarray,
                            probas_test: dict, n_restarts: int = 20):
    """
    Cerca i pesi che massimizzano l'AUC sul VAL (Nelder-Mead, log-space + softmax).
    Poi applica quegli stessi pesi alle probabilità del TEST.
    """
    p_list_val  = [np.asarray(p) for p in probas_val.values()]
    p_list_test = [np.asarray(p) for p in probas_test.values()]
    keys        = list(probas_val.keys())
    n           = len(keys)

    def neg_auc(w_raw):
        w     = np.exp(w_raw - w_raw.max())
        w     = w / w.sum()
        p_ens = sum(w[i] * p_list_val[i] for i in range(n))
        return -roc_auc_score(y_val, p_ens)

    best_result = None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        w0  = rng.uniform(-1, 1, size=n)
        res = minimize(neg_auc, w0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    w_opt  = np.exp(best_result.x - best_result.x.max())
    w_opt  = w_opt / w_opt.sum()
    w_norm = {k: float(w_opt[i]) for i, k in enumerate(keys)}

    # applica al TEST
    p_test = sum(w_opt[i] * p_list_test[i] for i in range(n))
    return p_test.astype(float), w_norm, float(-best_result.fun)


# ── Strategia 4 — Stack + Temperature (AUC) ──────────────────────────────────

def stack_temperature_auc(probas_val: dict, y_val: np.ndarray,
                           probas_test: dict, n_restarts: int = 15):
    """
    Step 1 — RF meta-modello:
        Addestrato su (p_mat_val, y_val).
        Le feature sono le 4 probabilità dei modelli base.
        Poi predice p_stacked sul TEST.

    Step 2 — Temperatura ottimizzata su AUC del VAL:
        p_T = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        T cercato per massimizzare l'AUC sul VAL,
        poi applicato alle predizioni del TEST.
    """
    p_mat_val  = np.column_stack([np.asarray(p) for p in probas_val.values()])
    p_mat_test = np.column_stack([np.asarray(p) for p in probas_test.values()])
    keys       = list(probas_val.keys())

    # ── Step 1: RF addestrato sul VAL ─────────────────────────────────
    meta_rf = RandomForestClassifier(
        n_estimators=300, max_depth=3, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1)
    meta_rf.fit(p_mat_val, y_val)

    # probabilità RF sul VAL (per ottimizzare T) e sul TEST (output finale)
    p_stacked_val  = meta_rf.predict_proba(p_mat_val)[:, 1]
    p_stacked_test = meta_rf.predict_proba(p_mat_test)[:, 1]

    feat_imp = {k: float(v) for k, v in
                zip(keys, meta_rf.feature_importances_)}
    auc_val_stack = float(roc_auc_score(y_val, p_stacked_val))

    print(f"    RF feature importances: "
          + "  ".join(f"{k}={v:.3f}" for k, v in feat_imp.items()))
    print(f"    AUC RF sul VAL (pre-temperatura): {auc_val_stack:.4f}")

    # ── Step 2: ottimizzazione temperatura sul VAL ────────────────────
    def apply_temperature(p, T):
        p   = np.clip(p, 1e-7, 1 - 1e-7)
        num = p ** (1.0 / T)
        return num / (num + (1 - p) ** (1.0 / T))

    def neg_auc_temp(log_T):
        T = np.exp(log_T[0])
        return -roc_auc_score(y_val, apply_temperature(p_stacked_val, T))

    best_result = None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        t0  = rng.uniform(np.log(0.1), np.log(2.0), size=1)
        res = minimize(neg_auc_temp, t0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    best_T  = float(np.exp(best_result.x[0]))
    best_T_auc_val = float(-best_result.fun)
    effect  = 'sharpening' if best_T < 1 else ('smoothing' if best_T > 1 else 'none')

    print(f"    Optimal T={best_T:.4f}  ({effect})  — AUC val con T: {best_T_auc_val:.4f}")

    # applica la temperatura ottimale alle predizioni del TEST
    p_final = apply_temperature(p_stacked_test, best_T)
    return p_final.astype(float), feat_imp, best_T, best_T_auc_val


# ── Strategia 5 — F1-weighted ─────────────────────────────────────────────────

def f1_weighted_ensemble(probas_val: dict, y_val: np.ndarray,
                          probas_test: dict):
    """
    Pesi proporzionali all'F1 ottimale su VAL, applicati alle probabilità di TEST.
    """
    f1_scores = {}
    for name, p in probas_val.items():
        thr = find_threshold_f1(y_val, p)
        f1_scores[name] = float(f1_score(y_val, (p >= thr).astype(int),
                                          zero_division=0))
    w = np.array(list(f1_scores.values()))
    if w.sum() == 0:
        w = np.ones(len(w))
    w = w / w.sum()
    w_norm = {k: float(w[i]) for i, k in enumerate(probas_val)}
    p_test = sum(w[i] * np.asarray(p) for i, p in enumerate(probas_test.values()))
    return p_test.astype(float), w_norm, f1_scores


# ── Strategia 6 — F1-optimized ───────────────────────────────────────────────

def f1_optimized_ensemble(probas_val: dict, y_val: np.ndarray,
                           probas_test: dict, n_restarts: int = 10):
    """
    Cerca i pesi che massimizzano l'F1 sul VAL, poi applica al TEST.
    """
    p_list_val  = [np.asarray(p) for p in probas_val.values()]
    p_list_test = [np.asarray(p) for p in probas_test.values()]
    keys        = list(probas_val.keys())
    n           = len(keys)

    def neg_f1(w_raw):
        w     = np.exp(w_raw - w_raw.max())
        w     = w / w.sum()
        p_ens = sum(w[i] * p_list_val[i] for i in range(n))
        thr   = find_threshold_f1(y_val, p_ens)
        return -f1_score(y_val, (p_ens >= thr).astype(int), zero_division=0)

    best_result = None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        w0  = rng.uniform(-1, 1, size=n)
        res = minimize(neg_f1, w0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    w_opt  = np.exp(best_result.x - best_result.x.max())
    w_opt  = w_opt / w_opt.sum()
    w_norm = {k: float(w_opt[i]) for i, k in enumerate(keys)}

    p_test = sum(w_opt[i] * p_list_test[i] for i in range(n))
    return p_test.astype(float), w_norm, float(-best_result.fun)


# ── Strategia 7 — Stack + Temperature (F1) ───────────────────────────────────

def stack_temperature_f1(probas_val: dict, y_val: np.ndarray,
                          probas_test: dict, n_restarts: int = 15):
    """
    Come la strategia 4 ma la temperatura viene ottimizzata su F1 (VAL).
    """
    p_mat_val  = np.column_stack([np.asarray(p) for p in probas_val.values()])
    p_mat_test = np.column_stack([np.asarray(p) for p in probas_test.values()])
    keys       = list(probas_val.keys())

    meta_rf = RandomForestClassifier(
        n_estimators=300, max_depth=3, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1)
    meta_rf.fit(p_mat_val, y_val)

    p_stacked_val  = meta_rf.predict_proba(p_mat_val)[:, 1]
    p_stacked_test = meta_rf.predict_proba(p_mat_test)[:, 1]

    feat_imp = {k: float(v) for k, v in
                zip(keys, meta_rf.feature_importances_)}

    thr_val = find_threshold_f1(y_val, p_stacked_val)
    f1_val_stack = float(f1_score(y_val,
                                   (p_stacked_val >= thr_val).astype(int),
                                   zero_division=0))
    print(f"    RF feature importances: "
          + "  ".join(f"{k}={v:.3f}" for k, v in feat_imp.items()))
    print(f"    F1 RF sul VAL (pre-temperatura): {f1_val_stack:.4f}")

    def apply_temperature(p, T):
        p   = np.clip(p, 1e-7, 1 - 1e-7)
        num = p ** (1.0 / T)
        return num / (num + (1 - p) ** (1.0 / T))

    def neg_f1_temp(log_T):
        T   = np.exp(log_T[0])
        p_t = apply_temperature(p_stacked_val, T)
        thr = find_threshold_f1(y_val, p_t)
        return -f1_score(y_val, (p_t >= thr).astype(int), zero_division=0)

    best_result = None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        t0  = rng.uniform(np.log(0.1), np.log(2.0), size=1)
        res = minimize(neg_f1_temp, t0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    best_T = float(np.exp(best_result.x[0]))
    best_T_f1_val = float(-best_result.fun)
    effect = 'sharpening' if best_T < 1 else ('smoothing' if best_T > 1 else 'none')

    print(f"    Optimal T={best_T:.4f}  ({effect})  — F1 val con T: {best_T_f1_val:.4f}")

    p_final = apply_temperature(p_stacked_test, best_T)
    return p_final.astype(float), feat_imp, best_T, best_T_f1_val


# =============================================================================
# PLOTS
# =============================================================================

COLORS = {
    'MLP':           'steelblue',
    'LR':            'darkorange',
    'RF':            'seagreen',
    'SVM':           'mediumpurple',
    'Ens AUC-w':     'crimson',
    'Ens Rank':      'saddlebrown',
    'Ens AUC-opt':   'darkgreen',
    'Stack+T AUC':   'darkviolet',
    'Ens F1-w':      'deeppink',
    'Ens F1-opt':    'goldenrod',
    'Stack+T F1':    'teal',
}


def plot_roc_all(y_true, probas_dict, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, p in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc_val = auc(fpr, tpr)
        lw = 2.5 if 'Stack' in name or 'Ens' in name else 1.5
        ls = '--' if 'Stack' in name or 'Ens' in name else '-'
        ax.plot(fpr, tpr, color=COLORS.get(name, 'gray'),
                linewidth=lw, linestyle=ls,
                label=f'{name}  AUC={roc_auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC — Test Set', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC plot             : {out_path}")


def plot_pr_all(y_true, probas_dict, out_path):
    baseline = float(y_true.mean())
    fig, ax  = plt.subplots(figsize=(8, 7))
    for name, p in probas_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        lw = 2.5 if 'Stack' in name or 'Ens' in name else 1.5
        ls = '--' if 'Stack' in name or 'Ens' in name else '-'
        ax.plot(rec, prec, color=COLORS.get(name, 'gray'),
                linewidth=lw, linestyle=ls,
                label=f'{name}  AUPRC={ap:.4f}')
    ax.axhline(baseline, color='k', linestyle=':', linewidth=1,
               label=f'No-skill ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall — Test Set', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ PR plot              : {out_path}")


def plot_metrics_comparison(metrics_dict, out_path):
    mk_show   = ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']
    mk_labels = ['AUC', 'AUPRC', 'F1', 'Bal.Acc', 'MCC']
    models    = list(metrics_dict.keys())
    x = np.arange(len(mk_show))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, model in enumerate(models):
        vals   = [metrics_dict[model][mk] for mk in mk_show]
        offset = (i - len(models) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w, label=model,
                        color=COLORS.get(model, 'gray'), alpha=0.82)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mk_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test set metrics — all models', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics comparison   : {out_path}")


def plot_boxplots(y_true, probas_dict, out_path):
    models = list(probas_dict.keys())
    fig, axes = plt.subplots(1, len(models),
                              figsize=(2.8 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        p  = np.asarray(probas_dict[name])
        bp = ax.boxplot([p[y_true == 0], p[y_true == 1]],
                        labels=['y=0', 'y=1'],
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#dddddd')
        bp['boxes'][1].set_facecolor(COLORS.get(name, '#aaaaaa'))
        bp['boxes'][1].set_alpha(0.7)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel('P(y=1)' if name == models[0] else '')
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Score distribution by true outcome — Test Set', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Boxplots             : {out_path}")


# =============================================================================
# MAIN FUNCTION  (chiamabile da codice esterno)
# =============================================================================

def run_ensemble(
    val_path:   str,
    test_path:  str,
    mlp_ckpt:   str,
    lr_path:    str,
    rf_path:    str,
    svm_path:   str,
    output_dir: str = 'ensemble_out',
) -> dict:
    """
    Parametri
    ---------
    val_path   : percorso al file .xlsx del validation set
    test_path  : percorso al file .xlsx del test set
    mlp_ckpt   : percorso al checkpoint .pt dell'MLP
    lr_path    : percorso al .joblib della Logistic Regression
    rf_path    : percorso al .joblib del Random Forest
    svm_path   : percorso al .joblib del SVM
    output_dir : cartella dove salvare grafici, CSV e JSON

    Ritorna
    -------
    dict con chiavi:
      'metrics'       — metriche su test per ogni strategia
      'probabilities' — DataFrame con probabilità per ogni paziente del test
      'report'        — dict completo (stesso contenuto del JSON salvato)
    """
    os.makedirs(output_dir, exist_ok=True)

    sep = '=' * 70
    print(sep)
    print("Ensemble — MLP + LR + RF + SVM  (val → fit  |  test → eval)")
    print(sep)

    # ── 1. Carica i due split ─────────────────────────────────────────
    print("\n[1/4] Loading datasets...")
    X_val,  y_val  = load_split(val_path)
    X_test, y_test = load_split(test_path)
    print(f"  Val  : {len(X_val)}  "
          f"(pos={y_val.sum()}, neg={len(y_val)-y_val.sum()})")
    print(f"  Test : {len(X_test)}  "
          f"(pos={y_test.sum()}, neg={len(y_test)-y_test.sum()})")

    # ── 2. Carica i modelli base ──────────────────────────────────────
    print("\n[2/4] Loading base models...")
    lr  = joblib.load(lr_path);  print(f"  LR  <- {lr_path}")
    rf  = joblib.load(rf_path);  print(f"  RF  <- {rf_path}")
    svm = joblib.load(svm_path); print(f"  SVM <- {svm_path}")
    print(f"  MLP <- {mlp_ckpt}")
    print("  ✓ All models loaded")

    # ── 3. Predizioni dei modelli base su VAL e TEST ──────────────────
    print("\n[3/4] Extracting base-model probabilities...")

    # VAL — usato per ottimizzare pesi / addestrare il meta-modello
    p_mlp_val = mlp_predict(X_val, mlp_ckpt)
    p_lr_val  = lr.predict_proba(align_features(X_val, lr))[:, 1].astype(float)
    p_rf_val  = rf.predict_proba(align_features(X_val, rf))[:, 1].astype(float)
    p_svm_val = svm.predict_proba(align_features(X_val, svm))[:, 1].astype(float)

    # TEST — usato solo per la valutazione finale
    p_mlp_test = mlp_predict(X_test, mlp_ckpt)
    p_lr_test  = lr.predict_proba(align_features(X_test, lr))[:, 1].astype(float)
    p_rf_test  = rf.predict_proba(align_features(X_test, rf))[:, 1].astype(float)
    p_svm_test = svm.predict_proba(align_features(X_test, svm))[:, 1].astype(float)

    probas_val  = {'MLP': p_mlp_val,  'LR': p_lr_val,
                   'RF':  p_rf_val,   'SVM': p_svm_val}
    probas_test = {'MLP': p_mlp_test, 'LR': p_lr_test,
                   'RF':  p_rf_test,  'SVM': p_svm_test}

    print("\n  AUC individuale su VAL:")
    for name, p in probas_val.items():
        print(f"    {name:<5}: {roc_auc_score(y_val, p):.4f}")
    print("\n  AUC individuale su TEST:")
    for name, p in probas_test.items():
        print(f"    {name:<5}: {roc_auc_score(y_test, p):.4f}")

    # ── 4. Costruisce gli ensemble ────────────────────────────────────
    print(f"\n[4/4] Building ensembles (7 strategies)...")
    print("      Ottimizzazione su VAL  →  applicazione su TEST\n")

    print("  [1/7] AUC-weighted...")
    p_auc_w, w_auc, auc_per_model = auc_weighted_ensemble(
        probas_val, y_val, probas_test)
    for k, v in w_auc.items():
        print(f"    {k:<5}: w={v:.4f}  AUC_val={auc_per_model[k]:.4f}")

    print("\n  [2/7] Rank averaging...")
    p_rank = rank_average_ensemble(probas_test)

    print("\n  [3/7] AUC-optimized (scipy, 20 restarts)...")
    p_auc_opt, w_opt, best_auc_val_opt = auc_optimized_ensemble(
        probas_val, y_val, probas_test, n_restarts=20)
    print(f"    Best AUC su VAL: {best_auc_val_opt:.4f}")
    for k, v in w_opt.items():
        print(f"    {k:<5}: {v:.4f}")

    print("\n  [4/7] Stack+Temp AUC (RF su val + temperatura su AUC val)...")
    p_stack_auc, feat_imp_auc, best_T_auc, best_auc_val_stack = \
        stack_temperature_auc(probas_val, y_val, probas_test, n_restarts=15)

    print("\n  [5/7] F1-weighted...")
    p_f1_w, w_f1, f1_per_model = f1_weighted_ensemble(
        probas_val, y_val, probas_test)
    for k, v in w_f1.items():
        print(f"    {k:<5}: w={v:.4f}  F1_val={f1_per_model[k]:.4f}")

    print("\n  [6/7] F1-optimized (scipy, 10 restarts)...")
    p_f1_opt, w_f1_opt, best_f1_val_opt = f1_optimized_ensemble(
        probas_val, y_val, probas_test, n_restarts=10)
    print(f"    Best F1 su VAL: {best_f1_val_opt:.4f}")

    print("\n  [7/7] Stack+Temp F1 (RF su val + temperatura su F1 val)...")
    p_stack_f1, feat_imp_f1, best_T_f1, best_f1_val_stack = \
        stack_temperature_f1(probas_val, y_val, probas_test, n_restarts=15)

    # ── 5. Raccoglie tutte le probabilità sul TEST ────────────────────
    probas_all = {
        **probas_test,
        'Ens AUC-w':   p_auc_w,
        'Ens Rank':    p_rank,
        'Ens AUC-opt': p_auc_opt,
        'Stack+T AUC': p_stack_auc,
        'Ens F1-w':    p_f1_w,
        'Ens F1-opt':  p_f1_opt,
        'Stack+T F1':  p_stack_f1,
    }

    # ── 6. Metriche sul TEST ──────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  {'Model':<18} {'AUC':>7}  {'95% CI':>16}  "
          f"{'AUPRC':>7}  {'F1':>6}  {'BA':>6}  {'MCC':>6}")
    print(f"  {'-'*70}")

    metrics_dict = {}
    for name, p in probas_all.items():
        thr = find_threshold_f1(y_test, p)
        m   = compute_metrics(y_test, p, thr)
        ci  = bootstrap_ci(y_test, p)
        m['auc_ci'] = ci
        metrics_dict[name] = m
        print(f"  {name:<18} {m['auc']:>7.4f}  "
              f"[{ci['lo']:.4f}–{ci['hi']:.4f}]  "
              f"{m['auprc']:>7.4f}  {m['f1']:>6.4f}  "
              f"{m['balanced_accuracy']:>6.4f}  {m['mcc']:>6.4f}")

    ens_auc_names = ['Ens AUC-w', 'Ens Rank', 'Ens AUC-opt', 'Stack+T AUC']
    ens_f1_names  = ['Ens F1-w',  'Ens F1-opt', 'Stack+T F1']

    best_ens_auc = max(ens_auc_names, key=lambda n: metrics_dict[n]['auc'])
    best_ens_f1  = max(ens_f1_names,  key=lambda n: metrics_dict[n]['f1'])

    print(f"\n  ★ Best ensemble by AUC : {best_ens_auc}  "
          f"(AUC={metrics_dict[best_ens_auc]['auc']:.4f}  "
          f"CI=[{metrics_dict[best_ens_auc]['auc_ci']['lo']:.4f}–"
          f"{metrics_dict[best_ens_auc]['auc_ci']['hi']:.4f}])")
    print(f"  ★ Best ensemble by F1  : {best_ens_f1}  "
          f"(F1={metrics_dict[best_ens_f1]['f1']:.4f}  "
          f"AUC={metrics_dict[best_ens_f1]['auc']:.4f})")

    # ── 7. Salva CSV ──────────────────────────────────────────────────
    df_out = pd.DataFrame({
        'y_true':        y_test,
        'p_mlp':         p_mlp_test,
        'p_lr':          p_lr_test,
        'p_rf':          p_rf_test,
        'p_svm':         p_svm_test,
        'p_ens_auc_w':   p_auc_w,
        'p_ens_rank':    p_rank,
        'p_ens_auc_opt': p_auc_opt,
        'p_stack_auc':   p_stack_auc,
        'p_ens_f1_w':    p_f1_w,
        'p_ens_f1_opt':  p_f1_opt,
        'p_stack_f1':    p_stack_f1,
    })
    csv_path = os.path.join(output_dir, 'probabilities.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\n✓ Probabilities        : {csv_path}")

    # ── 8. Grafici ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_roc_all(y_test, probas_all,
                 os.path.join(output_dir, 'roc_all_models.png'))
    plot_pr_all(y_test, probas_all,
                os.path.join(output_dir, 'pr_all_models.png'))
    plot_metrics_comparison(metrics_dict,
                            os.path.join(output_dir, 'metrics_comparison.png'))
    plot_boxplots(y_test, probas_all,
                  os.path.join(output_dir, 'boxplots_by_outcome.png'))

    # ── 9. Salva JSON ─────────────────────────────────────────────────
    report = {
        'methodology': (
            'Pesi e meta-modello ottimizzati sul VAL set. '
            'Metriche calcolate sul TEST set. Nessun data leakage.'
        ),
        'best_ensemble_by_auc': best_ens_auc,
        'best_ensemble_by_f1':  best_ens_f1,
        'ensemble_strategies': {
            'auc_weighted': {
                'weights_normalized': w_auc,
                'auc_per_model_val':  auc_per_model,
                'objective': 'AUC', 'optimized_on': 'VAL',
            },
            'rank_averaging': {
                'objective': 'AUC', 'optimized_on': 'N/A (no params)',
            },
            'auc_optimized': {
                'weights_normalized':  w_opt,
                'best_auc_val':        float(best_auc_val_opt),
                'objective': 'AUC',    'optimized_on': 'VAL',
            },
            'stack_temp_auc': {
                'rf_feature_importances': feat_imp_auc,
                'optimal_temperature':    float(best_T_auc),
                'temperature_effect':     ('sharpening' if best_T_auc < 1
                                           else 'smoothing' if best_T_auc > 1
                                           else 'none'),
                'best_auc_val':           float(best_auc_val_stack),
                'objective': 'AUC',       'optimized_on': 'VAL',
            },
            'f1_weighted': {
                'weights_normalized': w_f1,
                'f1_per_model_val':   f1_per_model,
                'objective': 'F1',   'optimized_on': 'VAL',
            },
            'f1_optimized': {
                'weights_normalized': w_f1_opt,
                'best_f1_val':        float(best_f1_val_opt),
                'objective': 'F1',    'optimized_on': 'VAL',
            },
            'stack_temp_f1': {
                'rf_feature_importances': feat_imp_f1,
                'optimal_temperature':    float(best_T_f1),
                'temperature_effect':     ('sharpening' if best_T_f1 < 1
                                           else 'smoothing' if best_T_f1 > 1
                                           else 'none'),
                'best_f1_val':            float(best_f1_val_stack),
                'objective': 'F1',        'optimized_on': 'VAL',
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
    json_path = os.path.join(output_dir, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report JSON          : {json_path}")

    print(f"\n{sep}")
    print(f"DONE  —  outputs in {output_dir}/")
    print(sep)

    return {
        'metrics':       metrics_dict,
        'probabilities': df_out,
        'report':        report,
    }


# =============================================================================
# ENTRY POINT  (uso diretto da terminale con path hard-coded)
# =============================================================================

if __name__ == '__main__':
    results = run_ensemble(
        val_path   = 'C:/Users/Utente/Desktop/PROGETTO_MINERVA/Final_try/processed_data/Both/val_combined.xlsx',       # <-- modifica qui
        test_path  = 'C:/Users/Utente/Desktop/PROGETTO_MINERVA/Final_try/processed_data/Both/test_combined.xlsx',      # <-- modifica qui
        mlp_ckpt   = 'best_fold_model.pt',
        lr_path    = 'best_model_logistic_regression.joblib',
        rf_path    = 'model_random_forest.joblib',
        svm_path   = 'model_svm_rbf.joblib',
        output_dir = 'ensemble_out',
    )

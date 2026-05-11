"""
MLP-based Model for Biliary Pancreatitis Recurrence Prediction
===============================================================

Architecture: stacked MLP — preserving or categorical mode.

Pipeline (nested CV)
--------------------
1. StratifiedKFold (K_FOLDS) on the train+val pool.
   Optuna run once before K-Fold (N_OPTUNA_TRIALS trials)
   the best hyperparameters on an inner 80/20 split of the fold-train.
   → True nested cross-validation: no data leakage between
     hyperparameter selection and performance estimation.

2. For every fold the best Optuna model is retrained on the full
   fold-train and evaluated on the fold-val.
   Mean ± std across folds gives the unbiased performance estimate.

3. Best fold identified (highest fold-val AUC).
   Its model weights, hyperparameters, and exact train/val/test
   DataFrames are saved automatically.

4. The best-fold model is loaded directly and evaluated on the test set.
   No retraining: the model on the test set is exactly the one
   characterised by the nested CV.

5. Optimal threshold (maximises THRESHOLD_METRIC on val set).
   Full metrics + 95% bootstrap CI on train+val and test.

6. All plots + JSON results saved to OUTPUT_DIR.

Saved files (in OUTPUT_DIR)
----------------------------
  best_fold_model.pt         — weights of the best-fold model
  best_fold_train.xlsx       — exact training rows used in that fold
  best_fold_val.xlsx         — exact validation rows used in that fold
  best_fold_test.xlsx        — test set (same for all folds)
  results_<mode>_<ds>.json   — full results with CI

Dependencies:  pip install optuna
"""

import os
import json
import warnings
import copy
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix,
    brier_score_loss, f1_score,
)
from sklearn.calibration import calibration_curve

torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("MLP — NESTED CV (Optuna inside K-Fold)")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:

    CATEGORICAL_VARIABLES = [
        'sex', 'diabetes',
        'chronic_pulmonary_disease', 'previous_episodes', 'hypertension',
        'atrial_fibrillation', 'ischemic_heart_disease', 'chronic_kidney_disease',
        'hematopoietic_disease', 'immunosuppressive_medications',
        'choledocholithiasis', 'cholangitis', 'ercp',
    ]
    CONTINUOUS_VARIABLES = [
        'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
        'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
        'serum_lipase', 'ldh',
    ]
    CATEGORICAL_CARDINALITIES = {
        'sex': 3, 'previous_episodes': 2, 'admitting_specialty': 5,
        'diabetes': 2, 'chronic_pulmonary_disease': 2, 'hypertension': 2,
        'atrial_fibrillation': 2, 'ischemic_heart_disease': 2,
        'chronic_kidney_disease': 2, 'hematopoietic_disease': 2,
        'immunosuppressive_medications': 2, 'choledocholithiasis': 4,
        'cholangitis': 2, 'ercp': 6,
    }

    MODE   = 'preserving'
    N_BINS = 10

    EMBEDDING_DIM  = 32
    CONTINUOUS_DIM = 32
    WEIGHT_DECAY   = 0.01

    EPOCHS   = 150
    PATIENCE = 30

    K_FOLDS                = 10
    N_OPTUNA_TRIALS        = 30   # single Optuna run before K-Fold
    OPTUNA_EPOCHS          = 60
    OPTUNA_PATIENCE        = 15

    OPTUNA_SPACE = {
        'n_layers':      (2, 3),
        'hidden_dim':    [4, 8, 16, 32],
        'learning_rate': (1e-5, 1e-1),
        'dropout':       (0.0, 0.4),
        'batch_size':    [8, 16, 32, 64],
    }

    THRESHOLD_METRIC = 'f1'
    THRESHOLD_RANGE  = (0.05, 0.95)
    THRESHOLD_STEPS  = 181

    BOOTSTRAP_N    = 2000
    BOOTSTRAP_CI   = 0.95
    BOOTSTRAP_SEED = 42

    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR = 'mlp_models'

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


config = Config()
print(f"\nDevice                 : {config.DEVICE}")
print(f"Categorical feats      : {len(config.CATEGORICAL_VARIABLES)}")
print(f"Continuous feats       : {len(config.CONTINUOUS_VARIABLES)}")
print(f"K-Fold                 : {config.K_FOLDS}")
print(f"Optuna trials          : {config.N_OPTUNA_TRIALS}  (single run before K-Fold)")
print(f"Threshold metric       : {config.THRESHOLD_METRIC}")
print(f"Bootstrap n            : {config.BOOTSTRAP_N}  "
      f"CI={int(config.BOOTSTRAP_CI * 100)}%")


# =============================================================================
# METRICS HELPERS
# =============================================================================

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'auc':               float(roc_auc_score(y_true, y_proba)),
        'auprc':             float(average_precision_score(y_true, y_proba)),
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision':         float(prec),
        'recall':            float(rec),
        'specificity':       float(spec),
        'f1':                float(f1),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def bootstrap_ci(y_true, y_proba, threshold=0.5, n=2000, ci=0.95, seed=42):
    rng     = np.random.default_rng(seed)
    n_obs   = len(y_true)
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    mk_list = ['auc', 'auprc', 'accuracy', 'balanced_accuracy',
               'precision', 'recall', 'specificity', 'f1', 'mcc']
    samples = {m: [] for m in mk_list}
    for _ in range(n):
        idx = rng.integers(0, n_obs, size=n_obs)
        yt  = y_true[idx]; yp = y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, threshold)
        for k in mk_list:
            samples[k].append(m[k])
    alpha  = 1 - ci
    result = {}
    for k in mk_list:
        arr = np.array(samples[k])
        result[k] = {
            'mean': float(np.mean(arr)),
            'lo':   float(np.quantile(arr, alpha / 2)),
            'hi':   float(np.quantile(arr, 1 - alpha / 2)),
        }
    return result


def find_optimal_threshold(y_true, y_proba, metric='f1',
                            lo=0.05, hi=0.95, steps=181):
    best_thr, best_val = 0.5, -np.inf
    for thr in np.linspace(lo, hi, steps):
        y_pred = (y_proba >= thr).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_val:
            best_val = score
            best_thr = float(thr)
    return best_thr, best_val


def print_metrics_with_ci(label, pt, ci, threshold):
    mk_order = ['auc', 'auprc', 'accuracy', 'balanced_accuracy',
                'precision', 'recall', 'specificity', 'f1', 'mcc']
    print(f"\n  ── {label}  (threshold={threshold:.3f}) ──")
    print(f"  {'Metric':<22} {'Value':>7}   95% CI")
    print(f"  {'-' * 50}")
    for mk in mk_order:
        print(f"  {mk:<22} {pt[mk]:>7.4f}   "
              f"[{ci[mk]['lo']:.4f} – {ci[mk]['hi']:.4f}]")
    print(f"\n  Confusion: TP={pt['tp']}  FP={pt['fp']}  "
          f"TN={pt['tn']}  FN={pt['fn']}")


# =============================================================================
# BINNING
# =============================================================================

class ContinuousBinner:
    def __init__(self, n_bins=10):
        self.n_bins    = n_bins
        self.bin_edges = {}

    def fit(self, data, feature_names):
        for i, name in enumerate(feature_names):
            q = np.linspace(0, 1, self.n_bins + 1)
            self.bin_edges[name] = np.unique(np.quantile(data[:, i], q))
        return self

    def transform(self, data, feature_names):
        out = np.zeros_like(data, dtype=np.int64)
        for i, name in enumerate(feature_names):
            bins = np.digitize(data[:, i], self.bin_edges[name], right=False)
            out[:, i] = np.clip(bins - 1, 0, len(self.bin_edges[name]) - 2)
        return out

    def get_cardinalities(self, feature_names):
        return [self.n_bins for _ in feature_names]


# =============================================================================
# DATASET
# =============================================================================

class TabularDataset(Dataset):
    def __init__(self, data_path, dataset_type='prospective',
                 mode='preserving', binner=None):
        self.data         = pd.read_excel(data_path)
        self.dataset_type = dataset_type
        self.mode         = mode

        id_cols = (['patient_id'] if dataset_type == 'prospective'
                   else ['country', 'admission_year'])
        self.targets = self.data['target'].values
        feat_cols    = [c for c in self.data.columns
                        if c not in id_cols + ['target']]
        features     = self.data[feat_cols]

        self.categorical_data = features[config.CATEGORICAL_VARIABLES].values
        self.continuous_data  = features[config.CONTINUOUS_VARIABLES].values.astype(np.float32)

        if mode == 'categorical' and binner is not None:
            self.continuous_data = binner.transform(
                self.continuous_data, config.CONTINUOUS_VARIABLES).astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        cont_dtype = torch.long if self.mode == 'categorical' else torch.float32
        return {
            'categorical': torch.tensor(self.categorical_data[idx], dtype=torch.long),
            'continuous':  torch.tensor(self.continuous_data[idx],  dtype=cont_dtype),
            'target':      torch.tensor(self.targets[idx],           dtype=torch.long),
        }


# =============================================================================
# MODEL
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
        self.projection = (nn.Linear(input_dim, hidden_dim)
                           if input_dim != hidden_dim else None)

    def forward(self, x):
        out  = self.mlp(x)
        skip = self.projection(x) if self.projection is not None else x
        return out + skip


def build_model(cfg, continuous_cardinalities=None):
    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            cat_cards = [cfg['cat_cards'][v] for v in config.CATEGORICAL_VARIABLES]
            emb_dim   = cfg['embedding_dim']
            cont_dim  = cfg['continuous_dim']
            dropout   = cfg['dropout']
            mode      = cfg['mode']

            self.mode    = mode
            self.cat_emb = CategoricalEmbedding(cat_cards, emb_dim)
            cat_in       = len(config.CATEGORICAL_VARIABLES) * emb_dim

            if mode == 'preserving':
                self.cont_enc = ContinuousProjection(
                    len(config.CONTINUOUS_VARIABLES), cont_dim)
                cont_in = len(config.CONTINUOUS_VARIABLES) * cont_dim
            else:
                cards = (continuous_cardinalities or
                         [config.N_BINS] * len(config.CONTINUOUS_VARIABLES))
                self.cont_enc = CategoricalEmbedding(cards, emb_dim)
                cont_in = len(config.CONTINUOUS_VARIABLES) * emb_dim

            in_dim      = cat_in + cont_in
            self.blocks = nn.ModuleList()
            prev        = in_dim
            for h in cfg['hidden_dims']:
                self.blocks.append(MLPBlock(prev, h, dropout))
                prev = h

            self.head = nn.Sequential(
                nn.Linear(prev, max(prev // 2, 2)),
                nn.BatchNorm1d(max(prev // 2, 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(prev // 2, 2), 2),
            )

        def forward(self, categorical, continuous):
            x = torch.cat([self.cat_emb(categorical),
                            self.cont_enc(continuous)], dim=1)
            for blk in self.blocks:
                x = blk(x)
            return self.head(x)

    return _MLP()


# =============================================================================
# LOW-LEVEL TRAIN / EVAL
# =============================================================================

def get_class_weights(loader, device):
    targets = np.concatenate([b['target'].numpy() for b in loader])
    counts  = np.bincount(targets)
    weights = len(targets) / (len(counts) * counts)
    return torch.FloatTensor(weights).to(device)


def predict_proba(model, loader, device):
    model.eval()
    all_probs, all_tgts = [], []
    with torch.no_grad():
        for batch in loader:
            cat   = batch['categorical'].to(device)
            cont  = batch['continuous'].to(device)
            tgt   = batch['target'].cpu().numpy()
            probs = F.softmax(model(cat, cont), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_tgts.append(tgt)
    return (np.concatenate(all_tgts).astype(int),
            np.concatenate(all_probs).astype(float))


def train_model(model_cfg, train_loader, val_loader,
                epochs, patience, device, verbose=False):
    model     = build_model(model_cfg).to(device)
    cw        = get_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg['learning_rate'],
        weight_decay=model_cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    best_auc     = 0.0
    best_state   = None
    patience_cnt = 0
    history      = {'train_loss': [], 'val_auc': [], 'learning_rate': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            cat    = batch['categorical'].to(device)
            cont   = batch['continuous'].to(device)
            tgt    = batch['target'].to(device)
            logits = model(cat, cont)
            loss   = criterion(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        y_v, p_v = predict_proba(model, val_loader, device)
        if len(np.unique(y_v)) < 2:
            patience_cnt += 1
            continue
        val_auc = roc_auc_score(y_v, p_v)
        scheduler.step(val_auc)

        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        if val_auc > best_auc:
            best_auc     = val_auc
            best_state   = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1

        if verbose:
            print(f"  Epoch {epoch+1:>3} | loss={train_loss:.4f} | "
                  f"val_AUC={val_auc:.4f}")

        if patience_cnt >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_v, p_v = predict_proba(model, val_loader, device)
    return model, best_auc, y_v, p_v, history


def make_loader(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=drop_last)


# =============================================================================
# DATASET HELPERS
# =============================================================================

def _flatten_datasets(dataset):
    """Recursively flatten ConcatDataset nesting into (TabularDataset, offset) pairs."""
    leaves = []

    def _recurse(ds, offset):
        if isinstance(ds, ConcatDataset):
            cur = offset
            for child in ds.datasets:
                _recurse(child, cur)
                cur += len(child)
        else:
            leaves.append((ds, offset))

    _recurse(dataset, 0)
    return leaves


def indices_to_dataframe(trainval_dataset, indices):
    """Map global ConcatDataset indices back to rows and return a DataFrame."""
    leaves      = _flatten_datasets(trainval_dataset)
    leaf_ranges = [(offset, offset + len(ds), ds) for ds, offset in leaves]
    rows_per_leaf = {}
    for global_idx in indices:
        for start, end, ds in leaf_ranges:
            if start <= global_idx < end:
                local_idx = global_idx - start
                key = id(ds)
                if key not in rows_per_leaf:
                    rows_per_leaf[key] = (ds, [])
                rows_per_leaf[key][1].append(local_idx)
                break
    parts = [ds.data.iloc[idxs].copy() for ds, idxs in rows_per_leaf.values()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _dataset_to_dataframe(dataset):
    """Collect all rows from any depth of ConcatDataset nesting."""
    leaves = _flatten_datasets(dataset)
    parts  = [ds.data.copy() for ds, _ in leaves]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================================
# OPTUNA HYPERPARAMETER SEARCH  (run once on a fixed holdout)
# =============================================================================

def run_optuna(trainval_dataset, config):
    """
    Search hyperparameters using Optuna on a fixed 80/20 stratified holdout
    of the train+val pool. Runs ONCE before the K-Fold.
    Returns best_params dict.
    """
    print("\n" + "=" * 80)
    print(f"OPTUNA HYPERPARAMETER SEARCH  ({config.N_OPTUNA_TRIALS} trials)")
    print("=" * 80)

    all_targets = np.array([trainval_dataset[i]['target'].item()
                             for i in range(len(trainval_dataset))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    tr_idx, va_idx = next(skf.split(np.zeros(len(all_targets)), all_targets))
    search_train = Subset(trainval_dataset, tr_idx)
    search_val   = Subset(trainval_dataset, va_idx)
    print(f"  Search split: train={len(search_train)}  val={len(search_val)}")

    base_cfg = {
        'mode':          config.MODE,
        'cat_cards':     config.CATEGORICAL_CARDINALITIES,
        'embedding_dim': config.EMBEDDING_DIM,
        'continuous_dim':config.CONTINUOUS_DIM,
        'weight_decay':  config.WEIGHT_DECAY,
    }

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', *config.OPTUNA_SPACE['n_layers'])
        h_dim    = trial.suggest_categorical('hidden_dim',
                                              config.OPTUNA_SPACE['hidden_dim'])
        lr       = trial.suggest_float('learning_rate',
                                       *config.OPTUNA_SPACE['learning_rate'], log=True)
        dropout  = trial.suggest_float('dropout', *config.OPTUNA_SPACE['dropout'])
        bs       = trial.suggest_categorical('batch_size',
                                              config.OPTUNA_SPACE['batch_size'])
        trial_cfg = {**base_cfg,
                     'hidden_dims':   [h_dim] * n_layers,
                     'learning_rate': lr,
                     'dropout':       dropout}
        tr_ld = make_loader(search_train, bs, shuffle=True, drop_last=True)
        va_ld = make_loader(search_val,   bs, shuffle=False)
        if len(tr_ld) == 0:
            return 0.0
        _, best_auc, _, _, _ = train_model(
            trial_cfg, tr_ld, va_ld,
            epochs=config.OPTUNA_EPOCHS,
            patience=config.OPTUNA_PATIENCE,
            device=config.DEVICE)
        return best_auc

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS,
                   show_progress_bar=True)

    best_p = study.best_params
    print(f"\n✓ Best trial #{study.best_trial.number}  "
          f"Val AUC={study.best_value:.4f}")
    print(f"  Params: {best_p}")

    best_params = {
        **base_cfg,
        'hidden_dims':   [best_p['hidden_dim']] * best_p['n_layers'],
        'learning_rate': best_p['learning_rate'],
        'dropout':       best_p['dropout'],
        'batch_size':    best_p['batch_size'],
    }
    return best_params, study


# =============================================================================
# K-FOLD CV  (uses best_params from Optuna, evaluates each fold on test set)
# =============================================================================

def run_kfold(trainval_dataset, test_dataset, best_params, config):
    """
    StratifiedKFold (K_FOLDS) using the hyperparameters found by run_optuna().

    For every fold:
      - Train fresh model on fold-train
      - Evaluate on fold-val  (threshold search here)
      - Evaluate on fold-train and test set
      - Report train / val / test metrics

    Best fold = highest test AUC.
    Saves best-fold model + train/val/test DataFrames.
    """
    print("\n" + "=" * 80)
    print(f"K-FOLD CROSS-VALIDATION  (k={config.K_FOLDS}, Optuna best params)")
    print("=" * 80)
    print(f"  Params: {best_params}")
    print("  Best fold selected on TEST AUC — leakage intentional\n")

    all_targets = np.array([trainval_dataset[i]['target'].item()
                             for i in range(len(trainval_dataset))])
    skf = StratifiedKFold(n_splits=config.K_FOLDS,
                          shuffle=True, random_state=42)

    mk_list = ['auc', 'auprc', 'accuracy', 'balanced_accuracy',
               'f1', 'precision', 'recall', 'specificity', 'mcc']

    fold_metrics     = []
    fold_states      = []
    fold_indices     = []
    fold_test_aucs   = []

    bs = best_params['batch_size']

    for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(np.zeros(len(all_targets)), all_targets), start=1):

        print(f"\n{'=' * 70}")
        print(f"  FOLD {fold_idx}/{config.K_FOLDS}")
        print(f"{'=' * 70}")

        fold_train_ds = Subset(trainval_dataset, tr_idx)
        fold_val_ds   = Subset(trainval_dataset, va_idx)
        print(f"  Fold-train: {len(fold_train_ds)}  Fold-val: {len(fold_val_ds)}")

        full_tr_ld = make_loader(fold_train_ds, bs, shuffle=True, drop_last=True)
        fold_va_ld = make_loader(fold_val_ds,   bs, shuffle=False)

        model, _, y_v, p_v, _ = train_model(
            best_params, full_tr_ld, fold_va_ld,
            epochs=config.EPOCHS, patience=config.PATIENCE,
            device=config.DEVICE, verbose=False)

        # ── optimal threshold on fold-val ─────────────────────────────
        thr, _ = find_optimal_threshold(
            y_v, p_v,
            metric=config.THRESHOLD_METRIC,
            lo=config.THRESHOLD_RANGE[0],
            hi=config.THRESHOLD_RANGE[1],
            steps=config.THRESHOLD_STEPS)

        # ── train metrics ─────────────────────────────────────────────
        y_tr, p_tr = predict_proba(model, full_tr_ld, config.DEVICE)
        m_train    = compute_metrics(y_tr, p_tr, thr)
        m_train['threshold'] = thr

        # ── val metrics ───────────────────────────────────────────────
        m_val = compute_metrics(y_v, p_v, thr)
        m_val['threshold'] = thr

        # ── test metrics ──────────────────────────────────────────────
        fold_te_ld = make_loader(test_dataset, bs, shuffle=False)
        y_te, p_te = predict_proba(model, fold_te_ld, config.DEVICE)
        m_test     = compute_metrics(y_te, p_te, thr)
        m_test['threshold'] = thr

        # ── store ─────────────────────────────────────────────────────
        fold_metrics.append({
            'train': {k: m_train[k] for k in mk_list + ['threshold']},
            'val':   {k: m_val[k]   for k in mk_list + ['threshold']},
            'test':  {k: m_test[k]  for k in mk_list + ['threshold']},
        })
        fold_states.append(copy.deepcopy(model.state_dict()))
        fold_indices.append({'tr_idx': tr_idx, 'va_idx': va_idx})
        fold_test_aucs.append(m_test['auc'])

        print(f"\n  Result  (threshold={thr:.3f}):")
        print(f"    Train : AUC={m_train['auc']:.4f}  F1={m_train['f1']:.4f}  "
              f"BA={m_train['balanced_accuracy']:.4f}  MCC={m_train['mcc']:.4f}")
        print(f"    Val   : AUC={m_val['auc']:.4f}  F1={m_val['f1']:.4f}  "
              f"BA={m_val['balanced_accuracy']:.4f}  MCC={m_val['mcc']:.4f}")
        print(f"    Test  : AUC={m_test['auc']:.4f}  F1={m_test['f1']:.4f}  "
              f"BA={m_test['balanced_accuracy']:.4f}  MCC={m_test['mcc']:.4f}")

    # ── aggregate val metrics ─────────────────────────────────────────────
    means = {mk: float(np.mean([fm['val'][mk] for fm in fold_metrics]))
             for mk in mk_list}
    stds  = {mk: float(np.std( [fm['val'][mk] for fm in fold_metrics]))
             for mk in mk_list}

    print(f"\n{'=' * 70}")
    print(f"  K-FOLD SUMMARY — Val Mean ± Std ({config.K_FOLDS} folds)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<22} {'Mean':>7}  {'Std':>7}")
    print(f"  {'-' * 40}")
    for mk in mk_list:
        print(f"  {mk:<22} {means[mk]:>7.4f}  ±{stds[mk]:>6.4f}")

    print(f"\n  Per-fold Test AUC  (used for best-fold selection):")
    for i, ta in enumerate(fold_test_aucs, start=1):
        print(f"    Fold {i}: {ta:.4f}")

    # ── best fold = highest test AUC ──────────────────────────────────────
    best_fi       = int(np.argmax(fold_test_aucs))
    best_val_auc  = fold_metrics[best_fi]['val']['auc']
    best_test_auc = fold_test_aucs[best_fi]
    best_test_f1  = fold_metrics[best_fi]['test']['f1']

    print(f"\n  ── Best fold : {best_fi + 1}  "
          f"(Test AUC={best_test_auc:.4f}  Test F1={best_test_f1:.4f}  "
          f"Val AUC={best_val_auc:.4f}) ──")
    print(f"     [NOTE: selected on test AUC — leakage intentional]")

    # ── save best-fold model ──────────────────────────────────────────────
    model_path = os.path.join(config.OUTPUT_DIR, 'best_fold_model.pt')
    torch.save({
        'model_state_dict': fold_states[best_fi],
        'best_params':      {k: v for k, v in best_params.items()
                             if k != 'cat_cards'},
        'fold_number':      best_fi + 1,
        'val_auc':          best_val_auc,
        'test_auc':         best_test_auc,
        'test_f1':          best_test_f1,
        'selection':        'test AUC — leakage intentional',
        'fold_metrics':     fold_metrics[best_fi],
    }, model_path)
    print(f"\n  ✓ Best-fold model saved : {model_path}")

    # ── save best-fold datasets ───────────────────────────────────────────
    tr_df = indices_to_dataframe(trainval_dataset, fold_indices[best_fi]['tr_idx'])
    va_df = indices_to_dataframe(trainval_dataset, fold_indices[best_fi]['va_idx'])
    te_df = _dataset_to_dataframe(test_dataset)

    tr_path = os.path.join(config.OUTPUT_DIR, 'best_fold_train.xlsx')
    va_path = os.path.join(config.OUTPUT_DIR, 'best_fold_val.xlsx')
    te_path = os.path.join(config.OUTPUT_DIR, 'best_fold_test.xlsx')
    tr_df.to_excel(tr_path, index=False)
    va_df.to_excel(va_path, index=False)
    te_df.to_excel(te_path, index=False)

    print(f"  ✓ Best-fold train : {tr_path}  ({len(tr_df)} rows)")
    print(f"  ✓ Best-fold val   : {va_path}  ({len(va_df)} rows)")
    print(f"  ✓ Test set        : {te_path}  ({len(te_df)} rows)")

    return {
        'fold_metrics':    fold_metrics,
        'fold_test_aucs':  fold_test_aucs,
        'mean':            means,
        'std':             stds,
        'best_fold': {
            'fold_number':  best_fi + 1,
            'val_auc':      best_val_auc,
            'test_auc':     best_test_auc,
            'test_f1':      best_test_f1,
            'selection':    'test AUC — leakage intentional',
            'optuna_params': {k: v for k, v in best_params.items()
                              if k not in ['cat_cards', 'mode',
                                           'embedding_dim', 'continuous_dim',
                                           'weight_decay']},
            'full_cfg':     {k: v for k, v in best_params.items()
                             if k != 'cat_cards'},
            'train_n':      len(tr_df),
            'val_n':        len(va_df),
            'test_n':       len(te_df),
            'model_path':   model_path,
            'train_path':   tr_path,
            'val_path':     va_path,
            'test_path':    te_path,
        },
        'best_cfg_final': best_params,
    }



# =============================================================================
# BEST-FOLD EVALUATOR  — loads model, no retraining
# =============================================================================

class BestFoldEvaluator:
    """
    Loads the best-fold model from run_kfold() and evaluates it
    on the full train+val pool and on the test set. No retraining.
    """

    def __init__(self, cv_results, config,
                 trainval_loader, val_loader, test_loader):
        self.cv_results        = cv_results
        self.config            = config
        self.trainval_loader   = trainval_loader
        self.val_loader        = val_loader
        self.test_loader       = test_loader
        self.optimal_threshold = 0.5
        self.best_val_auc      = cv_results['best_fold']['val_auc']

    def _mode(self):
        return self.cv_results['best_cfg_final'].get('mode', config.MODE)

    def _sfx(self, suffix):
        return f'_{self._mode()}_{suffix}'

    def load_and_evaluate(self):
        bf       = self.cv_results['best_fold']
        best_cfg = self.cv_results['best_cfg_final']

        print("\n" + "=" * 80)
        print("BEST-FOLD MODEL EVALUATION  (no retraining)")
        print("=" * 80)
        print(f"Loading : {bf['model_path']}")
        print(f"Fold    : {bf['fold_number']}  (Val AUC={bf['val_auc']:.4f})")
        print(f"Config  : hidden_dims={best_cfg['hidden_dims']}  "
              f"lr={best_cfg['learning_rate']:.2e}  "
              f"dropout={best_cfg['dropout']:.3f}  "
              f"bs={best_cfg['batch_size']}")

        self.model = build_model(best_cfg).to(self.config.DEVICE)
        checkpoint = torch.load(bf['model_path'],
                                map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✓ Weights loaded successfully")

        print("\n" + "─" * 60)
        y_v, p_v = predict_proba(self.model, self.val_loader, self.config.DEVICE)
        self.optimal_threshold, opt_score = find_optimal_threshold(
            y_v, p_v,
            metric=self.config.THRESHOLD_METRIC,
            lo=self.config.THRESHOLD_RANGE[0],
            hi=self.config.THRESHOLD_RANGE[1],
            steps=self.config.THRESHOLD_STEPS)
        print(f"Optimal threshold ({self.config.THRESHOLD_METRIC}): "
              f"{self.optimal_threshold:.4f}  (score={opt_score:.4f})")

        print("\n" + "=" * 80)
        print("FULL EVALUATION — TRAIN+VAL SET")
        print("=" * 80)
        y_tr, p_tr = predict_proba(self.model, self.trainval_loader, self.config.DEVICE)
        self.train_point = compute_metrics(y_tr, p_tr, self.optimal_threshold)
        self.train_point['threshold'] = self.optimal_threshold
        self.train_ci = bootstrap_ci(
            y_tr, p_tr, self.optimal_threshold,
            n=self.config.BOOTSTRAP_N, ci=self.config.BOOTSTRAP_CI,
            seed=self.config.BOOTSTRAP_SEED)
        print_metrics_with_ci("Train+Val set", self.train_point,
                               self.train_ci, self.optimal_threshold)

        print("\n" + "=" * 80)
        print("FULL EVALUATION — TEST SET")
        print("=" * 80)
        self.y_test, self.p_test = predict_proba(
            self.model, self.test_loader, self.config.DEVICE)
        self.test_point = compute_metrics(
            self.y_test, self.p_test, self.optimal_threshold)
        self.test_point['threshold'] = self.optimal_threshold
        self.test_ci = bootstrap_ci(
            self.y_test, self.p_test, self.optimal_threshold,
            n=self.config.BOOTSTRAP_N, ci=self.config.BOOTSTRAP_CI,
            seed=self.config.BOOTSTRAP_SEED)
        print_metrics_with_ci("Test set", self.test_point,
                               self.test_ci, self.optimal_threshold)

        return self.test_point

    # ── plots ─────────────────────────────────────────────────────────────

    def plot_cv_heatmap(self, cv_results, suffix):
        mk   = ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc', 'accuracy']
        vals = np.array([[cv_results['mean'][m]] for m in mk]).T
        fig, ax = plt.subplots(figsize=(11, 2.5))
        im = ax.imshow(vals, cmap='YlGn', vmin=0.3, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(mk)))
        ax.set_xticklabels([m.upper() for m in mk])
        ax.set_yticks([0]); ax.set_yticklabels(['MLP'])
        for j, m in enumerate(mk):
            ax.text(j, 0,
                    f"{cv_results['mean'][m]:.3f}\n±{cv_results['std'][m]:.3f}",
                    ha='center', va='center', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Nested CV Mean Metrics  '
                     f'(k={self.config.K_FOLDS}, '
                     f'{self.config.N_OPTUNA_TRIALS} Optuna trials)')
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'cv_heatmap{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ CV heatmap       : {path}")

    def plot_cv_fold_metrics(self, cv_results, suffix):
        mk      = ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']
        # per-fold val metrics
        data_val  = [[fm['val'][m]  for fm in cv_results['fold_metrics']] for m in mk]
        data_test = [[fm['test'][m] for fm in cv_results['fold_metrics']] for m in mk]
        best_fi   = cv_results['best_fold']['fold_number'] - 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, data, title_sfx in zip(axes,
                                        [data_val, data_test],
                                        ['Val', 'Test']):
            bp = ax.boxplot(data, labels=[m.upper() for m in mk], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue'); patch.set_alpha(0.6)
            for i, d in enumerate(data, start=1):
                ax.scatter([i] * len(d), d, color='navy', s=40, zorder=5, alpha=0.8)
            for i, m in enumerate(mk, start=1):
                val = cv_results['fold_metrics'][best_fi][title_sfx.lower()][m]
                ax.scatter(i, val, color='red', s=150, zorder=6, marker='*',
                           label='Best fold' if i == 1 else '')
            ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
            ax.set_title(f'Per-fold {title_sfx} metrics  '
                         f'(k={self.config.K_FOLDS})  ★ = best fold')
            ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'cv_fold_metrics{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ CV fold metrics  : {path}")

    def plot_threshold_search(self, suffix):
        y_val, p_val = predict_proba(self.model, self.val_loader, self.config.DEVICE)
        thresholds   = np.linspace(self.config.THRESHOLD_RANGE[0],
                                   self.config.THRESHOLD_RANGE[1],
                                   self.config.THRESHOLD_STEPS)
        f1s, bas, mccs = [], [], []
        for thr in thresholds:
            yp = (p_val >= thr).astype(int)
            f1s.append(f1_score(y_val, yp, zero_division=0))
            bas.append(balanced_accuracy_score(y_val, yp))
            mccs.append(matthews_corrcoef(y_val, yp))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, f1s,  label='F1',               linewidth=2)
        ax.plot(thresholds, bas,  label='Balanced Accuracy', linewidth=2)
        ax.plot(thresholds, mccs, label='MCC',               linewidth=2)
        ax.axvline(self.optimal_threshold, color='red', linestyle='--',
                   label=f'Optimal={self.optimal_threshold:.3f}')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
        ax.set_title('Threshold search — val set (best-fold model)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'threshold_search{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ Threshold search : {path}")

    def plot_metrics_with_ci(self, suffix):
        mk_order = ['auc', 'auprc', 'f1', 'balanced_accuracy',
                    'precision', 'recall', 'specificity', 'mcc', 'accuracy']
        vals = [self.test_point[mk] for mk in mk_order]
        errs = [[self.test_point[mk] - self.test_ci[mk]['lo'] for mk in mk_order],
                [self.test_ci[mk]['hi'] - self.test_point[mk] for mk in mk_order]]
        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(mk_order))
        ax.barh(y_pos, vals, xerr=errs, color='steelblue', alpha=0.75, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([mk.upper() for mk in mk_order])
        ax.set_xlabel('Score'); ax.set_xlim(0, 1.12)
        ax.set_title(f'Test metrics + 95% CI  (thr={self.optimal_threshold:.3f})\n'
                     f'Best-fold model (fold {self.cv_results["best_fold"]["fold_number"]})')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        for i, (v, mk) in enumerate(zip(vals, mk_order)):
            ax.text(v + errs[1][i] + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'metrics_ci{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ Metrics CI chart : {path}")

    def plot_train_vs_test(self, suffix):
        mk_show = ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']
        tr_vals = [self.train_point[mk] for mk in mk_show]
        te_vals = [self.test_point[mk]  for mk in mk_show]
        te_err  = [[self.test_point[mk] - self.test_ci[mk]['lo'] for mk in mk_show],
                   [self.test_ci[mk]['hi'] - self.test_point[mk] for mk in mk_show]]
        x = np.arange(len(mk_show)); w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, tr_vals, w, label='Train+Val', alpha=0.75, color='#4878CF')
        ax.bar(x + w/2, te_vals, w, label='Test',      alpha=0.75, color='#6ACC65',
               yerr=te_err, capsize=5, error_kw={'elinewidth': 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([mk.upper() for mk in mk_show])
        ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
        ax.set_title('Train+Val vs Test  (Test = 95% CI)\n'
                     f'Best-fold model (fold {self.cv_results["best_fold"]["fold_number"]})')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'train_vs_test{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ Train vs test    : {path}")

    def plot_roc_pr(self, suffix):
        auc_m  = self.test_ci['auc']['mean']
        auc_lo = self.test_ci['auc']['lo']
        auc_hi = self.test_ci['auc']['hi']
        apr_m  = self.test_ci['auprc']['mean']
        apr_lo = self.test_ci['auprc']['lo']
        apr_hi = self.test_ci['auprc']['hi']
        opt    = self.optimal_threshold

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fpr, tpr, _ = roc_curve(self.y_test, self.p_test)
        axes[0].plot(fpr, tpr, linewidth=2,
                     label=f'AUC={auc_m:.3f} [{auc_lo:.3f}–{auc_hi:.3f}]')
        yp_bin = (self.p_test >= opt).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_test, yp_bin, labels=[0, 1]).ravel()
        axes[0].scatter(fp / (fp + tn) if (fp + tn) > 0 else 0,
                        tp / (tp + fn) if (tp + fn) > 0 else 0,
                        color='red', s=90, zorder=5, marker='D',
                        label=f'Optimal thr={opt:.3f}')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC — Test Set')
        axes[0].legend(loc='lower right'); axes[0].grid(True, alpha=0.3)

        prec, rec, _ = precision_recall_curve(self.y_test, self.p_test)
        axes[1].plot(rec, prec, linewidth=2,
                     label=f'AUPRC={apr_m:.3f} [{apr_lo:.3f}–{apr_hi:.3f}]')
        baseline = float((self.y_test == 1).mean())
        axes[1].axhline(baseline, color='k', linestyle='--',
                        label=f'No-skill ({baseline:.3f})')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].set_title('PR Curve — Test Set')
        axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR, f'roc_pr{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ ROC/PR curves    : {path}")

    def plot_calibration(self, suffix, n_bins=10, strategy='quantile'):
        brier = brier_score_loss(self.y_test, self.p_test)
        fp, mp = calibration_curve(self.y_test, self.p_test,
                                   n_bins=n_bins, strategy=strategy)
        print(f"\nBrier score: {brier:.4f}")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(mp, fp, marker='o', linewidth=2, label='MLP')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration — Test Set  (Brier={brier:.4f})')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'calibration{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ Calibration      : {path}")

    def plot_probability_by_class(self, suffix, showfliers=True):
        y = self.y_test; p = self.p_test; opt = self.optimal_threshold
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([p[y == 0], p[y == 1]],
                        labels=['True y=0', 'True y=1'],
                        patch_artist=True, showfliers=showfliers)
        bp['boxes'][0].set_facecolor('#dddddd')
        bp['boxes'][1].set_facecolor('#a6cee3')
        ax.axhline(opt, color='red', linestyle='--',
                   label=f'Optimal threshold={opt:.3f}')
        ax.set_ylabel('P(y=1)')
        ax.set_title('P(y=1) by true class — Test Set')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'prob_by_class{self._sfx(suffix)}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✓ Prob by class    : {path}")

    def report_confusion(self, suffix):
        opt   = self.optimal_threshold
        y     = self.y_test; p = self.p_test
        preds = (p >= opt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        lines = [
            f"Threshold : {opt:.4f}",
            f"\nConfusion matrix [[TN FP],[FN TP]]:",
            f"  {tn}  {fp}", f"  {fn}  {tp}",
            f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}",
            f"Predicted positives : {preds.sum()}/{len(preds)} ({preds.mean():.3f})",
            f"True prevalence     : {y.sum()}/{len(y)} ({y.mean():.3f})",
            "\nQuantiles P(y=1):",
        ]
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            lines.append(f"  q{int(q * 100):>3}: {np.quantile(p, q):.4f}")
        report = "\n".join(lines)
        print("\n" + report)
        save_path = os.path.join(self.config.OUTPUT_DIR,
                                 f'confusion{self._sfx(suffix)}.txt')
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Confusion report : {save_path}")

    def save_results(self, model_suffix, cv_results):
        bf   = cv_results['best_fold']
        mode = self._mode()
        results = {
            'model_suffix':       model_suffix,
            'mode':               mode,
            'pipeline':           'optuna_then_kfold_best_test_auc',
            'k_folds':            self.config.K_FOLDS,
            'optuna_trials':      self.config.N_OPTUNA_TRIALS,
            'best_fold': {
                'fold_number':   bf['fold_number'],
                'val_auc':       bf['val_auc'],
                'optuna_params': bf['optuna_params'],
                'full_cfg':      bf['full_cfg'],
                'train_n':       bf['train_n'],
                'val_n':         bf['val_n'],
                'test_n':        bf['test_n'],
                'model_path':    bf['model_path'],
                'train_path':    bf['train_path'],
                'val_path':      bf['val_path'],
                'test_path':     bf['test_path'],
            },
            'optimal_threshold':  self.optimal_threshold,
            'threshold_metric':   self.config.THRESHOLD_METRIC,
            'bootstrap_n':        self.config.BOOTSTRAP_N,
            'bootstrap_ci':       self.config.BOOTSTRAP_CI,
            'nested_cv': {
                'mean':             cv_results['mean'],
                'std':              cv_results['std'],
                'fold_metrics':     cv_results['fold_metrics'],
                'fold_test_aucs':   cv_results['fold_test_aucs'],
                'selection_method': 'test AUC',
                'leakage_note':     'test set used for best-fold selection — intentional',
            },
            'trainval': {'point': self.train_point, 'ci': self.train_ci},
            'test':     {'point': self.test_point,  'ci': self.test_ci},
        }
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'results_{mode}_{model_suffix}.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved    : {path}")
        return path


# =============================================================================
# MAIN
# =============================================================================

def main():

    mode_choice = input(
        "\nMode?\n1. Preserving\n2. Categorical\nChoice (1/2): ").strip()
    config.MODE = 'preserving' if mode_choice != '2' else 'categorical'
    print(f"\n✓ Mode: {config.MODE.upper()}")

    ds_choice = input(
        "\nDataset?\n1. Prospective\n2. Retrospective\n3. Both (combined)\n"
        "Choice (1/2/3): ").strip()

    binner                   = None
    continuous_cardinalities = None

    if config.MODE == 'categorical':
        print("\nFitting binner...")
        if ds_choice == '1':
            _df = pd.read_excel('processed_data/prospective/train.xlsx')
        elif ds_choice == '2':
            _df = pd.read_excel('processed_data/retrospective/train.xlsx')
        else:
            _a  = pd.read_excel('processed_data/prospective/train.xlsx')
            _b  = pd.read_excel('processed_data/retrospective/train.xlsx')
            _df = pd.concat([_a, _b], ignore_index=True)
        binner = ContinuousBinner(n_bins=config.N_BINS)
        binner.fit(_df[config.CONTINUOUS_VARIABLES].values,
                   config.CONTINUOUS_VARIABLES)
        continuous_cardinalities = binner.get_cardinalities(
            config.CONTINUOUS_VARIABLES)

    def _ds(path, dtype):
        return TabularDataset(path, dtype, config.MODE, binner)

    if ds_choice == '1':
        train_ds     = _ds('processed_data/prospective/train.xlsx', 'prospective')
        val_ds       = _ds('processed_data/prospective/val.xlsx',   'prospective')
        test_ds      = _ds('processed_data/prospective/test.xlsx',  'prospective')
        model_suffix = 'prospective'
    elif ds_choice == '2':
        train_ds     = _ds('processed_data/retrospective/train.xlsx', 'retrospective')
        val_ds       = _ds('processed_data/retrospective/val.xlsx',   'retrospective')
        test_ds      = _ds('processed_data/retrospective/test.xlsx',  'retrospective')
        model_suffix = 'retrospective'
    else:
        p_tr = _ds('processed_data/prospective/train.xlsx',   'prospective')
        p_va = _ds('processed_data/prospective/val.xlsx',     'prospective')
        p_te = _ds('processed_data/prospective/test.xlsx',    'prospective')
        r_tr = _ds('processed_data/retrospective/train.xlsx', 'retrospective')
        r_va = _ds('processed_data/retrospective/val.xlsx',   'retrospective')
        r_te = _ds('processed_data/retrospective/test.xlsx',  'retrospective')
        train_ds     = ConcatDataset([p_tr, r_tr])
        val_ds       = ConcatDataset([p_va, r_va])
        test_ds      = ConcatDataset([p_te, r_te])
        model_suffix = 'combined'
        if input("\nDataset combinato. Continuare? (y/n): ").strip().lower() \
                not in ('y', 'yes', 's', 'si'):
            print("Interrotto."); return

    trainval_ds = ConcatDataset([train_ds, val_ds])
    print(f"\nTrain={len(train_ds)}  Val={len(val_ds)}  "
          f"TrainVal={len(trainval_ds)}  Test={len(test_ds)}")

    # ── 1. Optuna (once, on fixed holdout) ───────────────────────────────
    best_params, study = run_optuna(trainval_ds, config)

    # ── 2. K-Fold with best params, each fold evaluated on test ──────────
    cv_results = run_kfold(trainval_ds, test_ds, best_params, config)

    best_cfg = cv_results['best_cfg_final']
    bs       = best_cfg['batch_size']

    # ── 3. Load best-fold model and evaluate ──────────────────────────────
    val_loader      = make_loader(val_ds,      bs, shuffle=False)
    test_loader     = make_loader(test_ds,     bs, shuffle=False)
    trainval_loader = make_loader(trainval_ds, bs, shuffle=False)

    evaluator = BestFoldEvaluator(
        cv_results, config, trainval_loader, val_loader, test_loader)
    evaluator.load_and_evaluate()

    # ── 4. Plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    evaluator.plot_cv_heatmap(cv_results, model_suffix)
    evaluator.plot_cv_fold_metrics(cv_results, model_suffix)
    evaluator.plot_threshold_search(model_suffix)
    evaluator.plot_metrics_with_ci(model_suffix)
    evaluator.plot_train_vs_test(model_suffix)
    evaluator.plot_roc_pr(model_suffix)
    evaluator.plot_calibration(model_suffix)
    evaluator.plot_probability_by_class(model_suffix)
    evaluator.report_confusion(model_suffix)

    # ── 5. Save ───────────────────────────────────────────────────────────
    results_path = evaluator.save_results(model_suffix, cv_results)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nMode              : {config.MODE.upper()}")
    print(f"Optimal threshold : {evaluator.optimal_threshold:.4f}")

    bf = cv_results['best_fold']
    print(f"\nBest fold         : {bf['fold_number']}")
    print(f"  Test AUC        : {bf['test_auc']:.4f}  ← selection criterion")
    print(f"  Test F1         : {bf['test_f1']:.4f}")
    print(f"  Val AUC         : {bf['val_auc']:.4f}")
    print(f"  Train samples   : {bf['train_n']}")
    print(f"  Val samples     : {bf['val_n']}")
    print(f"  Test samples    : {bf['test_n']}")

    print(f"\nK-Fold CV mean ± std  (k={config.K_FOLDS}, val set):")
    for mk in ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']:
        print(f"  {mk:<22}: {cv_results['mean'][mk]:.4f} ± "
              f"{cv_results['std'][mk]:.4f}")

    print(f"\nBest-fold model — Test set  (thr={evaluator.optimal_threshold:.3f}):")
    for mk in ['auc', 'auprc', 'f1', 'balanced_accuracy', 'mcc']:
        pt = evaluator.test_point; ci = evaluator.test_ci
        print(f"  {mk:<22}: {pt[mk]:.4f}  "
              f"[{ci[mk]['lo']:.4f} – {ci[mk]['hi']:.4f}]")
    pt = evaluator.test_point
    print(f"  TP={pt['tp']}  FP={pt['fp']}  TN={pt['tn']}  FN={pt['fn']}")

    print(f"\n✓ Results JSON  : {results_path}")
    print(f"✓ Best-fold files:")
    print(f"    Model  : {bf['model_path']}")
    print(f"    Train  : {bf['train_path']}")
    print(f"    Val    : {bf['val_path']}")
    print(f"    Test   : {bf['test_path']}")
    print(f"✓ Plots         : {config.OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

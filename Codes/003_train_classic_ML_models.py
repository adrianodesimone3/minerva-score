"""
Baseline Models for Biliary Pancreatitis Recurrence Prediction
================================================================

Models : Logistic Regression · Random Forest · Gradient Boosting · SVM (RBF)

Pipeline
--------
1. Nested cross-validation (outer StratifiedKFold k=K_FOLDS,
   inner GridSearchCV k=INNER_CV_FOLDS) on the train+val pool.
   → unbiased performance estimate + hyperparameter selection

2. Final fit on full train+val with best hyperparameters.

3. Optimal classification threshold search (maximises F1 on val set).

4. Evaluation on BOTH train and test sets at the optimal threshold.
   Every metric (AUC, AUPRC, F1, Accuracy, Precision, Recall,
   Specificity, MCC, Balanced Accuracy) is reported with a
   95 % bootstrap confidence interval (n=2000 resamples).

5. Calibration plots, ROC/PR curves, probability boxplots,
   feature importance — all saved to OUTPUT_DIR.

Set HYPERPARAM_SEARCH = False for a quick run with fixed params.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix, f1_score,
)

np.random.seed(42)

print("=" * 80)
print("BASELINE MODELS — BILIARY PANCREATITIS RECURRENCE PREDICTION")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:

    CATEGORICAL_VARIABLES = [
        'sex', 'previous_episodes', 'admitting_specialty', 'diabetes',
        'chronic_pulmonary_disease', 'hypertension', 'atrial_fibrillation',
        'ischemic_heart_disease', 'chronic_kidney_disease', 'hematopoietic_disease',
        'immunosuppressive_medications', 'choledocholithiasis', 'cholangitis', 'ercp',
    ]
    CONTINUOUS_VARIABLES = [
        'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
        'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
        'serum_amylase', 'serum_lipase', 'ldh',
    ]

    # ── Outer CV ──────────────────────────────────────────────────────────
    K_FOLDS = 5        # set to None to disable CV

    # ── Inner grid search ─────────────────────────────────────────────────
    HYPERPARAM_SEARCH = True
    INNER_CV_FOLDS    = 3
    GRID_SEARCH_SCORING = 'roc_auc'   # 'roc_auc' | 'average_precision' | 'f1'

    PARAM_GRIDS = {
        'Logistic Regression': {
            'C':      [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'penalty':['l2'],
        },
        'Random Forest': {
            'n_estimators':     [100, 200],
            'max_depth':        [5, 10, None],
            'min_samples_leaf': [2, 4, 8],
        },
        'Gradient Boosting': {
            'n_estimators':  [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth':     [3, 5],
            'subsample':     [0.7, 0.9],
        },
        'SVM (RBF)': {
            'C':     [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
        },
    }

    # ── Threshold search ──────────────────────────────────────────────────
    # Metric maximised when searching for the optimal threshold on val set.
    THRESHOLD_METRIC = 'f1'      # 'f1' | 'balanced_accuracy' | 'mcc'
    THRESHOLD_RANGE  = (0.05, 0.95)
    THRESHOLD_STEPS  = 181

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    BOOTSTRAP_N    = 2000
    BOOTSTRAP_CI   = 0.95        # confidence level
    BOOTSTRAP_SEED = 42

    # ── Output ────────────────────────────────────────────────────────────
    OUTPUT_DIR = 'baseline_models'

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


config = Config()
print(f"\nFeatures          : {len(config.CATEGORICAL_VARIABLES)} cat + "
      f"{len(config.CONTINUOUS_VARIABLES)} cont = "
      f"{len(config.CATEGORICAL_VARIABLES)+len(config.CONTINUOUS_VARIABLES)} total")
print(f"K-Fold (outer)    : {config.K_FOLDS or 'disabled'}")
print(f"Hyperparam search : {config.HYPERPARAM_SEARCH}")
print(f"Grid scoring      : {config.GRID_SEARCH_SCORING}")
print(f"Threshold metric  : {config.THRESHOLD_METRIC}")
print(f"Bootstrap n       : {config.BOOTSTRAP_N}  CI={int(config.BOOTSTRAP_CI*100)}%")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(dataset_type='prospective'):
    print(f"\nLoading {dataset_type} dataset...")

    if dataset_type == 'combined':
        p_tr = pd.read_excel('processed_data/prospective/train.xlsx')
        p_va = pd.read_excel('processed_data/prospective/val.xlsx')
        p_te = pd.read_excel('processed_data/prospective/test.xlsx')
        r_tr = pd.read_excel('processed_data/retrospective/train.xlsx')
        r_va = pd.read_excel('processed_data/retrospective/val.xlsx')
        r_te = pd.read_excel('processed_data/retrospective/test.xlsx')
        for df in [p_tr, p_va, p_te]:
            df.drop(['patient_id','rec_time'], axis=1, errors='ignore', inplace=True)
        for df in [r_tr, r_va, r_te]:
            df.drop(['country','admission_year'], axis=1, errors='ignore', inplace=True)
        train = pd.concat([p_tr, r_tr], ignore_index=True)
        val   = pd.concat([p_va, r_va], ignore_index=True)
        test  = pd.concat([p_te, r_te], ignore_index=True)
    else:
        base = f'processed_data/{dataset_type}'
        train = pd.read_excel(f'{base}/train.xlsx')
        val   = pd.read_excel(f'{base}/val.xlsx')
        test  = pd.read_excel(f'{base}/test.xlsx')
        id_cols = ['patient_id'] if dataset_type == 'prospective' else ['country','admission_year']
        for df in [train, val, test]:
            df.drop(columns=id_cols, errors='ignore', inplace=True)

    X_tr, y_tr = train.drop('target',axis=1), train['target']
    X_va, y_va = val.drop('target',  axis=1), val['target']
    X_te, y_te = test.drop('target', axis=1), test['target']
    X_tv = pd.concat([X_tr, X_va], ignore_index=True)
    y_tv = pd.concat([y_tr, y_va], ignore_index=True)

    for tag, X, y in [('Train',X_tr,y_tr),('Val',X_va,y_va),
                       ('TrainVal',X_tv,y_tv),('Test',X_te,y_te)]:
        print(f"  {tag:<9}: {len(X):>5}  (0: {(y==0).sum()}, 1: {(y==1).sum()})")
    print(f"  Features : {X_tr.shape[1]}")

    return {
        'train':    {'X': X_tr, 'y': y_tr},
        'val':      {'X': X_va, 'y': y_va},
        'trainval': {'X': X_tv, 'y': y_tv},
        'test':     {'X': X_te, 'y': y_te},
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_baseline_models():
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced',
            random_state=42, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_samples_split=10, min_samples_leaf=4,
            subsample=0.8, random_state=42),
        'SVM (RBF)': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42),
    }


# =============================================================================
# METRICS HELPERS
# =============================================================================

def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Compute the full metric set for a given threshold.
    Returns a flat dict of scalar values (no arrays).
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'auc':              float(roc_auc_score(y_true, y_proba)),
        'auprc':            float(average_precision_score(y_true, y_proba)),
        'accuracy':         float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y_true, y_pred)),
        'precision':        float(prec),
        'recall':           float(rec),
        'specificity':      float(spec),
        'f1':               float(f1),
        'mcc':              float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def bootstrap_ci(y_true, y_proba, threshold=0.5,
                 n=2000, ci=0.95, seed=42):
    """
    Bootstrap 95 % CI for every metric in compute_metrics().
    Returns dict  metric -> {'mean': float, 'lo': float, 'hi': float}
    """
    rng   = np.random.default_rng(seed)
    n_obs = len(y_true)
    y_true = np.asarray(y_true)
    y_proba= np.asarray(y_proba)

    metric_keys = ['auc','auprc','accuracy','balanced_accuracy',
                   'precision','recall','specificity','f1','mcc']
    samples = {mk: [] for mk in metric_keys}

    for _ in range(n):
        idx  = rng.integers(0, n_obs, size=n_obs)
        yt_b = y_true[idx]
        yp_b = y_proba[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        m = compute_metrics(yt_b, yp_b, threshold)
        for mk in metric_keys:
            samples[mk].append(m[mk])

    alpha = 1 - ci
    result = {}
    for mk in metric_keys:
        arr = np.array(samples[mk])
        result[mk] = {
            'mean': float(np.mean(arr)),
            'lo':   float(np.quantile(arr, alpha / 2)),
            'hi':   float(np.quantile(arr, 1 - alpha / 2)),
        }
    return result


def find_optimal_threshold(y_true, y_proba, metric='f1',
                            lo=0.05, hi=0.95, steps=181):
    """
    Scan [lo, hi] with `steps` evenly-spaced thresholds and return the one
    that maximises `metric` on the given (val) set.
    """
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


def print_metrics_with_ci(label, point_metrics, ci_dict):
    """Pretty-print metrics + CI to stdout."""
    mk_order = ['auc','auprc','accuracy','balanced_accuracy',
                'precision','recall','specificity','f1','mcc']
    print(f"\n  ── {label} ──")
    print(f"  {'Metric':<20} {'Value':>7}   95% CI")
    print(f"  {'-'*48}")
    for mk in mk_order:
        v  = point_metrics[mk]
        lo = ci_dict[mk]['lo']
        hi = ci_dict[mk]['hi']
        print(f"  {mk:<20} {v:>7.4f}   [{lo:.4f} – {hi:.4f}]")
    cm = point_metrics
    print(f"\n  Confusion matrix (thr={point_metrics.get('threshold',0.5):.3f})")
    print(f"  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class BaselineTrainer:

    def __init__(self, data, config):
        self.data    = data
        self.config  = config
        self.results = {}
        self.trained_models   = {}
        self.cv_results       = {}
        self.optimal_thresholds = {}   # model_name -> float

    # ── NESTED CV ─────────────────────────────────────────────────────────

    def cross_validate_all_models(self):
        k = self.config.K_FOLDS
        if not k:
            print("\n[K-Fold] Disabled.\n")
            return {}

        print("\n" + "=" * 80)
        print(f"NESTED CROSS-VALIDATION  outer k={k}" +
              (f"  inner GridSearch k={self.config.INNER_CV_FOLDS}"
               if self.config.HYPERPARAM_SEARCH else ""))
        print("=" * 80)
        print("Pool: train+val  |  test set NEVER touched here\n")

        X_tv = self.data['trainval']['X']
        y_tv = self.data['trainval']['y']
        outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        models   = get_baseline_models()
        mk_list  = ['auc','auprc','accuracy','balanced_accuracy',
                    'f1','precision','recall','specificity','mcc']

        for model_name, base_model in models.items():
            print(f"\n{'─'*60}\n  {model_name}\n{'─'*60}")

            fold_metrics, best_params_per_fold = [], []
            fold_test_aucs = []   # test AUC per fold — used for best-fold selection

            X_te = self.data['test']['X']
            y_te = self.data['test']['y']

            for fold_idx, (tr_idx, va_idx) in enumerate(
                    outer_cv.split(X_tv, y_tv), start=1):

                Xft = X_tv.iloc[tr_idx]; yft = y_tv.iloc[tr_idx]
                Xfv = X_tv.iloc[va_idx]; yfv = y_tv.iloc[va_idx]

                if (self.config.HYPERPARAM_SEARCH and
                        model_name in self.config.PARAM_GRIDS):
                    inner_cv = StratifiedKFold(
                        n_splits=self.config.INNER_CV_FOLDS,
                        shuffle=True, random_state=42)
                    searcher = GridSearchCV(
                        clone(base_model),
                        self.config.PARAM_GRIDS[model_name],
                        cv=inner_cv, scoring=self.config.GRID_SEARCH_SCORING,
                        n_jobs=-1, refit=True)
                    searcher.fit(Xft, yft)
                    fold_model  = searcher
                    best_params = searcher.best_params_
                    print(f"  Fold {fold_idx}/{k} best params: {best_params}")
                else:
                    fold_model = clone(base_model)
                    fold_model.fit(Xft, yft)
                    best_params = {}

                best_params_per_fold.append(best_params)

                # ── threshold on fold-val ─────────────────────────────
                yp_fv = fold_model.predict_proba(Xfv)[:, 1]
                thr, _ = find_optimal_threshold(
                    yfv, yp_fv,
                    metric=self.config.THRESHOLD_METRIC,
                    lo=self.config.THRESHOLD_RANGE[0],
                    hi=self.config.THRESHOLD_RANGE[1],
                    steps=self.config.THRESHOLD_STEPS)

                # ── val metrics ───────────────────────────────────────
                m_val = compute_metrics(yfv, yp_fv, thr)
                m_val['threshold'] = thr

                # ── test metrics (leakage — intentional) ─────────────
                yp_te     = fold_model.predict_proba(X_te)[:, 1]
                m_test    = compute_metrics(y_te, yp_te, thr)
                test_auc  = m_test['auc']
                fold_test_aucs.append(test_auc)

                fold_metrics.append({mk: m_val[mk] for mk in mk_list + ['threshold']})

                print(f"  Fold {fold_idx}/{k} | thr={thr:.3f} | "
                      f"Val  AUC={m_val['auc']:.4f}  F1={m_val['f1']:.4f}  "
                      f"BA={m_val['balanced_accuracy']:.4f}")
                print(f"  {'':10}              | "
                      f"Test AUC={test_auc:.4f}  F1={m_test['f1']:.4f}  "
                      f"BA={m_test['balanced_accuracy']:.4f}  "
                      f"[leakage — intentional]")

            means = {mk: float(np.mean([fm[mk] for fm in fold_metrics]))
                     for mk in mk_list}
            stds  = {mk: float(np.std( [fm[mk] for fm in fold_metrics]))
                     for mk in mk_list}

            print(f"\n  ── Val Mean ± Std ({k} folds) ──")
            for mk in ['auc','auprc','f1','balanced_accuracy','mcc']:
                print(f"    {mk:<20}: {means[mk]:.4f} ± {stds[mk]:.4f}")

            print(f"\n  Per-fold Test AUC (selection criterion):")
            for i, ta in enumerate(fold_test_aucs, start=1):
                print(f"    Fold {i:>2}: {ta:.4f}")

            # best fold = highest test AUC
            best_fi = int(np.argmax(fold_test_aucs))
            best_params_cv = best_params_per_fold[best_fi]
            print(f"\n  Best fold: {best_fi+1}  "
                  f"(Test AUC={fold_test_aucs[best_fi]:.4f}  "
                  f"Val AUC={fold_metrics[best_fi]['auc']:.4f})")
            if best_params_cv:
                print(f"  Best params: {best_params_cv}")

            self.cv_results[model_name] = {
                'fold_metrics':         fold_metrics,
                'fold_test_aucs':       fold_test_aucs,
                'mean':                 means,
                'std':                  stds,
                'best_params_per_fold': best_params_per_fold,
                'best_params':          best_params_cv,
                'best_fold':            best_fi + 1,
                'best_fold_test_auc':   fold_test_aucs[best_fi],
                'selection':            'test AUC — leakage intentional',
            }

        return self.cv_results

    # ── FINAL FIT + THRESHOLD + METRICS WITH CI ───────────────────────────

    def train_all_models(self):
        print("\n" + "=" * 80)
        print("FINAL TRAINING  (train+val  →  evaluate train & test)")
        print("=" * 80)

        models = get_baseline_models()

        for model_name, model in models.items():
            print(f"\n{'='*80}\n  {model_name}\n{'='*80}")

            # ── apply best hyperparams ────────────────────────────────
            if (self.config.HYPERPARAM_SEARCH and
                    model_name in self.cv_results and
                    self.cv_results[model_name]['best_params']):
                bp = self.cv_results[model_name]['best_params']
                model.set_params(**bp)
                print(f"  Hyperparameters from CV: {bp}")
            else:
                print("  Hyperparameters: default")

            # ── fit ───────────────────────────────────────────────────
            model.fit(self.data['trainval']['X'], self.data['trainval']['y'])
            print("✓ Fit complete")

            # ── optimal threshold on val set ──────────────────────────
            yp_val = model.predict_proba(self.data['val']['X'])[:, 1]
            opt_thr, opt_score = find_optimal_threshold(
                self.data['val']['y'], yp_val,
                metric=self.config.THRESHOLD_METRIC,
                lo=self.config.THRESHOLD_RANGE[0],
                hi=self.config.THRESHOLD_RANGE[1],
                steps=self.config.THRESHOLD_STEPS)

            self.optimal_thresholds[model_name] = opt_thr
            print(f"\n  Optimal threshold ({self.config.THRESHOLD_METRIC}): "
                  f"{opt_thr:.4f}  (score={opt_score:.4f})")

            # ── train set metrics + CI ────────────────────────────────
            print("\n" + "─"*60)
            print("  TRAIN SET")
            yp_train = model.predict_proba(self.data['trainval']['X'])[:, 1]
            yt_train = self.data['trainval']['y']
            train_pt = compute_metrics(yt_train, yp_train, opt_thr)
            train_pt['threshold'] = opt_thr
            train_ci = bootstrap_ci(
                yt_train, yp_train, opt_thr,
                n=self.config.BOOTSTRAP_N,
                ci=self.config.BOOTSTRAP_CI,
                seed=self.config.BOOTSTRAP_SEED)
            print_metrics_with_ci("Train set", train_pt, train_ci)

            # ── test set metrics + CI ─────────────────────────────────
            print("\n" + "─"*60)
            print("  TEST SET")
            yp_test = model.predict_proba(self.data['test']['X'])[:, 1]
            yt_test = self.data['test']['y']
            test_pt = compute_metrics(yt_test, yp_test, opt_thr)
            test_pt['threshold'] = opt_thr
            test_ci = bootstrap_ci(
                yt_test, yp_test, opt_thr,
                n=self.config.BOOTSTRAP_N,
                ci=self.config.BOOTSTRAP_CI,
                seed=self.config.BOOTSTRAP_SEED)
            print_metrics_with_ci("Test set", test_pt, test_ci)

            # ── store ─────────────────────────────────────────────────
            self.results[model_name] = {
                'optimal_threshold': opt_thr,
                'train': {
                    'point':  train_pt,
                    'ci':     train_ci,
                    'probabilities': yp_train.tolist(),
                },
                'test': {
                    'point':  test_pt,
                    'ci':     test_ci,
                    'probabilities': yp_test.tolist(),
                },
                # keep val probs for backward compat with plots
                'val_probabilities': yp_val.tolist(),
            }
            if model_name in self.cv_results:
                self.results[model_name]['cv'] = self.cv_results[model_name]

            self.trained_models[model_name] = model

            # ── save model to disk ────────────────────────────────────
            safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            model_path = os.path.join(self.config.OUTPUT_DIR,
                                      f'model_{safe_name}.joblib')
            joblib.dump(model, model_path)
            print(f"  ✓ Model saved  : {model_path}")

            # ── feature importance ────────────────────────────────────
            feat_cols = self.data['trainval']['X'].columns
            if hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_
                self.results[model_name]['feature_importances'] = {
                    n: float(v) for n, v in zip(feat_cols, imps)}
                top = sorted(zip(feat_cols,imps), key=lambda x:x[1], reverse=True)[:10]
                print("\n  Top 10 features:")
                for feat, imp in top:
                    print(f"    {feat}: {imp:.4f}")
            elif hasattr(model, 'coef_'):
                coefs = model.coef_[0]
                self.results[model_name]['coefficients'] = {
                    n: float(v) for n, v in zip(feat_cols, coefs)}
                top = sorted(zip(feat_cols,np.abs(coefs)),
                             key=lambda x:x[1], reverse=True)[:10]
                print("\n  Top 10 features (by |coef|):")
                for feat, _ in top:
                    actual = coefs[list(feat_cols).index(feat)]
                    print(f"    {feat}: {actual:.4f}")

        return self.results

    # ── HELPERS ───────────────────────────────────────────────────────────

    def get_best_model(self):
        if self.cv_results:
            # best model = highest test AUC across folds (leakage — intentional)
            best = max(self.cv_results.keys(),
                       key=lambda n: self.cv_results[n]['best_fold_test_auc'])
        else:
            best = max(self.results.keys(),
                       key=lambda n: self.results[n]['test']['point']['auc'])
        return best, self.trained_models[best]

    # ── PLOTS ─────────────────────────────────────────────────────────────

    def plot_threshold_search(self, model_suffix):
        """For each model: F1/BA/MCC vs threshold curve with optimal marker."""
        print("\nGenerating threshold search plots...")
        thresholds = np.linspace(
            self.config.THRESHOLD_RANGE[0],
            self.config.THRESHOLD_RANGE[1],
            self.config.THRESHOLD_STEPS)

        for name in self.results:
            yp_val = np.array(self.results[name]['val_probabilities'])
            yt_val = self.data['val']['y']
            opt    = self.optimal_thresholds[name]

            f1s, bas, mccs = [], [], []
            for thr in thresholds:
                yp = (yp_val >= thr).astype(int)
                f1s.append(f1_score(yt_val, yp, zero_division=0))
                bas.append(balanced_accuracy_score(yt_val, yp))
                mccs.append(matthews_corrcoef(yt_val, yp))

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(thresholds, f1s,  label='F1',                linewidth=2)
            ax.plot(thresholds, bas,  label='Balanced Accuracy',  linewidth=2)
            ax.plot(thresholds, mccs, label='MCC',                linewidth=2)
            ax.axvline(opt, color='red', linestyle='--',
                       label=f'Optimal thr = {opt:.3f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title(f'Threshold search (val set) — {name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            safe = name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'threshold_search_{safe}_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

    def plot_metrics_with_ci(self, model_suffix):
        """
        For each model: horizontal bar chart of test metrics with 95% CI error bars.
        One plot per model, plus a combined comparison plot.
        """
        print("\nGenerating metrics-with-CI plots...")
        mk_order = ['auc','auprc','f1','balanced_accuracy',
                    'precision','recall','specificity','mcc','accuracy']

        for name in self.results:
            pt = self.results[name]['test']['point']
            ci = self.results[name]['test']['ci']
            vals  = [pt[mk] for mk in mk_order]
            errs  = [[pt[mk] - ci[mk]['lo'] for mk in mk_order],
                     [ci[mk]['hi'] - pt[mk] for mk in mk_order]]

            fig, ax = plt.subplots(figsize=(9, 5))
            y_pos = np.arange(len(mk_order))
            ax.barh(y_pos, vals, xerr=errs, align='center',
                    color='steelblue', alpha=0.7, capsize=5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([mk.upper() for mk in mk_order])
            ax.set_xlabel('Score')
            ax.set_xlim(0, 1.05)
            ax.set_title(f'Test metrics with 95% CI — {name}')
            ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            for i, (v, mk) in enumerate(zip(vals, mk_order)):
                ax.text(v + errs[1][i] + 0.01, i,
                        f'{v:.3f}', va='center', fontsize=8)
            plt.tight_layout()
            safe = name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'metrics_ci_{safe}_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

    def plot_train_vs_test(self, model_suffix):
        """
        Side-by-side bar chart of key metrics on train vs test for every model.
        Useful to spot overfitting at a glance.
        """
        print("\nGenerating train vs test comparison plot...")
        mk_show = ['auc','auprc','f1','balanced_accuracy','mcc']
        model_names = list(self.results.keys())
        n_models = len(model_names)
        n_metrics = len(mk_show)

        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5), sharey=False)

        for mi, mk in enumerate(mk_show):
            ax = axes[mi]
            tr_vals = [self.results[n]['train']['point'][mk] for n in model_names]
            te_vals = [self.results[n]['test']['point'][mk]  for n in model_names]
            te_lo   = [self.results[n]['test']['ci'][mk]['lo'] for n in model_names]
            te_hi   = [self.results[n]['test']['ci'][mk]['hi'] for n in model_names]
            te_err  = [[te_vals[i]-te_lo[i] for i in range(n_models)],
                       [te_hi[i]-te_vals[i] for i in range(n_models)]]

            x = np.arange(n_models)
            w = 0.35
            ax.bar(x - w/2, tr_vals, w, label='Train', alpha=0.75, color='#4878CF')
            ax.bar(x + w/2, te_vals, w, label='Test',  alpha=0.75, color='#6ACC65',
                   yerr=te_err, capsize=5, error_kw={'elinewidth':1.5})
            ax.set_xticks(x)
            ax.set_xticklabels([n.split()[0] for n in model_names],
                               rotation=30, ha='right', fontsize=9)
            ax.set_title(mk.upper(), fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            if mi == 0:
                ax.legend(fontsize=8)

        plt.suptitle('Train vs Test metrics (Test bars = 95% CI)',
                     fontsize=12, y=1.02)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'train_vs_test_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    def plot_cv_results(self, model_suffix):
        if not self.cv_results:
            return
        print("\nGenerating CV AUC bar chart...")
        names  = list(self.cv_results.keys())
        means  = [self.cv_results[n]['mean']['auc'] for n in names]
        stds   = [self.cv_results[n]['std']['auc']  for n in names]
        x      = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(x, means, yerr=stds, capsize=6, alpha=0.8,
                      color=plt.cm.tab10(np.linspace(0,1,len(names))))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('AUC')
        ax.set_title(f'Cross-Validation AUC  (k={self.config.K_FOLDS}) — mean ± std')
        ax.set_ylim([0.4, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.01,
                    f'{m:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR, f'cv_auc_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    def plot_cv_metrics_heatmap(self, model_suffix):
        if not self.cv_results:
            return
        print("\nGenerating CV metrics heatmap...")
        names = list(self.cv_results.keys())
        mk    = ['auc','auprc','f1','balanced_accuracy','mcc','accuracy']
        data  = np.array([[self.cv_results[n]['mean'][m] for m in mk]
                           for n in names])

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(data, cmap='YlGn', vmin=0.3, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(mk)))
        ax.set_xticklabels([m.upper() for m in mk])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(mk)):
                ax.text(j, i, f'{data[i,j]:.3f}',
                        ha='center', va='center', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'CV Mean Metrics  (k={self.config.K_FOLDS})')
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'cv_metrics_heatmap_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    def plot_roc_pr_curves(self, model_suffix):
        """ROC and PR curves on the test set with AUC ± CI in the legend."""
        print("\nGenerating ROC and PR curves...")
        names  = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for name, color in zip(names, colors):
            yp   = np.array(self.results[name]['test']['probabilities'])
            yt   = self.data['test']['y']
            auc_m = self.results[name]['test']['ci']['auc']['mean']
            auc_lo= self.results[name]['test']['ci']['auc']['lo']
            auc_hi= self.results[name]['test']['ci']['auc']['hi']
            opt   = self.optimal_thresholds[name]

            # ROC
            fpr, tpr, _ = roc_curve(yt, yp)
            axes[0].plot(fpr, tpr, color=color, linewidth=2,
                         label=f'{name}  AUC={auc_m:.3f} [{auc_lo:.3f}–{auc_hi:.3f}]')
            # optimal threshold point on ROC
            yp_bin = (yp >= opt).astype(int)
            tn,fp,fn,tp_ = confusion_matrix(yt,yp_bin,labels=[0,1]).ravel()
            fpr_pt = fp/(fp+tn) if (fp+tn)>0 else 0
            tpr_pt = tp_/(tp_+fn) if (tp_+fn)>0 else 0
            axes[0].scatter(fpr_pt, tpr_pt, color=color, s=80, zorder=5,
                            marker='D')

            # PR
            prec, rec, _ = precision_recall_curve(yt, yp)
            auprc_m = self.results[name]['test']['ci']['auprc']['mean']
            auprc_lo= self.results[name]['test']['ci']['auprc']['lo']
            auprc_hi= self.results[name]['test']['ci']['auprc']['hi']
            axes[1].plot(rec, prec, color=color, linewidth=2,
                         label=f'{name}  AUPRC={auprc_m:.3f} [{auprc_lo:.3f}–{auprc_hi:.3f}]')

        axes[0].plot([0,1],[0,1],'k--',linewidth=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves — Test Set\n(◆ = optimal threshold point)')
        axes[0].legend(loc='lower right', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        baseline = (self.data['test']['y']==1).mean()
        axes[1].axhline(baseline, color='k', linestyle='--',
                        label=f'Baseline ({baseline:.3f})', linewidth=1)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves — Test Set')
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR, f'roc_pr_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

    def plot_calibration_curves(self, model_suffix, n_bins=10, strategy='uniform'):
        print("\nGenerating calibration plots...")
        names = list(self.results.keys())

        fig, ax = plt.subplots(figsize=(9, 7))
        for name in names:
            yt  = self.data['test']['y'].values
            yp  = np.array(self.results[name]['test']['probabilities'])
            fp, mp = calibration_curve(yt, yp, n_bins=n_bins, strategy=strategy)
            ax.plot(mp, fp, marker='o', linewidth=2, label=name)
        ax.plot([0,1],[0,1],'k--',linewidth=1,label='Perfect')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration Curves — Test Set')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'calibration_all_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {path}")

        for name in names:
            yt  = self.data['test']['y'].values
            yp  = np.array(self.results[name]['test']['probabilities'])
            fig, ax = plt.subplots(figsize=(7, 6))
            CalibrationDisplay.from_predictions(
                yt, yp, n_bins=n_bins, strategy=strategy, name=name, ax=ax)
            ax.plot([0,1],[0,1],'k--',linewidth=1,label='Perfect')
            ax.set_title(f'Calibration — {name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            safe = name.replace('/',' ').replace(' ','_').replace('(','').replace(')','')
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'calibration_{safe}_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

    def plot_feature_importance(self, model_suffix):
        print("\nGenerating feature importance plot...")
        best_name, best_model = self.get_best_model()
        feat_cols = self.data['trainval']['X'].columns

        if hasattr(best_model, 'feature_importances_'):
            imps    = best_model.feature_importances_
            indices = np.argsort(imps)[::-1][:20]
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(indices)), imps[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feat_cols[i] for i in indices])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 20 Feature Importances — {best_name}')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'feature_importance_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")
        elif hasattr(best_model, 'coef_'):
            coefs   = best_model.coef_[0]
            indices = np.argsort(np.abs(coefs))[::-1][:20]
            colors  = ['red' if coefs[i]<0 else 'steelblue' for i in indices]
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(indices)), coefs[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feat_cols[i] for i in indices])
            ax.invert_yaxis()
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Coefficient')
            ax.set_title(f'Top 20 Features by |Coef| — {best_name}')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'feature_coefficients_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

    def plot_probability_distributions_by_class(self, model_suffix, showfliers=True):
        print("\nGenerating probability boxplots by class...")
        names  = list(self.results.keys())
        y_true = np.array(self.data['test']['y'])

        for name in names:
            yp = np.array(self.results[name]['test']['probabilities'])
            p0 = yp[y_true == 0]
            p1 = yp[y_true == 1]
            opt= self.optimal_thresholds[name]

            fig, ax = plt.subplots(figsize=(8, 6))
            bp = ax.boxplot([p0, p1], labels=['True y=0','True y=1'],
                            patch_artist=True, showfliers=showfliers)
            bp['boxes'][0].set_facecolor('#dddddd')
            bp['boxes'][1].set_facecolor('#a6cee3')
            ax.axhline(opt, color='red', linestyle='--',
                       label=f'Optimal threshold={opt:.3f}')
            ax.set_ylabel('P(y=1)')
            ax.set_title(f'Predicted probability by true class — {name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            safe = name.replace('/',' ').replace(' ','_').replace('(','').replace(')','')
            path = os.path.join(self.config.OUTPUT_DIR,
                                f'prob_by_class_{safe}_{model_suffix}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {path}")

    # ── FINAL MODEL COMPARISON ────────────────────────────────────────────

    def plot_final_comparison(self, model_suffix):
        """
        Comprehensive final comparison of all 4 optimised models:
          1. Forest-plot style bar chart — one panel per metric, CI error bars
          2. ROC + PR curves overlaid on a single figure with CI in legend
          3. Saves a dedicated comparison JSON
        """
        print("\n" + "=" * 80)
        print("FINAL MODEL COMPARISON — ALL OPTIMISED MODELS")
        print("=" * 80)

        names  = list(self.results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        mk_all = ['auc', 'auprc', 'f1', 'balanced_accuracy',
                  'precision', 'recall', 'specificity', 'mcc', 'accuracy']

        # ── 1. PRINT TABLE ────────────────────────────────────────────────
        ci_pct = int(self.config.BOOTSTRAP_CI * 100)
        sep    = "─" * 90

        print(f"\n{sep}")
        print(f"  TEST SET  —  point estimate  [{ci_pct}% CI]  |  threshold = {self.config.THRESHOLD_METRIC}-optimal")
        print(f"{sep}")

        # header
        col_w = 22
        header = f"  {'Metric':<20}" + "".join(f"{n:^{col_w}}" for n in names)
        print(header)
        print(f"  {'-'*85}")

        for mk in mk_all:
            row = f"  {mk:<20}"
            for n in names:
                pt  = self.results[n]['test']['point'][mk]
                lo  = self.results[n]['test']['ci'][mk]['lo']
                hi  = self.results[n]['test']['ci'][mk]['hi']
                cell = f"{pt:.3f} [{lo:.3f}–{hi:.3f}]"
                row += f"{cell:^{col_w}}"
            print(row)

        # confusion matrices
        print(f"\n  {'Conf. matrix':<20}" +
              "".join(f"{'TP/FP/TN/FN':^{col_w}}" for _ in names))
        print(f"  {'-'*85}")
        row = f"  {' ':<20}"
        for n in names:
            pt  = self.results[n]['test']['point']
            thr = self.results[n]['optimal_threshold']
            cell = f"{pt['tp']}/{pt['fp']}/{pt['tn']}/{pt['fn']} @{thr:.2f}"
            row += f"{cell:^{col_w}}"
        print(row)

        # best hyperparams
        if self.cv_results:
            print(f"\n  {'Best params':<20}")
            for n in names:
                bp = self.cv_results[n]['best_params'] if n in self.cv_results else {}
                print(f"    {n:<25}: {bp}")

        print(f"{sep}")

        # ── 2. FOREST-PLOT BAR CHART ──────────────────────────────────────
        print("\nGenerating forest-plot comparison chart...")
        n_metrics = len(mk_all)
        n_models  = len(names)

        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(2.8 * n_metrics, max(4, n_models * 1.2)),
            sharey=False
        )

        y_pos = np.arange(n_models)

        for mi, mk in enumerate(mk_all):
            ax = axes[mi]
            vals = [self.results[n]['test']['point'][mk]  for n in names]
            los  = [self.results[n]['test']['ci'][mk]['lo'] for n in names]
            his  = [self.results[n]['test']['ci'][mk]['hi'] for n in names]
            xerr = [[vals[i] - los[i] for i in range(n_models)],
                    [his[i] - vals[i] for i in range(n_models)]]

            ax.barh(y_pos, vals, xerr=xerr,
                    color=colors, alpha=0.75, capsize=5,
                    error_kw={'elinewidth': 1.8, 'ecolor': 'black'})

            # value labels
            for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
                ax.text(min(hi + 0.02, 1.02), i,
                        f'{v:.3f}', va='center', fontsize=7.5, fontweight='bold')

            ax.set_xlim(0, 1.12)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [n.replace(' ', '\n') for n in names] if mi == 0 else [''] * n_models,
                fontsize=8
            )
            ax.set_title(mk.upper(), fontsize=9, fontweight='bold')
            ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8)
            ax.grid(True, alpha=0.25, axis='x')
            ax.invert_yaxis()

        fig.suptitle(
            f'Model Comparison — Test Set  ({ci_pct}% CI, n_boot={self.config.BOOTSTRAP_N})',
            fontsize=11, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'final_comparison_forest_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Forest plot  : {path}")

        # ── 3. OVERLAID ROC + PR ──────────────────────────────────────────
        print("Generating overlaid ROC / PR curves...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for name, color in zip(names, colors):
            yp   = np.array(self.results[name]['test']['probabilities'])
            yt   = self.data['test']['y']
            opt  = self.optimal_thresholds[name]

            auc_m  = self.results[name]['test']['ci']['auc']['mean']
            auc_lo = self.results[name]['test']['ci']['auc']['lo']
            auc_hi = self.results[name]['test']['ci']['auc']['hi']

            # ROC
            fpr, tpr, _ = roc_curve(yt, yp)
            axes[0].plot(
                fpr, tpr, color=color, linewidth=2,
                label=f"{name}\nAUC={auc_m:.3f} [{auc_lo:.3f}–{auc_hi:.3f}]"
            )
            # optimal threshold dot
            yp_bin = (yp >= opt).astype(int)
            tn, fp, fn, tp_ = confusion_matrix(yt, yp_bin, labels=[0,1]).ravel()
            fpr_pt = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_pt = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0
            axes[0].scatter(fpr_pt, tpr_pt, color=color, s=90,
                            zorder=5, marker='D', edgecolors='black', linewidths=0.6)

            # PR
            auprc_m  = self.results[name]['test']['ci']['auprc']['mean']
            auprc_lo = self.results[name]['test']['ci']['auprc']['lo']
            auprc_hi = self.results[name]['test']['ci']['auprc']['hi']
            prec, rec, _ = precision_recall_curve(yt, yp)
            axes[1].plot(
                rec, prec, color=color, linewidth=2,
                label=f"{name}\nAUPRC={auprc_m:.3f} [{auprc_lo:.3f}–{auprc_hi:.3f}]"
            )

        # ROC decorations
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0].set_xlabel('False Positive Rate', fontsize=10)
        axes[0].set_ylabel('True Positive Rate', fontsize=10)
        axes[0].set_title('ROC Curves — Test Set\n(◆ = optimal threshold)',
                          fontsize=10, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=7.5,
                       framealpha=0.9, ncol=1)
        axes[0].grid(True, alpha=0.3)

        # PR decorations
        baseline = float((self.data['test']['y'] == 1).mean())
        axes[1].axhline(baseline, color='k', linestyle='--', linewidth=1,
                        label=f'No-skill ({baseline:.3f})')
        axes[1].set_xlabel('Recall', fontsize=10)
        axes[1].set_ylabel('Precision', fontsize=10)
        axes[1].set_title('Precision-Recall Curves — Test Set',
                          fontsize=10, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=7.5,
                       framealpha=0.9, ncol=1)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.config.OUTPUT_DIR,
                            f'final_comparison_roc_pr_{model_suffix}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ ROC/PR plot  : {path}")

        # ── 4. SAVE COMPARISON JSON ───────────────────────────────────────
        print("Saving comparison JSON...")
        best_name, _ = self.get_best_model()
        comparison = {
            'dataset':          model_suffix,
            'best_model':       best_name,
            'threshold_metric': self.config.THRESHOLD_METRIC,
            'bootstrap_n':      self.config.BOOTSTRAP_N,
            'bootstrap_ci':     self.config.BOOTSTRAP_CI,
            'models': {}
        }

        for n in names:
            comparison['models'][n] = {
                'optimal_threshold': self.results[n]['optimal_threshold'],
                'best_params': (self.cv_results[n]['best_params']
                                if n in self.cv_results else {}),
                'cv_mean_auc': (self.cv_results[n]['mean']['auc']
                                if n in self.cv_results else None),
                'cv_std_auc':  (self.cv_results[n]['std']['auc']
                                if n in self.cv_results else None),
                'test_metrics': {
                    mk: {
                        'value': self.results[n]['test']['point'][mk],
                        'ci_lo': self.results[n]['test']['ci'][mk]['lo'],
                        'ci_hi': self.results[n]['test']['ci'][mk]['hi'],
                    }
                    for mk in mk_all
                },
                'confusion_matrix': {
                    k: self.results[n]['test']['point'][k]
                    for k in ['tp', 'fp', 'tn', 'fn']
                }
            }

        cmp_path = os.path.join(self.config.OUTPUT_DIR,
                                f'comparison_{model_suffix}.json')
        with open(cmp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"  ✓ Comparison JSON : {cmp_path}")

        return cmp_path

    # ── SAVE RESULTS ──────────────────────────────────────────────────────

    def save_results(self, model_suffix):
        print("\nSaving results to JSON...")
        best_name, _ = self.get_best_model()

        summary = {
            'model_suffix':      model_suffix,
            'k_folds':           self.config.K_FOLDS,
            'hyperparam_search': self.config.HYPERPARAM_SEARCH,
            'threshold_metric':  self.config.THRESHOLD_METRIC,
            'bootstrap_n':       self.config.BOOTSTRAP_N,
            'best_model':        best_name,
            'models':            {},
        }

        for name, res in self.results.items():
            entry = {
                'optimal_threshold': res['optimal_threshold'],
                'train_point':  res['train']['point'],
                'train_ci':     res['train']['ci'],
                'test_point':   res['test']['point'],
                'test_ci':      res['test']['ci'],
                'best_params':  (self.cv_results[name]['best_params']
                                 if name in self.cv_results else {}),
            }
            if name in self.cv_results:
                entry['cv'] = {
                    'mean': self.cv_results[name]['mean'],
                    'std':  self.cv_results[name]['std'],
                }
            for key in ['feature_importances', 'coefficients']:
                if key in res:
                    top = sorted(res[key].items(),
                                 key=lambda x: abs(x[1]), reverse=True)[:20]
                    entry['top_features'] = dict(top)
            summary['models'][name] = entry

        path = os.path.join(self.config.OUTPUT_DIR,
                            f'results_{model_suffix}.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Results saved to: {path}")
        return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80 + "\nLOADING DATA\n" + "="*80)

    choice = input(
        "\nWhich dataset?\n"
        "1. Prospective\n2. Retrospective\n3. Both (combined)\n"
        "Enter choice (1/2/3): "
    ).strip()
    dmap = {'1':('prospective','prospective'),
            '2':('retrospective','retrospective'),
            '3':('combined','combined')}
    dtype, suffix = dmap.get(choice, ('prospective','prospective'))
    print(f"\n✓ Dataset: {dtype}")

    data    = load_data(dtype)
    trainer = BaselineTrainer(data, config)

    # 1 ── nested CV
    trainer.cross_validate_all_models()

    # 2 ── final fit + threshold + metrics with CI
    results = trainer.train_all_models()

    best_name, best_model = trainer.get_best_model()

    # ── save best model with dedicated name ───────────────────────────
    best_safe = best_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    best_path = os.path.join(config.OUTPUT_DIR, f'best_model_{best_safe}.joblib')
    joblib.dump(best_model, best_path)
    print(f"\n✓ Best model saved : {best_path}  ({best_name})")

    # 3 ── plots
    trainer.plot_cv_results(suffix)
    trainer.plot_cv_metrics_heatmap(suffix)
    trainer.plot_threshold_search(suffix)
    trainer.plot_metrics_with_ci(suffix)
    trainer.plot_train_vs_test(suffix)
    trainer.plot_roc_pr_curves(suffix)
    trainer.plot_calibration_curves(suffix, n_bins=10, strategy='uniform')
    trainer.plot_feature_importance(suffix)
    trainer.plot_probability_distributions_by_class(suffix, showfliers=True)

    # 4 ── save full results
    results_path = trainer.save_results(suffix)

    # 5 ── final model comparison (table + forest plot + ROC/PR + JSON)
    cmp_path = trainer.plot_final_comparison(suffix)

    # 6 ── brief summary to stdout
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    best_name, _ = trainer.get_best_model()
    print(f"\nBest model (by best-fold test AUC): {best_name}")
    if trainer.cv_results:
        cv_m  = trainer.cv_results[best_name]['mean']['auc']
        cv_s  = trainer.cv_results[best_name]['std']['auc']
        bf    = trainer.cv_results[best_name]['best_fold']
        bfauc = trainer.cv_results[best_name]['best_fold_test_auc']
        print(f"  CV val AUC  : {cv_m:.4f} ± {cv_s:.4f}  (k={config.K_FOLDS})")
        print(f"  Best fold   : {bf}  (test AUC={bfauc:.4f}  [leakage intentional])")
        bp = trainer.cv_results[best_name]['best_params']
        if bp:
            print(f"  Params      : {bp}")

    pt  = results[best_name]['test']['point']
    ci  = results[best_name]['test']['ci']
    thr = results[best_name]['optimal_threshold']
    print(f"\nTest set  (threshold={thr:.3f})")
    for mk in ['auc','auprc','f1','balanced_accuracy','mcc']:
        print(f"  {mk:<20}: {pt[mk]:.4f}  "
              f"[{ci[mk]['lo']:.4f} – {ci[mk]['hi']:.4f}]")
    print(f"  TP={pt['tp']}  FP={pt['fp']}  TN={pt['tn']}  FN={pt['fn']}")

    print(f"\n✓ Full results    : {results_path}")
    print(f"✓ Comparison JSON : {cmp_path}")
    print(f"✓ Best model      : {best_path}")
    print(f"✓ All models      : {config.OUTPUT_DIR}/model_*.joblib")
    print(f"✓ Plots           : {config.OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
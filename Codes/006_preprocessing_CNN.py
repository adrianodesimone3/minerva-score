"""
data_preprocessing_only.py
==========================

Questo script gestisce SOLO:
- scelta MODE (preserving/categorical)
- scelta DATASET (prospective/retrospective/combined)
- fit del binner (solo in categorical mode) sul training
- caricamento dei CSV già split (train/val/test) e stampa statistiche base
- (se combined) salvataggio dei file combined (RAW + MODEL-READY) in OUTPUT_DIR

NON fa training, NON fa DeepInsight, NON fa modelli.

Assunzione: esistono già i CSV split in:
- processed_data/prospective/{train,val,test}.csv
- processed_data/retrospective/{train,val,test}.csv
"""

import os
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    CATEGORICAL_VARIABLES = [
        'sex', 'previous_episodes', 'admitting_specialty', 'diabetes',
        'chronic_pulmonary_disease', 'hypertension', 'atrial_fibrillation',
        'ischemic_heart_disease', 'chronic_kidney_disease', 'hematopoietic_disease',
        'immunosuppressive_medications', 'choledocholithiasis', 'cholangitis', 'ercp'
    ]

    CONTINUOUS_VARIABLES = [
        'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
        'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
        'serum_amylase', 'serum_lipase', 'ldh'
    ]

    MODE = "preserving"  # set by user
    N_BINS = 10

    OUTPUT_DIR = "preprocessing_outputs"

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


config = Config()


# =============================================================================
# BINNING (categorical mode)
# =============================================================================

class ContinuousBinner:
    """Bins continuous variables into discrete categories (quantile binning)."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges = {}

    def fit(self, data: np.ndarray, feature_names: list[str]):
        for i, name in enumerate(feature_names):
            x = data[:, i]
            q = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(x, q)
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([edges[0], edges[0] + 1e-6], dtype=float)
            self.bin_edges[name] = edges
        return self

    def transform(self, data: np.ndarray, feature_names: list[str]) -> np.ndarray:
        out = np.zeros_like(data, dtype=np.int64)
        for i, name in enumerate(feature_names):
            edges = self.bin_edges[name]
            bins = np.digitize(data[:, i], edges, right=False)
            bins = np.clip(bins - 1, 0, len(edges) - 2)
            out[:, i] = bins
        return out


# =============================================================================
# HELPERS
# =============================================================================

def _id_cols(dataset_type: str) -> list[str]:
    if dataset_type == "prospective":
        return ["patient_id"]
    return ["country", "admission_year"]


def load_split_csvs(dataset_type: str):
    base = f"processed_data/{dataset_type}"
    train_path = os.path.join(base, "train.csv")
    val_path = os.path.join(base, "val.csv")
    test_path = os.path.join(base, "test.csv")
    return train_path, val_path, test_path


def basic_report(df: pd.DataFrame, name: str):
    y = df["target"].astype(int).values
    n = len(y)
    pos = int(y.sum())
    print(f"{name}: n={n}, positives={pos} ({pos/max(n,1):.3f})")


def make_model_ready_df(df: pd.DataFrame, dataset_type: str, mode: str, binner: ContinuousBinner | None):
    """
    Ritorna un DataFrame 'model-ready' con:
    - colonne categorical vars (int/float come in input)
    - colonne continuous vars (float in preserving, bins int in categorical)
    - target
    (senza colonne ID)
    """
    id_cols = _id_cols(dataset_type)
    df = df.copy()
    y = df["target"].astype(int).values

    feat_cols = [c for c in df.columns if c not in id_cols + ["target"]]
    X = df[feat_cols]

    cat = X[config.CATEGORICAL_VARIABLES].values
    cont = X[config.CONTINUOUS_VARIABLES].values.astype(np.float32)

    if mode == "categorical":
        if binner is None:
            raise RuntimeError("categorical mode requires binner")
        cont = binner.transform(cont, config.CONTINUOUS_VARIABLES).astype(np.int64)

    out = pd.DataFrame(cat, columns=config.CATEGORICAL_VARIABLES)
    out_cont = pd.DataFrame(cont, columns=config.CONTINUOUS_VARIABLES)
    out = pd.concat([out, out_cont], axis=1)
    out["target"] = y
    return out


# =============================================================================
# MAIN (preprocessing only)
# =============================================================================

def main():
    print("=" * 80)
    print("PREPROCESSING / SPLITTING PIPELINE (ONLY)")
    print("=" * 80)

    # 1) mode
    mode_choice = input(
        "\nWhich mode?\n"
        "1. Preserving\n"
        "2. Categorical (bin continuous)\n"
        "Enter choice (1/2): "
    ).strip()

    if mode_choice == "2":
        config.MODE = "categorical"
        print(f"\nMode: CATEGORICAL (bins={config.N_BINS})")
    else:
        config.MODE = "preserving"
        print("\nMode: PRESERVING")

    # 2) dataset
    dataset_choice = input(
        "\nWhich dataset?\n"
        "1. Prospective\n"
        "2. Retrospective\n"
        "3. Both (combined)\n"
        "Enter choice (1/2/3): "
    ).strip()

    # 3) load train df(s) for binner fit (ONLY if categorical)
    binner = None
    if config.MODE == "categorical":
        print("\nFitting binner on training data...")
        if dataset_choice == "1":
            tr = pd.read_csv("processed_data/prospective/train.csv")
        elif dataset_choice == "2":
            tr = pd.read_csv("processed_data/retrospective/train.csv")
        else:
            tr1 = pd.read_csv("processed_data/prospective/train.csv")
            tr2 = pd.read_csv("processed_data/retrospective/train.csv")
            tr = pd.concat([tr1, tr2], ignore_index=True)

        cont = tr[config.CONTINUOUS_VARIABLES].values.astype(np.float32)
        binner = ContinuousBinner(n_bins=config.N_BINS).fit(cont, config.CONTINUOUS_VARIABLES)
        print("Binner fitted.\n")

    # 4) load CSV split e report
    if dataset_choice == "1":
        dataset_type = "prospective"
        train_path, val_path, test_path = load_split_csvs(dataset_type)

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        print("\nLoaded Prospective splits:")
        basic_report(df_train, "train")
        basic_report(df_val, "val")
        basic_report(df_test, "test")

        # salva model-ready (utile per DeepInsight su file separato)
        out_dir = os.path.join(config.OUTPUT_DIR, f"{dataset_type}_{config.MODE}")
        os.makedirs(out_dir, exist_ok=True)

        make_model_ready_df(df_train, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "train_model_ready.csv"), index=False)
        make_model_ready_df(df_val, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "val_model_ready.csv"), index=False)
        make_model_ready_df(df_test, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "test_model_ready.csv"), index=False)

        print(f"\n[Saved] Model-ready CSVs -> {out_dir}")

    elif dataset_choice == "2":
        dataset_type = "retrospective"
        train_path, val_path, test_path = load_split_csvs(dataset_type)

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        print("\nLoaded Retrospective splits:")
        basic_report(df_train, "train")
        basic_report(df_val, "val")
        basic_report(df_test, "test")

        out_dir = os.path.join(config.OUTPUT_DIR, f"{dataset_type}_{config.MODE}")
        os.makedirs(out_dir, exist_ok=True)

        make_model_ready_df(df_train, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "train_model_ready.csv"), index=False)
        make_model_ready_df(df_val, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "val_model_ready.csv"), index=False)
        make_model_ready_df(df_test, dataset_type, config.MODE, binner).to_csv(os.path.join(out_dir, "test_model_ready.csv"), index=False)

        print(f"\n[Saved] Model-ready CSVs -> {out_dir}")

    else:
        # combined
        print("\nLoaded Combined splits (prospective + retrospective):")

        # Prospective
        p_train = pd.read_csv("processed_data/prospective/train.csv")
        p_val = pd.read_csv("processed_data/prospective/val.csv")
        p_test = pd.read_csv("processed_data/prospective/test.csv")

        # Retrospective
        r_train = pd.read_csv("processed_data/retrospective/train.csv")
        r_val = pd.read_csv("processed_data/retrospective/val.csv")
        r_test = pd.read_csv("processed_data/retrospective/test.csv")

        basic_report(p_train, "prospective train")
        basic_report(r_train, "retrospective train")
        basic_report(p_val, "prospective val")
        basic_report(r_val, "retrospective val")
        basic_report(p_test, "prospective test")
        basic_report(r_test, "retrospective test")

        # Salva combined test RAW (come nel tuo script originario)
        p_test_raw = p_test.copy()
        r_test_raw = r_test.copy()
        p_test_raw["__source__"] = "prospective"
        r_test_raw["__source__"] = "retrospective"

        all_cols = sorted(set(p_test_raw.columns).union(set(r_test_raw.columns)))
        combined_test_raw = pd.concat(
            [p_test_raw.reindex(columns=all_cols), r_test_raw.reindex(columns=all_cols)],
            ignore_index=True
        )

        out_dir = os.path.join(config.OUTPUT_DIR, f"combined_{config.MODE}")
        os.makedirs(out_dir, exist_ok=True)

        combined_test_raw.to_csv(os.path.join(out_dir, "combined_test_raw.csv"), index=False)
        print(f"\n[Saved] combined_test_raw.csv -> {out_dir}")

        # Salva combined MODEL-READY per train/val/test (utile per DeepInsight su file separato)
        # Train
        p_tr_mr = make_model_ready_df(p_train, "prospective", config.MODE, binner)
        r_tr_mr = make_model_ready_df(r_train, "retrospective", config.MODE, binner)
        tr_mr = pd.concat([p_tr_mr.assign(__source__="prospective"), r_tr_mr.assign(__source__="retrospective")], ignore_index=True)

        # Val
        p_va_mr = make_model_ready_df(p_val, "prospective", config.MODE, binner)
        r_va_mr = make_model_ready_df(r_val, "retrospective", config.MODE, binner)
        va_mr = pd.concat([p_va_mr.assign(__source__="prospective"), r_va_mr.assign(__source__="retrospective")], ignore_index=True)

        # Test (model-ready)
        p_te_mr = make_model_ready_df(p_test, "prospective", config.MODE, binner)
        r_te_mr = make_model_ready_df(r_test, "retrospective", config.MODE, binner)
        te_mr = pd.concat([p_te_mr.assign(__source__="prospective"), r_te_mr.assign(__source__="retrospective")], ignore_index=True)

        tr_mr.to_csv(os.path.join(out_dir, "combined_train_model_ready.csv"), index=False)
        va_mr.to_csv(os.path.join(out_dir, "combined_val_model_ready.csv"), index=False)
        te_mr.to_csv(os.path.join(out_dir, "combined_test_model_ready.csv"), index=False)

        print(f"[Saved] combined_*_model_ready.csv -> {out_dir}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
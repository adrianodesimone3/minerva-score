#!/usr/bin/env python3
"""
deepinsight_umap_splat_16x16_export.py
=====================================

DeepInsight per UN CSV alla volta con:
- Embedding 2D tramite UMAP (al posto della PCA)
- Splatting bilineare su griglia 16x16 (immagini meno sparse)
- target ESCLUSO dal mapping
- nome file: ID_CLASS (ID = colonna id opzionale, altrimenti indice riga; CLASS = target)
- export:
    <stem>_deepinsight_16x16_umap/
        arrays/  -> .npy float32 (16x16) (consigliato per CNN)
        images/  -> .png preview (contrast robusto p1-p99)
        labels.csv
        mapping.json
        occupancy_heatmap.png

Uso:
    python deepinsight_umap_splat_16x16_export.py file.csv

Opzioni:
    --image-size 16
    --id-col patient_id
    --outdir path/output_base
    --suffix _deepinsight_16x16_umap
    --png-contrast 1 99
    --umap-n-neighbors 5
    --umap-min-dist 0.3
    --umap-metric correlation

Requisito:
    pip install umap-learn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# UMAP dependency
try:
    import umap  # umap-learn
except ImportError as e:
    raise SystemExit(
        "Missing dependency: umap-learn.\n"
        "Install with: pip install umap-learn\n"
        f"Original error: {e}"
    )

DEFAULT_DROP = ["patient_id", "country", "admission_year", "__source__"]
EPS = 1e-8


def coerce_numeric_and_impute(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric (coerce), median impute, then fill remaining NaN with 0."""
    out = df_feat.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    med = out.median(numeric_only=True)
    out = out.fillna(med).fillna(0.0)
    return out


def robust_minmax_01(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Robust normalization for PNG preview using percentiles."""
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    denom = max(hi - lo, EPS)
    out = (img - lo) / denom
    return np.clip(out, 0.0, 1.0)


def save_heatmap_png(path: Path, heat: np.ndarray, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(heat, interpolation="nearest")
    plt.colorbar(label="deposit weight")
    plt.title(title)
    plt.xticks(range(heat.shape[1]))
    plt.yticks(range(heat.shape[0]))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


class DeepInsightUMAPSplat:
    """
    DeepInsight mapping via UMAP 2D sulle FEATURES + splatting bilineare.

    Fit:
      - lavora su Fm = X^T (features x samples)
      - applica UMAP per ottenere coords 2D per ciascuna feature
      - scala coords continue in [0, S-1]

    Transform:
      - per ogni feature, splat bilineare su 4 pixel
      - output: (N, S, S) float32
    """

    def __init__(
        self,
        image_size=16,
        n_neighbors=5,
        min_dist=0.3,
        metric="correlation",
        random_state=42,
        eps=1e-8,
    ):
        self.S = int(image_size)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = str(metric)
        self.random_state = int(random_state)
        self.eps = float(eps)

        self.feature_names_ = None
        self.coords_cont_ = None  # (F,2) => (y,x) float in [0,S-1]
        self.n_features_ = None
        self._umap_model = None

    def fit(self, X: np.ndarray, feature_names: list[str]):
        X = np.asarray(X, dtype=np.float32)
        N, F = X.shape
        self.n_features_ = F
        self.feature_names_ = list(feature_names)

        # features x samples
        Fm = X.T  # (F,N)

        # UMAP expects samples x features => we treat each feature as a "sample" described by its values over patients
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        coords = reducer.fit_transform(Fm)  # (F,2)
        self._umap_model = reducer

        # scale to [0, S-1]
        cmin = coords.min(axis=0, keepdims=True)
        cmax = coords.max(axis=0, keepdims=True)
        denom = np.maximum(cmax - cmin, self.eps)
        coords01 = (coords - cmin) / denom

        x = coords01[:, 0] * (self.S - 1)
        y = coords01[:, 1] * (self.S - 1)
        self.coords_cont_ = np.stack([y, x], axis=1).astype(np.float32)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.coords_cont_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        N, F = X.shape
        if F != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {F}")

        S = self.S
        imgs = np.zeros((N, S, S), dtype=np.float32)

        for f in range(F):
            y, x = float(self.coords_cont_[f, 0]), float(self.coords_cont_[f, 1])

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, S - 1)
            y1 = min(y0 + 1, S - 1)

            dx = x - x0
            dy = y - y0

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            vals = X[:, f]

            imgs[:, y0, x0] += w00 * vals
            imgs[:, y0, x1] += w10 * vals
            imgs[:, y1, x0] += w01 * vals
            imgs[:, y1, x1] += w11 * vals

        return imgs

    def occupancy_heatmap(self) -> np.ndarray:
        """Heatmap of deposit weights per pixel (independent of sample values)."""
        if self.coords_cont_ is None:
            raise RuntimeError("Not fitted.")
        S = self.S
        heat = np.zeros((S, S), dtype=np.float32)
        for f in range(self.n_features_):
            y, x = float(self.coords_cont_[f, 0]), float(self.coords_cont_[f, 1])

            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, S - 1)
            y1 = min(y0 + 1, S - 1)

            dx = x - x0
            dy = y - y0

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            heat[y0, x0] += w00
            heat[y0, x1] += w10
            heat[y1, x0] += w01
            heat[y1, x1] += w11

        return heat

    def mapping_payload(self) -> dict:
        if self.coords_cont_ is None:
            raise RuntimeError("Not fitted.")
        mapping = []
        for name, (y, x) in zip(self.feature_names_, self.coords_cont_):
            mapping.append({"feature": name, "y": float(y), "x": float(x)})
        return {
            "deepinsight": {
                "image_size": self.S,
                "n_features": int(self.n_features_),
                "method": "UMAP2D(feature-embedding) + bilinear splat",
                "umap": {
                    "n_neighbors": self.n_neighbors,
                    "min_dist": self.min_dist,
                    "metric": self.metric,
                    "random_state": self.random_state,
                },
            },
            "mapping": mapping,
        }


def export_one_csv(
    csv_path: Path,
    out_base: Path | None,
    suffix: str,
    image_size: int,
    id_col: str | None,
    png_p1: float,
    png_p99: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
):
    #df = pd.read_csv(csv_path)
    ext = csv_path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(csv_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(csv_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Need target
    if "target" not in df.columns:
        raise ValueError("Missing required column 'target' in CSV.")
    y = df["target"].astype(int).values

    # ID for filenames
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).values
    else:
        ids = np.array([str(i) for i in range(len(df))], dtype=object)

    # Feature columns: exclude target ALWAYS, and drop known ID/source cols if present
    drop = set(DEFAULT_DROP)
    feat_cols = [c for c in df.columns if c not in drop and c.strip().lower() != "target"]

    # extra-safety: remove any weird casing/spaces variants of target
    feat_cols = [c for c in feat_cols if c.strip().lower() != "target"]

    if len(feat_cols) == 0:
        raise ValueError("No feature columns left after excluding target and drop cols.")

    df_feat = coerce_numeric_and_impute(df[feat_cols])
    X = df_feat.values.astype(np.float32)
    feature_names = list(df_feat.columns)

    di = DeepInsightUMAPSplat(
        image_size=image_size,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=42,
        eps=EPS,
    ).fit(X, feature_names)

    imgs = di.transform(X)  # (N,S,S)
    heat = di.occupancy_heatmap()

    # Output folder
    out_base = out_base if out_base is not None else csv_path.parent
    out_root = out_base / f"{csv_path.stem}{suffix}"
    arrays_dir = out_root / "arrays"
    images_dir = out_root / "images"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(df)):
        cls = int(y[i])
        fname_base = f"{ids[i]}_{cls}"

        npy_path = arrays_dir / f"{fname_base}.npy"
        np.save(npy_path, imgs[i].astype(np.float32))

        preview01 = robust_minmax_01(imgs[i], p_lo=png_p1, p_hi=png_p99)
        png_arr = (preview01 * 255.0).round().clip(0, 255).astype(np.uint8)
        png_path = images_dir / f"{fname_base}.png"
        Image.fromarray(png_arr, mode="L").save(png_path)

        rows.append({
            "id": ids[i],
            "target": cls,
            "array": str(Path("arrays") / f"{fname_base}.npy"),
            "image": str(Path("images") / f"{fname_base}.png"),
        })

    pd.DataFrame(rows).to_csv(out_root / "labels.csv", index=False)

    (out_root / "mapping.json").write_text(
        json.dumps(di.mapping_payload(), indent=2),
        encoding="utf-8"
    )
    save_heatmap_png(out_root / "occupancy_heatmap.png", heat, "Occupancy heatmap (UMAP + splat weights)")

    print(f"\n[OK] CSV: {csv_path}")
    print(f"[OK] Output folder: {out_root}")
    print(f"[OK] Samples: {len(df)}")
    print(f"[OK] Features used (target excluded): {len(feature_names)}")
    print(f"[OK] Saved arrays (.npy): {arrays_dir}")
    print(f"[OK] Saved PNG previews: {images_dir}")
    print(f"[OK] Saved labels.csv, mapping.json, occupancy_heatmap.png")
    print(f"[INFO] UMAP params: n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric={umap_metric}")
    print(f"[INFO] PNG preview contrast: p{png_p1}-p{png_p99}")


def main():
    ap = argparse.ArgumentParser(description="DeepInsight export (UMAP + splat) 16x16 for one CSV.")
    ap.add_argument("csv", type=str, help="Path al CSV (es. test_both.csv)")
    ap.add_argument("--image-size", type=int, default=16)
    ap.add_argument("--id-col", type=str, default=None, help="Colonna da usare come ID nel filename (default: row index)")
    ap.add_argument("--outdir", type=str, default=None, help="Directory output base (default: accanto al CSV)")
    ap.add_argument("--suffix", type=str, default="_deepinsight_16x16_umap", help="Suffisso cartella output")
    ap.add_argument("--png-contrast", type=float, nargs=2, default=[1.0, 99.0],
                    metavar=("P1", "P99"), help="Percentili per contrasto PNG preview (default: 1 99)")

    ap.add_argument("--umap-n-neighbors", type=int, default=5)
    ap.add_argument("--umap-min-dist", type=float, default=0.3)
    ap.add_argument("--umap-metric", type=str, default="correlation")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_base = Path(args.outdir).expanduser().resolve() if args.outdir else None
    p1, p99 = float(args.png_contrast[0]), float(args.png_contrast[1])

    export_one_csv(
        csv_path=csv_path,
        out_base=out_base,
        suffix=args.suffix,
        image_size=int(args.image_size),
        id_col=args.id_col,
        png_p1=p1,
        png_p99=p99,
        umap_n_neighbors=int(args.umap_n_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        umap_metric=str(args.umap_metric),
    )


if __name__ == "__main__":
    main()
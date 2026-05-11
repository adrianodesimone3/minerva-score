# MINERVA Score

A patient-facing risk calculator for 30-day hospital readmission after
gallstone (biliary) pancreatitis.

The model — a 5-seed ensemble of small 2D residual CNNs trained on
DeepInsight-style 16×16 images of 27 admission features — runs entirely in
the visitor's browser via [onnxruntime-web]. **No patient data ever leaves
the browser**: this server only ships the static HTML, CSS, JS, and ONNX
files, then steps out of the way.

The page is bilingual: English at `/`, Italian at `/it`.

## Layout

```
.
├── api.py                 # thin FastAPI shim — serves the static frontend
├── requirements.txt       # fastapi + uvicorn
├── render.yaml            # blueprint for Render (web service, python runtime)
├── index.html             # English page
├── index-it.html          # Italian page
├── styles.css             # shared design system (clinical / editorial)
├── app.js                 # English app: form, splat, ONNX ensemble, calibration
├── app-it.js              # Italian app (same pipeline, translated strings)
└── assets/
    ├── seed_0.onnx … seed_4.onnx     # 5-model ensemble, ~215 KB each
    ├── mapping.json                  # UMAP feature → (y, x) coordinates
    ├── feature_stats.json            # population mean/std for client-side z-scoring
    ├── calibration.json              # isotonic curve + threshold + headline metrics
    └── samples.json                  # 10 demo patients (5 positive, 5 negative)
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload
# -> http://localhost:8000
```

Or, since there is no server-side logic at all, you can also serve the
folder with any static server:

```bash
python3 -m http.server 8000
```

## Deploy

Render auto-deploys this branch as a Web Service running `uvicorn api:app`.
The build/start commands and the public URL are managed in the Render
dashboard. The `render.yaml` here is informational.

## Disclaimer

Research instrument — not a medical device. Do not use for clinical
decisions.

[onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/

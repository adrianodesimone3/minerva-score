"""
MINERVA Score · static-frontend server
======================================

This deployment serves a fully client-side application: the prediction model
(a 5-seed 2D-CNN ensemble) runs inside the visitor's browser via
onnxruntime-web, and never sends data to this server. There is no /predict
endpoint here — all inference happens in-browser.

The previous version of this repo ran a FastAPI service that did the
inference server-side. The Render service is configured to run
`uvicorn api:app`, so we keep this entrypoint and use it as a thin static
file server. That way the existing Render Web Service config (URL, build
command, start command, custom domain, env vars) keeps working with no
dashboard changes.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent

app = FastAPI(title="MINERVA Score (static frontend)")

# Open CORS so the page can be embedded if needed. No data is ever sent here
# anyway — inference is client-side.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Always-on routes ---------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
def healthz():
    """Render's default health check."""
    return Response(status_code=204)


@app.get("/it", include_in_schema=False)
def italian_alias():
    """Friendlier alias for /index-it.html."""
    return FileResponse(ROOT / "index-it.html")


# Static files -------------------------------------------------------------
#
# Mounting at "/" with html=True means GET / returns index.html and any
# other path is resolved relative to the repo root. The 5 ONNX models, all
# JSON config, the stylesheet, and both JS bundles are served the same way.
# This mount must come last so the explicit routes above win.

app.mount("/", StaticFiles(directory=ROOT, html=True), name="root")

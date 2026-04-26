"""
Minerva Score API
Predicts 30-day hospital readmission risk for acute biliary pancreatitis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

app = FastAPI(title="Minerva Score API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CONFIG (must match training config exactly)
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

CATEGORICAL_CARDINALITIES = {
    'sex': 3, 'previous_episodes': 2, 'admitting_specialty': 5,
    'diabetes': 2, 'chronic_pulmonary_disease': 2, 'hypertension': 2,
    'atrial_fibrillation': 2, 'ischemic_heart_disease': 2,
    'chronic_kidney_disease': 2, 'hematopoietic_disease': 2,
    'immunosuppressive_medications': 2, 'choledocholithiasis': 4,
    'cholangitis': 2, 'ercp': 6,
}

EMBEDDING_DIM  = 32
CONTINUOUS_DIM = 32
HIDDEN_DIMS    = [4, 8]
DROPOUT        = 0.1
THRESHOLD      = 0.415

# =============================================================================
# MODEL ARCHITECTURE (identical to training code)
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


class MinervaModel(nn.Module):
    def __init__(self):
        super().__init__()
        cat_cards = [CATEGORICAL_CARDINALITIES[v] for v in CATEGORICAL_VARIABLES]
        emb_dim   = EMBEDDING_DIM
        cont_dim  = CONTINUOUS_DIM

        self.cat_emb = CategoricalEmbedding(cat_cards, emb_dim)
        cat_in       = len(CATEGORICAL_VARIABLES) * emb_dim

        self.cont_enc = ContinuousProjection(len(CONTINUOUS_VARIABLES), cont_dim)
        cont_in = len(CONTINUOUS_VARIABLES) * cont_dim

        in_dim      = cat_in + cont_in
        self.blocks = nn.ModuleList()
        prev        = in_dim
        for h in HIDDEN_DIMS:
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
        x = torch.cat([self.cat_emb(categorical), self.cont_enc(continuous)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# =============================================================================
# LOAD MODEL + SCALER AT STARTUP
# =============================================================================

model  = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler

    model_path  = os.getenv("MODEL_PATH",  "best_fold_model")
    scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")

    # Load model weights
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    m = MinervaModel()
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    print("✓ Model loaded")

    # Load scaler if available
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")
    else:
        print("⚠ Scaler not found — continuous variables will NOT be normalized")


# =============================================================================
# INPUT SCHEMA
# =============================================================================

class PatientData(BaseModel):
    # Categorical (integers)
    sex: int                          # 0=unknown, 1=male, 2=female
    diabetes: int                     # 0=no, 1=yes
    chronic_pulmonary_disease: int    # 0=no, 1=yes
    previous_episodes: int            # 0=no, 1=yes
    hypertension: int                 # 0=no, 1=yes
    atrial_fibrillation: int          # 0=no, 1=yes
    ischemic_heart_disease: int       # 0=no, 1=yes
    chronic_kidney_disease: int       # 0=no, 1=yes
    hematopoietic_disease: int        # 0=no, 1=yes
    immunosuppressive_medications: int # 0=no, 1=yes
    choledocholithiasis: int          # 0-3
    cholangitis: int                  # 0=no, 1=yes
    ercp: int                         # 0-5

    # Continuous (floats)
    age: float
    bmi: float
    wbc: float
    neutrophils: float
    platelets: float
    inr: float
    crp: float
    ast: float
    alt: float
    total_bilirubin: float
    conjugated_bilirubin: float
    ggt: float
    serum_lipase: float
    ldh: float


# =============================================================================
# PREDICT ENDPOINT
# =============================================================================

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build categorical tensor
    cat_values = [getattr(data, v) for v in CATEGORICAL_VARIABLES]
    cat_tensor  = torch.tensor([cat_values], dtype=torch.long)

    # Build continuous tensor
    cont_values = np.array([[getattr(data, v) for v in CONTINUOUS_VARIABLES]], dtype=np.float32)
    if scaler is not None:
        cont_values = scaler.transform(cont_values)
    cont_tensor = torch.tensor(cont_values, dtype=torch.float32)

    # Inference
    with torch.no_grad():
        logits = model(cat_tensor, cont_tensor)
        prob   = float(F.softmax(logits, dim=1)[0, 1].item())

    risk_pct   = round(prob * 100, 1)
    high_risk  = prob >= THRESHOLD

    return {
        "probability":  risk_pct,
        "high_risk":    high_risk,
        "threshold":    round(THRESHOLD * 100, 1),
        "message":      "Alto rischio di riammissione" if high_risk else "Basso rischio di riammissione"
    }


@app.get("/")
def root():
    return {"status": "Minerva Score API online", "version": "1.0"}

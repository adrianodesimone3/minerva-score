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

SCALER_VARIABLES = [
    'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
    'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
    'serum_amylase', 'serum_lipase', 'ldh',
]

SCALER_IDX = [SCALER_VARIABLES.index(v) for v in CONTINUOUS_VARIABLES]

CATEGORICAL_CARDINALITIES = {
    'sex': 3, 'previous_episodes': 2,
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
        cat_cards     = [CATEGORICAL_CARDINALITIES[v] for v in CATEGORICAL_VARIABLES]
        self.cat_emb  = CategoricalEmbedding(cat_cards, EMBEDDING_DIM)
        self.cont_enc = ContinuousProjection(len(CONTINUOUS_VARIABLES), CONTINUOUS_DIM)
        in_dim        = len(CATEGORICAL_VARIABLES)*EMBEDDING_DIM + len(CONTINUOUS_VARIABLES)*CONTINUOUS_DIM
        self.blocks   = nn.ModuleList()
        prev          = in_dim
        for h in HIDDEN_DIMS:
            self.blocks.append(MLPBlock(prev, h, DROPOUT))
            prev = h
        self.head = nn.Sequential(
            nn.Linear(prev, max(prev//2, 2)),
            nn.BatchNorm1d(max(prev//2, 2)),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(max(prev//2, 2), 2),
        )
    def forward(self, categorical, continuous):
        x = torch.cat([self.cat_emb(categorical), self.cont_enc(continuous)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


model  = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    model_path  = os.getenv("MODEL_PATH",  "best_fold_model")
    scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")
    checkpoint  = torch.load(model_path, map_location="cpu", weights_only=False)
    m = MinervaModel()
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    model = m
    print("✓ Model loaded")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded ({scaler.n_features_in_} features)")
    else:
        print("⚠ Scaler not found")


class PatientData(BaseModel):
    sex: int
    diabetes: int
    chronic_pulmonary_disease: int
    previous_episodes: int
    hypertension: int
    atrial_fibrillation: int
    ischemic_heart_disease: int
    chronic_kidney_disease: int
    hematopoietic_disease: int
    immunosuppressive_medications: int
    choledocholithiasis: int
    cholangitis: int
    ercp: int
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
    serum_amylase: float = 0.0


@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cat_tensor = torch.tensor(
        [[getattr(data, v) for v in CATEGORICAL_VARIABLES]], dtype=torch.long)
    if scaler is not None:
        full_cont   = np.array([[getattr(data, v) for v in SCALER_VARIABLES]], dtype=np.float32)
        full_scaled = scaler.transform(full_cont)
        cont_values = full_scaled[:, SCALER_IDX]
    else:
        cont_values = np.array(
            [[getattr(data, v) for v in CONTINUOUS_VARIABLES]], dtype=np.float32)
    cont_tensor = torch.tensor(cont_values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(cat_tensor, cont_tensor)
        prob   = float(F.softmax(logits, dim=1)[0, 1].item())
    high_risk = prob >= THRESHOLD
    return {
        "probability": round(prob*100, 1),
        "high_risk":   high_risk,
        "threshold":   round(THRESHOLD*100, 1),
        "message":     "Alto rischio di riammissione" if high_risk else "Basso rischio di riammissione"
    }


@app.get("/")
def root():
    return {"status": "Minerva Score API online", "version": "1.0"}

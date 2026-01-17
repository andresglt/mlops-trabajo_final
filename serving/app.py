

import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Importa tu función de preprocesamiento
from serving.preprocess_inference import preprocess_new_data

# Ruta del modelo versionado con DVC
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

# Cargar el modelo desde archivo local
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Modelo cargado desde {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"❌ Modelo no encontrado en {MODEL_PATH}. Asegúrate de ejecutar 'dvc pull' antes de iniciar el servicio.")

app = FastAPI(title="Churn Model API")

class Record(BaseModel):
    features: Dict[str, Any]

class Batch(BaseModel):
    records: List[Record]

@app.get("/")
def root():
    return {"message": "Bienvenido a la Churn Model API. Usa /health o /predict."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(batch: Batch):
    if model is None:
        return {"error": "Modelo no cargado"}

    raw_records = [r.features for r in batch.records]

    try:
        # Preprocesar con tu módulo externo
        X = preprocess_new_data(raw_records)
    except Exception as e:
        return {"error": f"Error al preprocesar datos: {str(e)}"}

    try:
        preds = model.predict(X)
        preds_list = preds.tolist() if hasattr(preds, "tolist") else preds
    except Exception as e:
        return {"error": f"Error al predecir: {str(e)}"}

    # Emparejar cada predicción con su customerID original
    results = []
    for rec, pred in zip(raw_records, preds_list):
        results.append({
            "customerID": rec.get("customerID"),
            "prediction": pred
        })

    return {"results": results}
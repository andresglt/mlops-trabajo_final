import pandas as pd
import numpy as np
import pickle
import json

# Cargar el preprocesador entrenado
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Cargar metadata de columnas
meta = json.load(open("data/processed/meta.json"))
numeric_cols = meta["numeric_cols"]
categorical_cols = meta["categorical_cols"]

def preprocess_new_data(raw_records):
    """
    raw_records: lista de diccionarios con las variables en bruto
    Ejemplo:
    [{"customerID":"1234","gender":"Male","SeniorCitizen":0,...}]
    """
    df = pd.DataFrame(raw_records)

    # Excluir columnas irrelevantes si aparecen
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convertir TotalCharges a numérica si existe
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Asegurar orden de columnas
    expected_cols = numeric_cols + categorical_cols
    try:
        df = df[expected_cols]
    except KeyError as e:
        raise ValueError(f"Columnas faltantes o incorrectas en el payload: {str(e)}")

    # Transformar con el preprocesador entrenado
    X = preprocessor.transform(df)

    return X
    
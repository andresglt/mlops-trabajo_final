import pandas as pd
import numpy as np
import pickle
import json

# Cargar el preprocesador entrenado

model_path = os.path.join(os.getcwd(), "models", "preprocessor.pkl")
with open(model_path, "rb") as f:
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
    for col in ["y"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Asegurar orden de columnas
    expected_cols = numeric_cols + categorical_cols
    try:
        df = df[expected_cols]
    except KeyError as e:
        raise ValueError(f"Columnas faltantes o incorrectas en el payload: {str(e)}")

    # Transformar con el preprocesador entrenado
    X = preprocessor.transform(df)

    return X
    
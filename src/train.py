import os
import json
import yaml
import mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import joblib
import pathlib


# Azure ML SDK
from azureml.core import Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication

# 🔐 Cargar variables de entorno desde .env (solo en local)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Parámetros
params = yaml.safe_load(open("params.yaml"))
registered_model_name = params["model"]["registered_model_name"]

# Autenticación con Service Principal
sp_auth = ServicePrincipalAuthentication(
    tenant_id=os.environ["AZURE_TENANT_ID"],
    service_principal_id=os.environ["AZURE_CLIENT_ID"],
    service_principal_password=os.environ["AZURE_CLIENT_SECRET"]
)

# Conectar al Workspace
ws = Workspace.get(
    name=os.environ["AZURE_WORKSPACE_NAME"],
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group=os.environ["AZURE_RESOURCE_GROUP"],
    auth=sp_auth
)

# Configurar MLflow con Azure ML
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("churn-mlops")

# Carga data
X_train = pd.read_parquet("data/processed/X_train.parquet").values
y_train = pd.read_parquet("data/processed/y_train.parquet").values.ravel()
X_test = pd.read_parquet("data/processed/X_test.parquet").values
y_test = pd.read_parquet("data/processed/y_test.parquet").values.ravel()

# Definiciones de búsqueda
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

logreg_grid = {
    "C": params["logreg"]["C"],
    "penalty": params["logreg"]["penalty"],
    "solver": params["logreg"]["solver"],
}
logreg = LogisticRegression(max_iter=1000)

xgb_grid = {
    "n_estimators": params["xgb"]["n_estimators"],
    "max_depth": params["xgb"]["max_depth"],
    "learning_rate": params["xgb"]["learning_rate"],
    "subsample": params["xgb"]["subsample"],
    "colsample_bytree": params["xgb"]["colsample_bytree"],
}
xgb = XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    tree_method="hist"
)

# Entrena y evalúa LogReg
with mlflow.start_run(run_name="logreg"):
    grid_lr = GridSearchCV(logreg, logreg_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)

    mlflow.log_params({"model": "logreg", **grid_lr.best_params_})
    mlflow.log_metrics({"auc": auc_lr})

# Entrena y evalúa XGBoost
with mlflow.start_run(run_name="xgboost"):
    grid_xgb = GridSearchCV(xgb, xgb_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)

    mlflow.log_params({"model": "xgboost", **grid_xgb.best_params_})
    mlflow.log_metrics({"auc": auc_xgb})

# Selección por AUC
scores = {"logreg": float(auc_lr), "xgboost": float(auc_xgb)}
best_name = max(scores, key=scores.get)
best_auc = scores[best_name]
print(f"Mejor modelo: {best_name} (AUC={best_auc:.4f})")

# Guarda el mejor como artefacto local (para DVC)
pathlib.Path("models").mkdir(exist_ok=True)
best_model = best_lr if best_name == "logreg" else best_xgb
joblib.dump(best_model, "models/best_model.pkl")

# Guarda métricas
with open("metrics.json", "w") as f:
    json.dump({"auc": best_auc, **scores}, f)

# 📌 Registro oficial del modelo en Azure ML
Model.register(
    workspace=ws,
    model_path="models/best_model.pkl",   # ruta al archivo local
    model_name=registered_model_name
)

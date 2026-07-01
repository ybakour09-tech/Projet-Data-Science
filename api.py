from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import pandas as pd
import shap
import os

app = FastAPI(
    title="API Prédiction ROI & Mix Média - PRISMA",
    description=(
        "API REST pour estimer le ROI des campagnes publicitaires et segmenter "
        "les profils budgétaires, avec explicabilité SHAP.\n\n"
        "Architecture (corrigée) :\n"
        "1. Segmentation NON SUPERVISÉE (KMeans) des profils budgétaires "
        "TV/Radio/Social Media -> identifie un segment de performance, "
        "sans aucun classifieur supervisé.\n"
        "2. Régression du ROI (MLPRegressor / Deep Learning), entraînée et "
        "évaluée sur un vrai split train/test (voir /model-info)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
FEATURES = ["TV", "Radio", "Social Media"]

_models_loaded = False
load_error = None

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_segmentation.pkl"))
    cluster_label_map = joblib.load(os.path.join(MODELS_DIR, "cluster_label_map.pkl"))
    reg_roi = joblib.load(os.path.join(MODELS_DIR, "regressor_roi.pkl"))
    shap_bg = joblib.load(os.path.join(MODELS_DIR, "shap_background.pkl"))

    with open(os.path.join(MODELS_DIR, "evaluation_report.json"), "r", encoding="utf-8") as f:
        evaluation_report = json.load(f)

    explainer = shap.KernelExplainer(reg_roi.predict, shap_bg)
    _models_loaded = True
except Exception as e:
    load_error = str(e)
    print(f"Erreur lors du chargement des modèles : {e}")


class BudgetScenario(BaseModel):
    TV: float = Field(..., description="Budget alloué à la TV (ex: 50.0)")
    Radio: float = Field(..., description="Budget alloué à la Radio (ex: 20.0)")
    Social_Media: float = Field(..., alias="Social Media", description="Budget alloué aux Réseaux Sociaux (ex: 5.0)")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {"TV": 50.0, "Radio": 20.0, "Social Media": 5.0}
        }


def _scale_input(scenario: BudgetScenario) -> pd.DataFrame:
    raw_df = pd.DataFrame([[scenario.TV, scenario.Radio, scenario.Social_Media]], columns=FEATURES)
    scaled_array = scaler.transform(raw_df)
    return pd.DataFrame(scaled_array, columns=FEATURES)


def _assign_segment(X_scaled: pd.DataFrame):
    """Segmentation NON SUPERVISÉE : même fonction utilisée à l'entraînement,
    à l'évaluation sur le test set, et ici en production. Pas de classifieur
    supervisé, donc pas de décalage train/serving."""
    cluster_id = int(kmeans.predict(X_scaled)[0])
    label = cluster_label_map.get(cluster_id, f"Segment {cluster_id}")
    return cluster_id, label


def _require_models():
    if not _models_loaded:
        raise HTTPException(status_code=503, detail=f"Modèles non chargés : {load_error}")


@app.get("/health", summary="Vérifier que le service est actif")
def health():
    return {
        "status": "ok" if _models_loaded else "degraded",
        "models_loaded": _models_loaded,
        "error": load_error,
    }


@app.get("/model-info", summary="Informations et métriques honnêtes du modèle déployé")
def model_info():
    _require_models()
    return {
        "status": "success",
        "regression_model": "MLPRegressor (Deep Learning)",
        "segmentation_model": "KMeans (non supervisé, fit sur TV/Radio/Social Media uniquement)",
        "evaluation": evaluation_report,
        "note": (
            "Les métriques R²/RMSE/MAE ci-dessus sont calculées sur un test set "
            "jamais vu pendant l'entraînement (split 80/20, random_state=42)."
        ),
    }


@app.post("/predict/performance", summary="Segmenter le profil budgétaire (non supervisé)")
def predict_performance(scenario: BudgetScenario):
    """
    Assigne le scénario budgétaire à un segment ('Haute Performance' /
    'Basse Performance') via KMeans, sans aucun classifieur supervisé.
    Le segment est caractérisé a posteriori par le Sales moyen observé
    historiquement dans ce cluster.
    """
    _require_models()
    try:
        X_scaled = _scale_input(scenario)
        cluster_id, label = _assign_segment(X_scaled)
        distances = kmeans.transform(X_scaled)[0]
        return {
            "status": "success",
            "Segment_Cluster_Id": cluster_id,
            "Performance_Segment": label,
            "method": "KMeans (non supervisé)",
            "distance_to_centroids": [round(float(d), 4) for d in distances],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/roi", summary="Estimer le ROI (Valeur exacte)")
def predict_roi(scenario: BudgetScenario):
    """
    Estime le ROI (%) via le MLPRegressor (Deep Learning), entraîné et évalué
    sur un split train/test honnête (voir /model-info pour les métriques réelles).
    """
    _require_models()
    try:
        X_scaled = _scale_input(scenario)
        cluster_id, label = _assign_segment(X_scaled)

        X_reg = np.column_stack([X_scaled.values, [cluster_id]])
        roi_pred = reg_roi.predict(X_reg)[0]

        return {
            "status": "success",
            "Performance_Segment": label,
            "ROI_Prediction_Percentage": round(float(roi_pred), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/shap_impact", summary="Analyser l'Impact des Canaux (SHAP)")
def predict_shap(scenario: BudgetScenario):
    """
    Décompose la prédiction du ROI pour comprendre l'impact de chaque canal
    publicitaire et du segment budgétaire identifié.
    """
    _require_models()
    try:
        X_scaled = _scale_input(scenario)
        cluster_id, label = _assign_segment(X_scaled)

        X_reg = np.column_stack([X_scaled.values, [cluster_id]])
        shap_values = explainer.shap_values(X_reg)
        base_value = float(explainer.expected_value)

        feature_names = FEATURES + ["Segment_Cluster"]
        impact = {feat: round(float(val), 4) for feat, val in zip(feature_names, shap_values[0])}

        predicted_roi = base_value + sum(shap_values[0])

        return {
            "status": "success",
            "Performance_Segment": label,
            "Base_ROI_Average": round(base_value, 2),
            "Predicted_ROI": round(float(predicted_roi), 2),
            "SHAP_Impact_Breakdown": impact,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Démarrage du serveur FastAPI via Uvicorn sur le port 8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

# ⚠️ DÉPRÉCIÉ : remplacé par scripts/build_production_pipeline.py qui corrige
# 3 problèmes critiques : (1) absence de split train/test (le modèle déployé
# n'était jamais évalué), (2) segmentation par classifieur supervisé entraîné
# sur un label lui-même issu d'un clustering sur Sales (incohérence
# train/serving), (3) absence d'évaluation honnête du modèle réellement
# utilisé par l'API. NE PLUS EXÉCUTER CE SCRIPT : il écraserait les artefacts
# corrigés dans models/.

import pandas as pd
import numpy as np
import joblib
import os
import shap

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

def train_and_save_models():
    print("Chargement des données...")
    df = pd.read_csv("data/Classified_Data_HSS.csv")
    
    features_num = ['TV', 'Radio', 'Social Media']
    
    # 1. Scaler & Encoder
    print("Entraînement Scaler & Encoder...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_num])
    X_scaled_df = pd.DataFrame(X_scaled, columns=features_num)
    
    encoder = LabelEncoder()
    y_perf = encoder.fit_transform(df['Performance'])
    
    # 2. Modèle de Classification (Performance)
    print("Entraînement Modèle Classification (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled_df, y_perf)
    
    # 3. Modèle de Régression (ROI)
    print("Entraînement Modèle Régression (MLP)...")
    # Features = [TV_scaled, Radio_scaled, Social_scaled, Performance_encoded]
    X_reg = X_scaled_df.copy()
    X_reg['Performance_encoded'] = y_perf
    y_roi = df['ROI']
    
    reg = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                       solver='adam', max_iter=1000, random_state=42)
    reg.fit(X_reg, y_roi)
    
    # 4. Explainer SHAP (sur le modèle de régression)
    print("Création de l'Explainer SHAP (KernelExplainer sur MLP)...")
    # On échantillonne les données d'entraînement pour le background SHAP
    background = shap.sample(X_reg, 100)
    explainer = shap.KernelExplainer(reg.predict, background)
    
    # 5. Sauvegarde
    print("\nSauvegarde de tous les modèles dans le dossier 'models/'...")
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(clf, "models/classifier_perf.pkl")
    joblib.dump(reg, "models/regressor_roi.pkl")
    
    # Pour l'explainer, SHAP KernelExplainer peut être lourd à dumper. 
    # Alternative: on dumpe le background, et l'API reconstruira l'explainer (très rapide).
    joblib.dump(background, "models/shap_background.pkl")
    
    print("Terminé ! Modèles prêts pour l'API REST.")

if __name__ == "__main__":
    train_and_save_models()

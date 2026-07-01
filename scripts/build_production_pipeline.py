"""
build_production_pipeline.py
=============================

Remplace l'ancien `save_models.py` + `clustering.py` + `final_classification_pipeline.py`.

Corrections apportées par rapport à l'ancienne version :

1. SEGMENTATION DES CANAUX : 100% NON SUPERVISÉE (condition imposée).
   L'ancien système utilisait un KMeans sur la variable Sales SEULE pour fabriquer
   des labels "Performance", puis entraînait un RandomForestClassifier supervisé
   pour ré-apprendre ces labels à partir de TV/Radio/Social Media. Problème :
   au moment de l'entraînement du régresseur ROI, on utilisait le VRAI label
   (calculé sur Sales connu), alors qu'en production l'API utilise un label
   PRÉDIT (potentiellement faux) -> décalage train/serving non documenté.

   Ici, le KMeans est appliqué DIRECTEMENT sur les budgets (TV, Radio, Social
   Media) normalisés, jamais sur Sales. Le clustering devient une fonction
   déterministe du budget d'entrée : on peut appeler exactement la même
   fonction (`scaler.transform` puis `kmeans.predict`) à l'entraînement, à
   l'évaluation sur le test set, ET en production dans l'API. Plus aucun
   classifieur supervisé n'est nécessaire pour cette étape : on supprime donc
   l'incohérence train/serving.

   Les clusters sont ensuite caractérisés (a posteriori, donc sans fuite)
   par leur Sales moyen, pour leur donner un nom métier ("Haute Performance",
   "Basse Performance", ...).

2. PLUS DE FUITE DE DONNÉES : le split train/test est fait EN PREMIER. Le
   StandardScaler et le KMeans sont fit() UNIQUEMENT sur le train. Le test
   set ne sert qu'à `.transform()` / `.predict()`.

3. LE MODÈLE DÉPLOYÉ EST ENFIN ÉVALUÉ : le MLPRegressor (Deep Learning) qui
   sera utilisé par l'API est entraîné sur le train set et évalué sur le
   test set (R², RMSE, MAE). Ces métriques sont sauvegardées dans
   `models/evaluation_report.json` et doivent être citées dans le rapport
   à la place des chiffres "trop parfaits" obtenus sur 100% des données.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["TV", "Radio", "Social Media"]
MODELS_DIR = "models"


def compute_roi(df: pd.DataFrame) -> pd.DataFrame:
    """ROI = (Sales - Investissement total) / Investissement total * 100."""
    df = df.copy()
    df["Total_Investment"] = df["TV"] + df["Radio"] + df["Social Media"]
    df["ROI"] = ((df["Sales"] - df["Total_Investment"]) / df["Total_Investment"]) * 100
    return df


def pick_best_k(X_train_scaled: np.ndarray, k_range=range(2, 7)) -> tuple:
    """Sélectionne le nombre de clusters via le score de silhouette, sur le TRAIN uniquement."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_train_scaled)
        scores[k] = silhouette_score(X_train_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_clusters_by_sales(kmeans: KMeans, sales_train: np.ndarray, cluster_assignments: np.ndarray) -> dict:
    """Nomme chaque cluster ('Basse'/'Haute' Performance, etc.) selon le Sales moyen observé
    sur le TRAIN pour ce cluster. C'est une étape d'INTERPRÉTATION a posteriori (pas
    d'entraînement) : elle ne crée aucune fuite, car elle n'est utilisée que pour
    l'affichage métier, jamais comme variable d'entrée d'un modèle supervisé."""
    k = kmeans.n_clusters
    mean_sales_per_cluster = {
        c: float(np.mean(sales_train[cluster_assignments == c])) for c in range(k)
    }
    ordered = sorted(mean_sales_per_cluster, key=mean_sales_per_cluster.get)

    if k == 2:
        names = ["Basse Performance", "Haute Performance"]
    elif k == 3:
        names = ["Basse Performance", "Moyenne Performance", "Haute Performance"]
    elif k == 4:
        names = ["Basse", "Moyenne-Basse", "Moyenne-Haute", "Haute"]
    else:
        names = [f"Niveau {i + 1}" for i in range(k)]

    mapping = {cluster_id: names[rank] for rank, cluster_id in enumerate(ordered)}
    return mapping, mean_sales_per_cluster


def build_pipeline(file_path: str = "data/Clean_Data_HSS.csv"):
    print(f"Chargement des données nettoyées : {file_path}")
    df = compute_roi(pd.read_csv(file_path))

    # =========================================================================
    # 1. SPLIT TRAIN / TEST D'ABORD (avant tout fit de scaler/cluster/modèle)
    # =========================================================================
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train : {df_train.shape[0]} lignes | Test : {df_test.shape[0]} lignes")

    # =========================================================================
    # 2. SEGMENTATION NON SUPERVISÉE DES CANAUX (KMeans sur les budgets)
    # =========================================================================
    print("\n--- Segmentation non supervisée (KMeans sur TV / Radio / Social Media) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[FEATURES])
    X_test_scaled = scaler.transform(df_test[FEATURES])  # transform uniquement, pas de fit

    best_k, silhouette_scores = pick_best_k(X_train_scaled)
    print(f"Meilleur K (silhouette, calculé sur le TRAIN) : {best_k}")
    for k, s in silhouette_scores.items():
        print(f"  K={k} -> silhouette={s:.4f}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)  # même fonction qu'en production

    cluster_label_map, mean_sales_per_cluster = label_clusters_by_sales(
        kmeans, df_train["Sales"].values, train_clusters
    )
    print(f"Labels métier des clusters (caractérisés a posteriori) : {cluster_label_map}")
    print(f"Sales moyen par cluster (train) : {mean_sales_per_cluster}")

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["Cluster"] = train_clusters
    df_test["Cluster"] = test_clusters
    df_train["Performance_Segment"] = df_train["Cluster"].map(cluster_label_map)
    df_test["Performance_Segment"] = df_test["Cluster"].map(cluster_label_map)

    # =========================================================================
    # 3. RÉGRESSION ROI (Deep Learning - MLP) : train sur TRAIN, éval sur TEST
    # =========================================================================
    print("\n--- Régression ROI (MLPRegressor) ---")
    reg_features_train = np.column_stack([X_train_scaled, train_clusters])
    reg_features_test = np.column_stack([X_test_scaled, test_clusters])

    y_train = df_train["ROI"].values
    y_test = df_test["ROI"].values

    reg = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    reg.fit(reg_features_train, y_train)

    y_pred_train = reg.predict(reg_features_train)
    y_pred_test = reg.predict(reg_features_test)

    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "n_train": int(df_train.shape[0]),
        "n_test": int(df_test.shape[0]),
        "best_k": int(best_k),
        "silhouette_scores": {str(k): round(v, 4) for k, v in silhouette_scores.items()},
        "cluster_label_map": {str(k): v for k, v in cluster_label_map.items()},
        "mean_sales_per_cluster_train": {str(k): round(v, 2) for k, v in mean_sales_per_cluster.items()},
    }
    print("\nMétriques HONNÊTES sur le test set (jamais vu pendant l'entraînement) :")
    print(f"  R² train = {metrics['r2_train']:.4f} | R² test = {metrics['r2_test']:.4f}")
    print(f"  RMSE test = {metrics['rmse_test']:.4f} | MAE test = {metrics['mae_test']:.4f}")

    # =========================================================================
    # 4. SHAP BACKGROUND (échantillon du TRAIN — chargé par l'API au démarrage)
    #    Note : shap.sample() n'est pas utilisé ici pour éviter la dépendance
    #    à la compilation C (pas de wheel Python 3.14+). On sauvegarde un
    #    DataFrame pandas ordinaire ; l'API reconstruit le KernelExplainer à
    #    partir de ce background au démarrage (comportement identique).
    # =========================================================================
    print("\nPréparation du background SHAP (échantillon aléatoire du train)...")
    background_df = pd.DataFrame(reg_features_train, columns=FEATURES + ["Cluster"]).sample(
        n=min(100, len(reg_features_train)), random_state=42
    )

    # =========================================================================
    # 5. SAUVEGARDE DES ARTEFACTS DE PRODUCTION
    # =========================================================================
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_segmentation.pkl"))
    joblib.dump(cluster_label_map, os.path.join(MODELS_DIR, "cluster_label_map.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "regressor_roi.pkl"))
    joblib.dump(background_df, os.path.join(MODELS_DIR, "shap_background.pkl"))

    with open(os.path.join(MODELS_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nArtefacts sauvegardés dans '{MODELS_DIR}/' (incluant evaluation_report.json).")
    print("Anciens fichiers obsolètes à supprimer manuellement si présents : encoder.pkl, classifier_perf.pkl")

    return metrics


if __name__ == "__main__":
    build_pipeline()

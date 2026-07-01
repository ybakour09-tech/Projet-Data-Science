"""
roi_regression.py
==================
Script de COMPARAISON de 4 modèles pour la régression du ROI (livrable
rapport : "comparaison multi-algorithmes").

Corrections apportées par rapport à la version précédente :
  1. Plus de fuite de données : split train/test D'ABORD, puis fit du
     StandardScaler et du KMeans UNIQUEMENT sur le train (transform seulement
     sur le test).
  2. La variable "Performance" n'est plus un label supervisé entraîné sur
     l'ensemble du jeu de données (ancienne classification.py). C'est
     désormais un cluster NON SUPERVISÉ (KMeans sur TV/Radio/Social Media
     normalisés), strictement cohérent avec scripts/build_production_pipeline.py
     utilisé par l'API.
  3. Renommage correct des modèles : SGDRegressor est un modèle LINÉAIRE
     entraîné par descente de gradient -> c'est du Machine Learning classique,
     PAS du Deep Learning. Seul le MLPRegressor (réseau de neurones multicouche)
     est un modèle de Deep Learning ici.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["TV", "Radio", "Social Media"]


def compute_roi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Investment"] = df["TV"] + df["Radio"] + df["Social Media"]
    df["ROI"] = ((df["Sales"] - df["Total_Investment"]) / df["Total_Investment"]) * 100
    return df


def roi_regression_pipeline(file_path="data/Clean_Data_HSS.csv"):
    print(f"Chargement des données : {file_path}")
    df = compute_roi(pd.read_csv(file_path))
    print(f"Dimensions : {df.shape}")

    # =========================================================================
    # 1. SPLIT TRAIN / TEST EN PREMIER
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : SÉPARATION DES DONNÉES (avant tout fit)")
    print("=" * 60)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train : {df_train.shape[0]} lignes | Test : {df_test.shape[0]} lignes")

    # =========================================================================
    # 2. NORMALISATION (fit sur train uniquement) + SEGMENTATION NON SUPERVISÉE
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : NORMALISATION + SEGMENTATION (KMeans non supervisé)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[FEATURES])
    X_test_scaled = scaler.transform(df_test[FEATURES])  # transform seulement

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_train = kmeans.fit_predict(X_train_scaled)
    cluster_test = kmeans.predict(X_test_scaled)  # predict seulement, pas de fit

    X_train = np.column_stack([X_train_scaled, cluster_train])
    X_test = np.column_stack([X_test_scaled, cluster_test])
    feature_cols = FEATURES + ["Segment_Cluster"]

    y_train = df_train["ROI"].values
    y_test = df_test["ROI"].values

    # =========================================================================
    # 3. MODÉLISATION : 3 modèles Machine Learning + 1 modèle Deep Learning
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES (3 ML + 1 DL)")
    print("=" * 60)

    models = {
        # --- Machine Learning classique ---
        "Random Forest (ML)": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR Linéaire (ML)": SVR(kernel="linear", C=1.0),
        "SGD Linéaire (ML)": SGDRegressor(
            loss="squared_error", learning_rate="invscaling", max_iter=2000, random_state=42
        ),
        # --- Deep Learning ---
        "MLP Réseau de Neurones (DL)": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
        ),
    }

    results = []
    predictions = {"ROI_Réel": y_test}

    for name, model in models.items():
        print(f"\n  Entraînement : {name}...")
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[name] = y_pred_test

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        results.append(
            {
                "Modèle": name,
                "R² Train": round(r2_train, 4),
                "R² Test": round(r2_test, 4),
                "RMSE (Test)": round(rmse_test, 4),
                "MAE (Test)": round(mae_test, 4),
                "Écart Train/Test": round(abs(r2_train - r2_test), 4),
            }
        )
        print(f"    R² Train: {r2_train:.4f} | R² Test: {r2_test:.4f} | RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}")

    results_df = pd.DataFrame(results)

    # =========================================================================
    # 4. IMPORTANCE DES VARIABLES (Random Forest)
    # =========================================================================
    rf_model = models["Random Forest (ML)"]
    importance_df = pd.DataFrame(
        {"Variable": feature_cols, "Importance": rf_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # =========================================================================
    # 5. RAPPORT TEXTUEL
    # =========================================================================
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/rapport_regression_roi.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT COMPLET : ESTIMATION DU ROI PAR MIX MÉDIA (RÉGRESSION)\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. PREPROCESSING\n" + "-" * 50 + "\n")
        f.write(f"  Variables numériques normalisées (StandardScaler, fit sur TRAIN uniquement) : {FEATURES}\n")
        f.write("  Segment budgétaire (KMeans non supervisé, fit sur TRAIN uniquement, k=2)\n")
        f.write(f"  Variable cible (à prédire) : ROI\n")
        f.write(f"  Taille Train : {X_train.shape[0]} | Taille Test : {X_test.shape[0]}\n\n")
        f.write("2. MÉTRIQUES DE PERFORMANCE (calculées sur le TEST, jamais vu à l'entraînement)\n" + "-" * 50 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n3. IMPORTANCE DES VARIABLES (Random Forest)\n" + "-" * 50 + "\n")
        for _, row in importance_df.iterrows():
            f.write(f"  {row['Variable']:25s}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)\n")
        f.write("\n4. ÉCHANTILLON DE PRÉDICTIONS (10 premières lignes)\n" + "-" * 50 + "\n")
        preds_df = pd.DataFrame(predictions)
        f.write(preds_df.head(10).round(2).to_string(index=False))
        f.write("\n")

    print(f"\nRapport sauvegardé : {report_path}")
    preds_df.to_csv("data/Predictions_ROI.csv", index=False)

    # =========================================================================
    # 6. VISUALISATIONS COMPARATIVES
    # =========================================================================
    print("\nGénération des visualisations comparatives...")
    os.makedirs("plots", exist_ok=True)
    all_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width / 2, results_df["R² Train"], width, label="R² Train", color=all_colors, alpha=0.6, edgecolor="black")
    ax.bar(x + width / 2, results_df["R² Test"], width, label="R² Test", color=all_colors, alpha=1.0, edgecolor="black")
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Comparaison R² Train vs R² Test (Détection Surapprentissage)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Modèle"], rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/roi_r2_train_vs_test.png", dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Métriques d'Erreur des Modèles de Régression du ROI (Test set)", fontsize=14, fontweight="bold")
    ax1.barh(results_df["Modèle"], results_df["RMSE (Test)"], color=all_colors)
    ax1.set_xlabel("RMSE (plus bas = meilleur)")
    ax2.barh(results_df["Modèle"], results_df["MAE (Test)"], color=all_colors)
    ax2.set_xlabel("MAE (plus bas = meilleur)")
    plt.tight_layout()
    plt.savefig("plots/roi_rmse_mae_comparison.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("ROI Réel vs ROI Prédit (Test set)", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    for i, (name, color) in enumerate(zip(models.keys(), all_colors)):
        ax = axes[i]
        ax.scatter(y_test, predictions[name], alpha=0.4, color=color, s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Prédiction Parfaite")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("ROI Réel (%)")
        ax.set_ylabel("ROI Prédit (%)")
        ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/roi_scatter_predictions.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Variable"], importance_df["Importance"], color=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"])
    ax.set_xlabel("Importance")
    ax.set_title("Importance des Variables pour la Prédiction du ROI (Random Forest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/roi_feature_importance.png", dpi=300)
    plt.close()

    print("Toutes les visualisations sauvegardées dans 'plots/'.")
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX (test set)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    roi_regression_pipeline()

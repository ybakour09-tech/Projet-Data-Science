"""
sales_regression.py
====================
Régression du VOLUME DE VENTES (Sales) à partir de la combinaison COMPLÈTE des
budgets marketing : TV + Radio + Social Media (aucun canal ignoré).

Contexte : l'ancien script train_models.py ne prédisait Sales qu'à partir de
TV seule, ignorant Radio et Social Media, alors que le sujet demande
explicitement de prédire "le volume de ventes généré par une COMBINAISON de
budgets". Ce script corrige ce point.

Méthodologie (cohérente avec scripts/roi_regression.py) :
  1. Split train/test D'ABORD.
  2. StandardScaler fit sur le TRAIN uniquement (transform sur le test).
  3. 4 modèles comparés : 3 Machine Learning (Random Forest, SVR linéaire,
     SGD linéaire) + 1 Deep Learning (MLP - réseau de neurones multicouche).
  4. Évaluation sur le test set uniquement, avec détection du surapprentissage
     (écart R² train/test).
  5. Importance des variables (Random Forest) pour visualiser la contribution
     réelle de chaque canal à la prédiction des ventes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["TV", "Radio", "Social Media"]


def sales_regression_pipeline(file_path="data/Clean_Data_HSS.csv"):
    print(f"Chargement des données : {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dimensions : {df.shape}")
    print(f"Variables utilisées (les 3 canaux, aucun n'est ignoré) : {FEATURES}")

    # =========================================================================
    # 1. SPLIT TRAIN / TEST EN PREMIER
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : SÉPARATION DES DONNÉES")
    print("=" * 60)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train : {df_train.shape[0]} lignes | Test : {df_test.shape[0]} lignes")

    # =========================================================================
    # 2. NORMALISATION (fit sur train uniquement)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : NORMALISATION (StandardScaler, fit sur TRAIN uniquement)")
    print("=" * 60)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURES])
    X_test = scaler.transform(df_test[FEATURES])  # transform uniquement, pas de fit

    y_train = df_train["Sales"].values
    y_test = df_test["Sales"].values

    # =========================================================================
    # 3. MODÉLISATION : 3 modèles Machine Learning + 1 modèle Deep Learning
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES (3 ML + 1 DL)")
    print("=" * 60)

    models = {
        "Random Forest (ML)": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR Linéaire (ML)": SVR(kernel="linear", C=1.0),
        "SGD Linéaire (ML)": SGDRegressor(
            loss="squared_error", learning_rate="invscaling", max_iter=2000, random_state=42
        ),
        "MLP Réseau de Neurones (DL)": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
        ),
    }

    results = []
    predictions = {"Sales_Réel": y_test}

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
    # 4. IMPORTANCE RÉELLE DE CHAQUE CANAL (Random Forest + coefficients SGD)
    # =========================================================================
    rf_model = models["Random Forest (ML)"]
    importance_df = pd.DataFrame(
        {"Canal": FEATURES, "Importance (Random Forest)": rf_model.feature_importances_}
    ).sort_values(by="Importance (Random Forest)", ascending=False)

    sgd_model = models["SGD Linéaire (ML)"]
    coef_df = pd.DataFrame({"Canal": FEATURES, "Coefficient (SGD, standardisé)": sgd_model.coef_})

    # =========================================================================
    # 5. RAPPORT TEXTUEL
    # =========================================================================
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/rapport_regression_sales_multicanaux.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT : PRÉDICTION DU VOLUME DE VENTES (RÉGRESSION MULTICANAUX)\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. PREPROCESSING\n" + "-" * 50 + "\n")
        f.write(f"  Variables explicatives (les 3 canaux, aucun ignoré) : {FEATURES}\n")
        f.write("  Normalisation : StandardScaler, fit sur TRAIN uniquement\n")
        f.write("  Variable cible : Sales\n")
        f.write(f"  Taille Train : {X_train.shape[0]} | Taille Test : {X_test.shape[0]}\n\n")
        f.write("2. MÉTRIQUES DE PERFORMANCE (calculées sur le TEST, jamais vu à l'entraînement)\n" + "-" * 50 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n3. IMPORTANCE RÉELLE DE CHAQUE CANAL (Random Forest)\n" + "-" * 50 + "\n")
        for _, row in importance_df.iterrows():
            pct = row["Importance (Random Forest)"] * 100
            f.write(f"  {row['Canal']:20s}: {row['Importance (Random Forest)']:.4f} ({pct:.1f}%)\n")
        f.write("\n4. COEFFICIENTS DU MODÈLE LINÉAIRE (SGD, sur variables standardisées)\n" + "-" * 50 + "\n")
        f.write(
            "  Interprétation : signe = sens de l'effet, magnitude = force de l'effet\n"
            "  (comparable entre canaux car les variables sont standardisées).\n\n"
        )
        for _, row in coef_df.iterrows():
            f.write(f"  {row['Canal']:20s}: {row['Coefficient (SGD, standardisé)']:+.4f}\n")
        f.write("\n5. ÉCHANTILLON DE PRÉDICTIONS (10 premières lignes)\n" + "-" * 50 + "\n")
        preds_df = pd.DataFrame(predictions)
        f.write(preds_df.head(10).round(2).to_string(index=False))
        f.write(
            "\n\n6. NOTE MÉTHODOLOGIQUE\n" + "-" * 50 + "\n"
            "  Contrairement à l'ancienne version (train_models.py, dépréciée), ce script\n"
            "  n'ignore aucun canal budgétaire. Si Random Forest attribue une importance\n"
            "  très dominante à TV, ce n'est pas un biais de modélisation : c'est un\n"
            "  résultat empirique constaté sur ce dataset (voir Analyse_Fonctionnelle_\n"
            "  Projet.txt), à interpréter comme une caractéristique du jeu de données\n"
            "  Kaggle synthétique utilisé, et non comme une décision arbitraire du code.\n"
        )

    print(f"\nRapport sauvegardé : {report_path}")
    preds_df.to_csv("data/Predictions_Sales.csv", index=False)

    # =========================================================================
    # 6. VISUALISATIONS
    # =========================================================================
    print("\nGénération des visualisations...")
    os.makedirs("plots", exist_ok=True)
    all_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width / 2, results_df["R² Train"], width, label="R² Train", color=all_colors, alpha=0.6, edgecolor="black")
    ax.bar(x + width / 2, results_df["R² Test"], width, label="R² Test", color=all_colors, alpha=1.0, edgecolor="black")
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Prédiction des Ventes : R² Train vs R² Test (Détection Surapprentissage)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Modèle"], rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/sales_r2_train_vs_test.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Canal"], importance_df["Importance (Random Forest)"], color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_xlabel("Importance")
    ax.set_title("Contribution Réelle de Chaque Canal à la Prédiction des Ventes", fontsize=12, fontweight="bold")
    for i, v in enumerate(importance_df["Importance (Random Forest)"]):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/sales_feature_importance.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Ventes Réelles vs Ventes Prédites (Test set)", fontsize=16, fontweight="bold")
    axes = axes.flatten()
    for i, (name, color) in enumerate(zip(models.keys(), all_colors)):
        ax = axes[i]
        ax.scatter(y_test, predictions[name], alpha=0.4, color=color, s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Prédiction Parfaite")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Sales Réel")
        ax.set_ylabel("Sales Prédit")
        ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/sales_scatter_predictions.png", dpi=300)
    plt.close()

    print("Visualisations sauvegardées dans 'plots/'.")
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX (test set)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nImportance des canaux (Random Forest) :")
    print(importance_df.to_string(index=False))

    return results_df, importance_df


if __name__ == "__main__":
    sales_regression_pipeline()

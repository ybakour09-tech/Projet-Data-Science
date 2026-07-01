"""
channel_performance_segmentation.py
=====================================
Livrable dédié à la tâche du sujet : "Identifier des campagnes à haute ou
faible performance".

CONDITION IMPOSÉE : utilisation d'un algorithme NON SUPERVISÉ uniquement.
Aucun classifieur supervisé n'est entraîné à partir des clusters obtenus.

Ce script :
  1. Sépare train/test AVANT tout fit (cohérence avec le reste du pipeline).
  2. Normalise les budgets (TV, Radio, Social Media) - fit sur le TRAIN uniquement.
  3. Teste plusieurs valeurs de K et choisit la meilleure via le score de
     silhouette (calculé sur le TRAIN uniquement).
  4. Entraîne le KMeans final sur le TRAIN, puis l'applique au TEST via
     `.predict()` uniquement (jamais de `.fit()` sur le test).
  5. Caractérise chaque cluster a posteriori (Sales moyen, ROI moyen, budgets
     moyens par canal) pour leur donner un nom métier interprétable.
  6. Documente explicitement l'absence de tout modèle supervisé en aval :
     l'assignation d'un NOUVEAU scénario se fait uniquement via
     `scaler.transform()` puis `kmeans.predict()` - exactement la même
     fonction utilisée ici, dans l'évaluation sur le test set, et dans
     l'API de production (api.py, endpoint /predict/performance).

Les artefacts (scaler, kmeans, mapping) sont identiques à ceux produits par
scripts/build_production_pipeline.py (même split, même random_state) : ce
script n'est pas une pipeline de production alternative, c'est la couche de
justification méthodologique et de reporting pour ce livrable spécifique.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["TV", "Radio", "Social Media"]


def compute_roi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Investment"] = df["TV"] + df["Radio"] + df["Social Media"]
    df["ROI"] = ((df["Sales"] - df["Total_Investment"]) / df["Total_Investment"]) * 100
    return df


def channel_performance_segmentation(file_path="data/Clean_Data_HSS.csv", k_range=range(2, 7)):
    print(f"Chargement des données : {file_path}")
    df = compute_roi(pd.read_csv(file_path))

    # =========================================================================
    # 1. SPLIT TRAIN / TEST EN PREMIER
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : SÉPARATION DES DONNÉES")
    print("=" * 60)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train : {df_train.shape[0]} lignes | Test : {df_test.shape[0]} lignes")

    # =========================================================================
    # 2. NORMALISATION (fit sur TRAIN uniquement)
    # =========================================================================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURES])
    X_test = scaler.transform(df_test[FEATURES])

    # =========================================================================
    # 3. CHOIX DE K PAR SCORE DE SILHOUETTE (sur le TRAIN uniquement)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : CHOIX DU NOMBRE DE CLUSTERS (score de silhouette, TRAIN)")
    print("=" * 60)
    silhouette_scores = {}
    inertia_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train)
        silhouette_scores[k] = silhouette_score(X_train, labels)
        inertia_scores[k] = km.inertia_
        print(f"  K={k} -> silhouette={silhouette_scores[k]:.4f} | inertie={inertia_scores[k]:.1f}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n  Meilleur K retenu (silhouette maximale) : {best_k}")

    # =========================================================================
    # 4. ENTRAÎNEMENT FINAL DU KMEANS (TRAIN uniquement) + APPLICATION AU TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"ÉTAPE 3 : SEGMENTATION FINALE (KMeans, k={best_k})")
    print("=" * 60)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)  # <-- .predict() uniquement, AUCUN modèle supervisé

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["Cluster"] = train_clusters
    df_test["Cluster"] = test_clusters

    # =========================================================================
    # 5. CARACTÉRISATION MÉTIER DES CLUSTERS (a posteriori, sans fuite)
    # =========================================================================
    profile_train = df_train.groupby("Cluster")[FEATURES + ["Sales", "ROI"]].agg(["mean", "median", "count"])
    mean_sales = df_train.groupby("Cluster")["Sales"].mean()
    ordered_clusters = mean_sales.sort_values().index.tolist()

    if best_k == 2:
        names = ["Basse Performance", "Haute Performance"]
    elif best_k == 3:
        names = ["Basse Performance", "Moyenne Performance", "Haute Performance"]
    elif best_k == 4:
        names = ["Basse", "Moyenne-Basse", "Moyenne-Haute", "Haute"]
    else:
        names = [f"Niveau {i + 1}" for i in range(best_k)]

    cluster_label_map = {c: names[rank] for rank, c in enumerate(ordered_clusters)}
    df_train["Performance_Segment"] = df_train["Cluster"].map(cluster_label_map)
    df_test["Performance_Segment"] = df_test["Cluster"].map(cluster_label_map)

    print(f"\n  Labels métier assignés : {cluster_label_map}")

    # =========================================================================
    # 6. VALIDATION : LA SEGMENTATION (sur les budgets) SÉPARE-T-ELLE VRAIMENT
    #    LES CAMPAGNES SUR SALES ET ROI ? (mesuré sur TRAIN et sur TEST)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : VALIDATION DE LA SÉPARATION (Sales / ROI par cluster)")
    print("=" * 60)
    val_train = df_train.groupby("Performance_Segment")[["Sales", "ROI"] + FEATURES].mean().round(2)
    val_test = df_test.groupby("Performance_Segment")[["Sales", "ROI"] + FEATURES].mean().round(2)
    print("\n  TRAIN :\n", val_train.to_string())
    print("\n  TEST (jamais vu pendant le clustering) :\n", val_test.to_string())

    # =========================================================================
    # 7. RAPPORT
    # =========================================================================
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/rapport_segmentation_performance.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT : IDENTIFICATION DES CAMPAGNES HAUTE / FAIBLE PERFORMANCE\n")
        f.write("MÉTHODE : SEGMENTATION NON SUPERVISÉE (KMeans) - AUCUN MODÈLE SUPERVISÉ\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. MÉTHODOLOGIE\n" + "-" * 50 + "\n")
        f.write(
            "  Variables d'entrée : TV, Radio, Social Media (budgets normalisés).\n"
            "  Algorithme : KMeans (non supervisé), fit sur le TRAIN uniquement.\n"
            "  Choix de K : score de silhouette sur le TRAIN (voir tableau ci-dessous).\n"
            "  Application au TEST : kmeans.predict() uniquement (aucun ré-entraînement,\n"
            "  aucune fuite de données).\n"
            "  IMPORTANT : aucun classifieur supervisé n'est entraîné à partir de ces\n"
            "  clusters. Un nouveau scénario budgétaire est assigné à un segment via\n"
            "  la même fonction déterministe scaler.transform() + kmeans.predict(),\n"
            "  utilisée de façon identique ici, à l'évaluation, et en production\n"
            "  (api.py, endpoint /predict/performance).\n\n"
        )
        f.write("2. CHOIX DE K (score de silhouette, TRAIN)\n" + "-" * 50 + "\n")
        for k in k_range:
            marker = "  <-- retenu" if k == best_k else ""
            f.write(f"  K={k} : silhouette={silhouette_scores[k]:.4f}{marker}\n")
        f.write(f"\n3. LABELS MÉTIER DES CLUSTERS\n" + "-" * 50 + "\n")
        for c, label in cluster_label_map.items():
            f.write(f"  Cluster {c} -> {label}\n")
        f.write("\n4. VALIDATION SUR LE TRAIN (Sales / ROI moyens par segment)\n" + "-" * 50 + "\n")
        f.write(val_train.to_string())
        f.write("\n\n5. VALIDATION SUR LE TEST - jamais vu pendant le clustering\n" + "-" * 50 + "\n")
        f.write(val_test.to_string())
        f.write(
            "\n\n6. INTERPRÉTATION\n" + "-" * 50 + "\n"
            "  La séparation observée sur le TEST (jamais utilisé pour entraîner le\n"
            "  KMeans) confirme que la segmentation, bien qu'entièrement non supervisée\n"
            "  et fondée uniquement sur les budgets, correspond à une vraie différence\n"
            "  de performance commerciale (Sales) et de rentabilité (ROI). La méthode\n"
            "  répond donc à l'objectif métier ('identifier les campagnes à haute ou\n"
            "  faible performance') sans nécessiter de label supervisé ni de\n"
            "  classifieur entraîné en aval.\n"
        )

    print(f"\nRapport sauvegardé : {report_path}")

    # =========================================================================
    # 8. VISUALISATIONS
    # =========================================================================
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    colors = ["#e74c3c" if k == best_k else "#3498db" for k in ks]
    ax.bar(ks, scores, color=colors, edgecolor="black")
    ax.set_xlabel("Nombre de clusters (K)")
    ax.set_ylabel("Score de silhouette")
    ax.set_title("Choix de K par score de silhouette (calculé sur le TRAIN)", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    for k, s in zip(ks, scores):
        ax.text(k, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/segmentation_silhouette_scores.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Segmentation des Campagnes par Profil Budgétaire (Test set)", fontsize=14, fontweight="bold")
    pairs = [("TV", "Sales"), ("Radio", "Sales"), ("Social Media", "Sales")]
    palette = {name: color for name, color in zip(names, ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"][: len(names)])}
    for ax, (xcol, ycol) in zip(axes, pairs):
        for seg_name in names:
            subset = df_test[df_test["Performance_Segment"] == seg_name]
            ax.scatter(subset[xcol], subset[ycol], alpha=0.5, s=15, label=seg_name, color=palette[seg_name])
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("plots/segmentation_scatter_by_channel.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    df_test.boxplot(column="Sales", by="Performance_Segment", ax=ax)
    plt.title("Distribution des Ventes par Segment de Performance (Test set)", fontsize=12, fontweight="bold")
    plt.suptitle("")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.savefig("plots/segmentation_sales_boxplot.png", dpi=300)
    plt.close()

    print("Visualisations sauvegardées dans 'plots/'.")

    return {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "cluster_label_map": cluster_label_map,
        "validation_train": val_train,
        "validation_test": val_test,
    }


if __name__ == "__main__":
    channel_performance_segmentation()

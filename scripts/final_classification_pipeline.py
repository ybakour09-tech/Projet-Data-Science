# ⚠️ DÉPRÉCIÉ : ce script entraînait un classifieur SUPERVISÉ pour ré-apprendre
# des labels "Performance" eux-mêmes issus d'un clustering sur Sales. Problème :
# le modèle de régression ROI en production voyait le VRAI label à
# l'entraînement, mais le label PRÉDIT (potentiellement faux) en production
# -> décalage train/serving non documenté.
# Remplacé par une segmentation 100% non supervisée (KMeans direct sur les
# budgets) dans scripts/build_production_pipeline.py, utilisée de façon
# identique à l'entraînement, à l'évaluation et en production (API).
# Conservé ici uniquement à titre de comparaison / historique.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

def final_pipeline(file_path):
    print(f"Chargement des données : {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Préparation des données
    X_raw = df[['TV', 'Radio', 'Social Media']]
    y_raw = df['Performance']
    
    print("Encodage de la cible (Performance) avec LabelEncoder...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"Classes détectées : {encoder.classes_} -> {encoder.transform(encoder.classes_)}")
    
    print("Normalisation des variables numériques avec StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("\nSéparation des données en Train (80%) et Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 2. Modélisation
    models = {
        'Random Forest (ML)': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC (ML)': SVC(random_state=42),
        'MLP Classifier (DL)': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
        'SGD Classifier (DL)': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    }
    
    reports = {}
    matrices = {}
    
    report_file = "rapport_classification_finale.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("RAPPORTS FINAUX DE CLASSIFICATION (Données Normalisées + Encodées)\n")
        f.write("================================================================================\n\n")
        f.write(f"Mapping des classes : {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}\n\n")
        
        print("\nEntraînement des modèles en cours...")
        for name, model in models.items():
            print(f" - Modélisation avec {name}...")
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédiction
            y_pred = model.predict(X_test)
            
            # Rapport
            clf_report = classification_report(y_test, y_pred, target_names=encoder.classes_)
            reports[name] = clf_report
            
            # Matrice de confusion
            matrices[name] = confusion_matrix(y_test, y_pred, labels=encoder.transform(encoder.classes_))
            
            # Sauvegarde textuelle
            f.write(f"--- Modèle : {name} ---\n")
            f.write(clf_report)
            f.write("\n")
            
    print(f"\nRapports générés et sauvegardés dans : {report_file}")
    
    # 3. Visualisation
    print("Génération des matrices de confusion (visualisation)...")
    os.makedirs("plots", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Matrices de Confusion - Pipeline Final (ML & DL)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, name in enumerate(models.keys()):
        sns.heatmap(matrices[name], annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=axes[i])
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Prédiction')
        axes[i].set_ylabel('Vérité Terrain')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/final_confusion_matrices.png", dpi=300)
    plt.close()
    
    print("Matrices de confusion sauvegardées dans 'plots/final_confusion_matrices.png'.")

if __name__ == "__main__":
    final_pipeline("data/Classified_Data_HSS.csv")

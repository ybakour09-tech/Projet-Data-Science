import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_classification_models(file_path):
    print(f"Chargement du jeu de données classifié : {file_path}")
    df = pd.read_csv(file_path)
    
    X = df[['TV', 'Radio', 'Social Media']]
    y = df['Performance']
    
    print("\nSéparation des données en Train (80%) et Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Régression Logistique': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Classifier (SVC)': SVC(random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier()
    }
    
    reports = {}
    matrices = {}
    
    # Text report file setup
    report_file = "rapport_classification_detaille.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("RAPPORTS DÉTAILLÉS DE CLASSIFICATION\n")
        f.write("================================================================================\n\n")
        
        print("\nEntraînement des modèles en cours...")
        for name, model in models.items():
            print(f" - Modélisation avec {name}...")
            # Training
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Generating classification report
            clf_report = classification_report(y_test, y_pred)
            reports[name] = clf_report
            
            # Confusion matrix
            matrices[name] = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            
            # Write to file
            f.write(f"--- Modèle : {name} ---\n")
            f.write(clf_report)
            f.write("\n")
            
    print(f"\nRapports textuels générés et sauvegardés dans : {report_file}")
    
    # Plotting confusion matrices
    print("Génération des matrices de confusion (visualisation)...")
    os.makedirs("plots", exist_ok=True)
    
    labels = np.unique(y)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Matrices de Confusion - Classification des Campagnes', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, name in enumerate(models.keys()):
        sns.heatmap(matrices[name], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Prédiction')
        axes[i].set_ylabel('Vérité Terrain')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/confusion_matrices_classification.png", dpi=300)
    plt.close()
    
    print("Matrices de confusion sauvegardées dans 'plots/confusion_matrices_classification.png'.")

if __name__ == "__main__":
    train_classification_models("data/Classified_Data_HSS.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate():
    print("Chargement des données normalisées...")
    df = pd.read_csv("data/Normalized_Data_HSS.csv")
    
    X = df[['TV_norm']]
    y = df['Sales_norm']
    
    # 1. SPLIT DES DONNÉES
    print("Séparation des données en Train (80%) et Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialisation des modèles
    models = {
        # Machine Learning Classique
        'Régression Linéaire (ML)': LinearRegression(),
        'SVR Noyau Linéaire (ML)': SVR(kernel='linear', C=1.0),
        
        # Deep Learning (implémenté via les solveurs de Scikit-Learn pour éviter les dépendances lourdes)
        # MLP: Multi-Layer Perceptron (Réseau de neurones profond avec 2 couches cachées)
        'MLP / Réseau Profond (DL)': MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500, random_state=42),
        # Réseau Linéaire via Descente de Gradient (Equivalent d'un neurone unique sans activation non-linéaire)
        'Réseau Linéaire SGD (DL)': SGDRegressor(loss='squared_error', learning_rate='invscaling', max_iter=1000, random_state=42)
    }
    
    metrics_list = []
    predictions_dict = {'Actual_Sales': y_test.values, 'TV_Input': X_test['TV_norm'].values}
    
    print("\nEntraînement des modèles...")
    # 2. ENTRAÎNEMENT ET PRÉDICTION
    for name, model in models.items():
        print(f" - Entraînement de : {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        predictions_dict[name] = preds
        
        # Calcul des métriques
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        metrics_list.append({
            'Modèle': name,
            'R2 Score': r2,
            'RMSE': rmse,
            'MAE': mae
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    preds_df = pd.DataFrame(predictions_dict)
    
    # 3. CRÉATION DU RAPPORT DE PERFORMANCE (MARKDOWN)
    report_path = "rapport_predictions_performances.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Rapport de Prédiction et Performances des Modèles\n\n")
        f.write("Ce rapport compare 2 modèles de Machine Learning et 2 modèles de Deep Learning sur la prédiction des ventes.\n\n")
        f.write("## 1. Métriques de Performance\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n## 2. Échantillon des Prédictions (5 premières lignes)\n\n")
        f.write(preds_df.head(5).to_markdown(index=False))
        f.write("\n\n**Note :** Les valeurs sont normalisées (StandardScaler).\n")
        
    # Sauvegarde complète des prédictions en CSV
    preds_path = "data/Predictions_Comparees.csv"
    preds_df.to_csv(preds_path, index=False)
    
    # 4. VISUALISATIONS
    print("\nGénération des visualisations...")
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Bar chart des métriques
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    ax1.bar(x - width/2, metrics_df['R2 Score'], width, label='R² Score', color='skyblue')
    ax1.set_ylabel('R² Score', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0.998, 1.0) # Zoom car les R2 sont très hauts
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, metrics_df['RMSE'], width, label='RMSE', color='salmon')
    ax2.set_ylabel('RMSE', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Modèle'], rotation=15, ha='right', fontsize=10)
    plt.title('Comparaison des Performances (R² et RMSE) des 4 Modèles', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.savefig("plots/model_metrics_comparison.png", dpi=300)
    plt.close()
    
    # Plot 2: Scatter plot (Actual vs Predicted)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Valeurs Réelles vs Prédictions (Données Normalisées)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['blue', 'green', 'purple', 'red']
    
    for i, (name, color) in enumerate(zip(models.keys(), colors)):
        ax = axes[i]
        ax.scatter(y_test, predictions_dict[name], alpha=0.5, color=color)
        # Ligne parfaite
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Ventes Réelles")
        ax.set_ylabel("Ventes Prédites")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/model_scatter_comparison.png", dpi=300)
    plt.close()

    print(f"\nRapport généré avec succès : {report_path}")
    print(f"Prédictions sauvegardées : {preds_path}")
    print("Graphiques sauvegardés dans le dossier 'plots/'.")

if __name__ == "__main__":
    train_and_evaluate()

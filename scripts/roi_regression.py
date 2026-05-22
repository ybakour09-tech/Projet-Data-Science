import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def roi_regression_pipeline(file_path):
    print(f"Chargement des données : {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Dimensions : {df.shape}")
    print(f"Colonnes : {df.columns.tolist()}")
    
    # =========================================================================
    # 1. PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 1 : PREPROCESSING")
    print("=" * 60)
    
    features_num = ['TV', 'Radio', 'Social Media']
    target = 'ROI'
    
    # --- Encodage de la variable catégorielle ---
    print("\n[Encodage] Colonne catégorielle 'Performance' -> LabelEncoder")
    encoder = LabelEncoder()
    df['Performance_encoded'] = encoder.fit_transform(df['Performance'])
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(f"  Mapping : {mapping}")
    
    # --- Définition des features ---
    feature_cols = features_num + ['Performance_encoded']
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # --- Normalisation des colonnes numériques ---
    print("\n[Normalisation] Colonnes numériques -> StandardScaler")
    scaler = StandardScaler()
    X[features_num] = scaler.fit_transform(X[features_num])
    
    print("\n  Vérification (moyenne ~ 0, écart-type ~ 1) :")
    for col in features_num:
        print(f"    {col:15s} -> Moyenne: {X[col].mean():.4f} | Std: {X[col].std():.4f}")
    
    # =========================================================================
    # 2. SPLIT TRAIN / TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 2 : SÉPARATION DES DONNÉES")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train : {X_train.shape[0]} lignes")
    print(f"  Test  : {X_test.shape[0]} lignes")
    
    # =========================================================================
    # 3. MODÉLISATION (2 ML + 2 DL)
    # =========================================================================
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)
    
    models = {
        # Machine Learning
        'Random Forest (ML)': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR Linéaire (ML)': SVR(kernel='linear', C=1.0),
        # Deep Learning
        'MLP Réseau Profond (DL)': MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        ),
        'Réseau Linéaire SGD (DL)': SGDRegressor(
            loss='squared_error',
            learning_rate='invscaling',
            max_iter=2000,
            random_state=42
        )
    }
    
    results = []
    predictions = {'ROI_Réel': y_test.values}
    
    for name, model in models.items():
        print(f"\n  Entraînement : {name}...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        predictions[name] = y_pred_test
        
        # Métriques sur Train ET Test (pour détecter le surapprentissage)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        results.append({
            'Modèle': name,
            'R² Train': round(r2_train, 6),
            'R² Test': round(r2_test, 6),
            'RMSE (Test)': round(rmse_test, 4),
            'MAE (Test)': round(mae_test, 4),
            'Écart Train/Test': round(abs(r2_train - r2_test), 6)
        })
        
        print(f"    R² Train: {r2_train:.6f} | R² Test: {r2_test:.6f} | RMSE: {rmse_test:.4f} | MAE: {mae_test:.4f}")
        
    results_df = pd.DataFrame(results)
    
    # =========================================================================
    # 4. IMPORTANCE DES VARIABLES (Random Forest)
    # =========================================================================
    rf_model = models['Random Forest (ML)']
    importance_df = pd.DataFrame({
        'Variable': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # =========================================================================
    # 5. RAPPORT TEXTUEL
    # =========================================================================
    report_path = "rapport_regression_roi.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT COMPLET : ESTIMATION DU ROI PAR MIX MÉDIA (RÉGRESSION)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. PREPROCESSING\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Variables numériques normalisées (StandardScaler) : {features_num}\n")
        f.write(f"  Variable catégorielle encodée (LabelEncoder) : Performance\n")
        f.write(f"    Mapping : {mapping}\n")
        f.write(f"  Variable cible (à prédire) : {target}\n")
        f.write(f"  Taille Train : {X_train.shape[0]} | Taille Test : {X_test.shape[0]}\n\n")
        
        f.write("2. MÉTRIQUES DE PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("3. IMPORTANCE DES VARIABLES (Random Forest)\n")
        f.write("-" * 50 + "\n")
        for _, row in importance_df.iterrows():
            f.write(f"  {row['Variable']:25s}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)\n")
        f.write("\n")
        
        f.write("4. ÉCHANTILLON DE PRÉDICTIONS (10 premières lignes)\n")
        f.write("-" * 50 + "\n")
        preds_df = pd.DataFrame(predictions)
        f.write(preds_df.head(10).round(2).to_string(index=False))
        f.write("\n")
        
    print(f"\nRapport sauvegardé : {report_path}")
    
    # Sauvegarde prédictions complètes
    preds_df.to_csv("data/Predictions_ROI.csv", index=False)
    
    # =========================================================================
    # 6. VISUALISATIONS COMPARATIVES
    # =========================================================================
    print("\nGénération des visualisations comparatives...")
    os.makedirs("plots", exist_ok=True)
    
    colors_ml = ['#3498db', '#2ecc71']
    colors_dl = ['#9b59b6', '#e74c3c']
    all_colors = colors_ml + colors_dl
    
    # --- PLOT 1 : Bar chart R² Train vs Test (détection overfitting) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    bars_train = ax.bar(x - width/2, results_df['R² Train'], width, label='R² Train', color=all_colors, alpha=0.6, edgecolor='black')
    bars_test = ax.bar(x + width/2, results_df['R² Test'], width, label='R² Test', color=all_colors, alpha=1.0, edgecolor='black')
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Comparaison R² Train vs R² Test (Détection Surapprentissage)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Modèle'], rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/roi_r2_train_vs_test.png", dpi=300)
    plt.close()
    
    # --- PLOT 2 : Bar chart RMSE + MAE côte à côte ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Métriques d'Erreur des Modèles de Régression du ROI", fontsize=14, fontweight='bold')
    
    ax1.barh(results_df['Modèle'], results_df['RMSE (Test)'], color=all_colors)
    ax1.set_xlabel('RMSE (plus bas = meilleur)')
    ax1.set_title('RMSE sur le Test Set')
    for i, v in enumerate(results_df['RMSE (Test)']):
        ax1.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=10)
    
    ax2.barh(results_df['Modèle'], results_df['MAE (Test)'], color=all_colors)
    ax2.set_xlabel('MAE (plus bas = meilleur)')
    ax2.set_title('MAE sur le Test Set')
    for i, v in enumerate(results_df['MAE (Test)']):
        ax2.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("plots/roi_rmse_mae_comparison.png", dpi=300)
    plt.close()
    
    # --- PLOT 3 : Scatter Réel vs Prédit (4 modèles) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("ROI Réel vs ROI Prédit (Données de Test)", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, (name, color) in enumerate(zip(models.keys(), all_colors)):
        ax = axes[i]
        ax.scatter(y_test, predictions[name], alpha=0.4, color=color, s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Prédiction Parfaite')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel("ROI Réel (%)")
        ax.set_ylabel("ROI Prédit (%)")
        ax.legend(fontsize=9)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/roi_scatter_predictions.png", dpi=300)
    plt.close()
    
    # --- PLOT 4 : Feature Importance (Random Forest) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df['Variable'], importance_df['Importance'], color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    ax.set_xlabel('Importance')
    ax.set_title("Importance des Variables pour la Prédiction du ROI (Random Forest)", fontsize=13, fontweight='bold')
    for i, v in enumerate(importance_df['Importance']):
        ax.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/roi_feature_importance.png", dpi=300)
    plt.close()
    
    print("Toutes les visualisations sauvegardées dans 'plots/'.")
    
    # Affichage final
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    roi_regression_pipeline("data/Classified_Data_HSS.csv")

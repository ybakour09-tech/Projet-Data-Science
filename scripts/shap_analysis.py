import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def shap_analysis(file_path):
    print("=" * 70)
    print("ANALYSE SHAP : EXPLICABILITÉ DES CANAUX MARKETING SUR LE ROI")
    print("=" * 70)
    
    # 1. Chargement et preprocessing (identique au pipeline précédent)
    print("\n[1/6] Chargement et preprocessing des données...")
    df = pd.read_csv(file_path)
    
    features_num = ['TV', 'Radio', 'Social Media']
    target = 'ROI'
    
    encoder = LabelEncoder()
    df['Performance_encoded'] = encoder.fit_transform(df['Performance'])
    
    feature_cols = features_num + ['Performance_encoded']
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    scaler = StandardScaler()
    X[features_num] = scaler.fit_transform(X[features_num])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Entraînement des deux modèles
    print("[2/6] Entraînement du Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print("[3/6] Entraînement du MLP (Réseau Profond)...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu',
                       solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    
    os.makedirs("plots", exist_ok=True)
    
    # =========================================================================
    # 3. SHAP sur Random Forest (TreeExplainer - exact et rapide)
    # =========================================================================
    print("[4/6] Calcul des valeurs SHAP (Random Forest - TreeExplainer)...")
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(X_test)
    
    # --- PLOT 1 : Summary Plot (Beeswarm) ---
    print("  -> Génération du Summary Plot (RF)...")
    plt.figure()
    shap.summary_plot(shap_values_rf, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary Plot - Random Forest", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_summary_rf.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 2 : Bar Plot (importance moyenne) ---
    print("  -> Génération du Bar Plot d'importance SHAP (RF)...")
    plt.figure()
    shap.summary_plot(shap_values_rf, X_test, feature_names=feature_cols, 
                      plot_type="bar", show=False)
    plt.title("Importance Moyenne SHAP par Canal - Random Forest", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_bar_importance_rf.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOTS 3-5 : Dependence Plots (interaction entre canaux) ---
    print("  -> Génération des Dependence Plots (interactions entre canaux)...")
    
    # TV coloré par Radio
    plt.figure()
    shap.dependence_plot("TV", shap_values_rf, X_test, 
                         interaction_index="Radio",
                         feature_names=feature_cols, show=False)
    plt.title("Impact de TV sur le ROI (coloré par Radio)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_dependence_tv_radio.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Radio coloré par TV
    plt.figure()
    shap.dependence_plot("Radio", shap_values_rf, X_test, 
                         interaction_index="TV",
                         feature_names=feature_cols, show=False)
    plt.title("Impact de Radio sur le ROI (coloré par TV)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_dependence_radio_tv.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Social Media coloré par TV
    plt.figure()
    shap.dependence_plot("Social Media", shap_values_rf, X_test, 
                         interaction_index="TV",
                         feature_names=feature_cols, show=False)
    plt.title("Impact de Social Media sur le ROI (coloré par TV)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_dependence_social_tv.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 4. SHAP sur MLP (KernelExplainer - sur échantillon)
    # =========================================================================
    print("[5/6] Calcul des valeurs SHAP (MLP - KernelExplainer sur 200 échantillons)...")
    background = shap.sample(X_train, 100)
    X_test_sample = X_test.iloc[:200]
    
    explainer_mlp = shap.KernelExplainer(mlp.predict, background)
    shap_values_mlp = explainer_mlp.shap_values(X_test_sample)
    
    # --- PLOT 6 : Summary Plot MLP ---
    print("  -> Génération du Summary Plot (MLP)...")
    plt.figure()
    shap.summary_plot(shap_values_mlp, X_test_sample, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary Plot - MLP Réseau Profond", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/shap_summary_mlp.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PLOT 7 : Comparaison RF vs MLP (Bar) ---
    print("  -> Génération du graphique comparatif RF vs MLP...")
    mean_shap_rf = np.abs(shap_values_rf).mean(axis=0)
    mean_shap_mlp = np.abs(shap_values_mlp).mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(feature_cols))
    width = 0.35
    
    ax.bar(x - width/2, mean_shap_rf, width, label='Random Forest (ML)', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, mean_shap_mlp, width, label='MLP Réseau Profond (DL)', color='#9b59b6', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, fontsize=11)
    ax.set_ylabel('Valeur SHAP moyenne (|SHAP|)', fontsize=12)
    ax.set_title('Comparaison de l\'Importance SHAP : Random Forest vs MLP', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/shap_comparison_rf_vs_mlp.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # 5. ANALYSE PAR CONTEXTE BUDGÉTAIRE (combinaisons de canaux)
    # =========================================================================
    print("[6/6] Analyse SHAP par contexte budgétaire...")
    
    # On utilise les données non normalisées pour segmenter les budgets
    df_test = df.iloc[X_test.index].copy()
    df_test['SHAP_TV'] = shap_values_rf[:, 0]
    df_test['SHAP_Radio'] = shap_values_rf[:, 1]
    df_test['SHAP_Social'] = shap_values_rf[:, 2]
    
    # Segmentation des budgets en 3 tranches (Faible / Moyen / Élevé)
    for col in features_num:
        df_test[f'{col}_Niveau'] = pd.qcut(df_test[col], q=3, labels=['Faible', 'Moyen', 'Élevé'])
    
    # Tableau d'impact SHAP moyen par tranche de budget TV
    budget_analysis = df_test.groupby('TV_Niveau').agg(
        SHAP_TV_moyen=('SHAP_TV', 'mean'),
        SHAP_Radio_moyen=('SHAP_Radio', 'mean'),
        SHAP_Social_moyen=('SHAP_Social', 'mean'),
        ROI_moyen=('ROI', 'mean'),
        Nb_observations=('ROI', 'count')
    ).round(4)
    
    # Tableau croisé : combinaisons TV_Niveau x Radio_Niveau
    cross_analysis = df_test.groupby(['TV_Niveau', 'Radio_Niveau']).agg(
        ROI_moyen=('ROI', 'mean'),
        SHAP_TV_moyen=('SHAP_TV', 'mean'),
        SHAP_Radio_moyen=('SHAP_Radio', 'mean'),
        Nb=('ROI', 'count')
    ).round(2)
    
    # =========================================================================
    # 6. RAPPORT TEXTUEL
    # =========================================================================
    report_path = "rapport_shap_interpretabilite.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT SHAP : EXPLICABILITÉ ET INTERPRÉTABILITÉ DES CANAUX MARKETING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. IMPORTANCE GLOBALE DES CANAUX (Valeur SHAP moyenne absolue)\n")
        f.write("-" * 60 + "\n")
        f.write("  Random Forest :\n")
        for feat, val in zip(feature_cols, mean_shap_rf):
            f.write(f"    {feat:25s}: {val:.4f}\n")
        f.write("\n  MLP Réseau Profond :\n")
        for feat, val in zip(feature_cols, mean_shap_mlp):
            f.write(f"    {feat:25s}: {val:.4f}\n")
        
        f.write("\n\n2. IMPACT DES CANAUX SELON LE NIVEAU DE BUDGET TV\n")
        f.write("-" * 60 + "\n")
        f.write(budget_analysis.to_string())
        
        f.write("\n\n\n3. COMBINAISONS DE CANAUX (TV x Radio) ET IMPACT SUR LE ROI\n")
        f.write("-" * 60 + "\n")
        f.write(cross_analysis.to_string())
        
        f.write("\n\n\n4. INTERPRÉTATION BUSINESS\n")
        f.write("-" * 60 + "\n")
        
        # Identifier le canal le plus important
        top_channel_idx = np.argmax(mean_shap_rf[:3])
        top_channel = features_num[top_channel_idx]
        f.write(f"  - Le canal le plus influent sur le ROI est : {top_channel}\n")
        f.write(f"    (Valeur SHAP moyenne = {mean_shap_rf[top_channel_idx]:.4f})\n\n")
        
        # Analyser les interactions
        f.write("  - Les Dependence Plots révèlent les interactions entre canaux.\n")
        f.write("    Consultez les graphiques dans le dossier 'plots/' pour visualiser\n")
        f.write("    comment chaque canal augmente ou diminue le ROI en fonction\n")
        f.write("    du niveau d'investissement dans les autres canaux.\n")
        
    print(f"\nRapport SHAP sauvegardé : {report_path}")
    print("Tous les graphiques SHAP sauvegardés dans 'plots/'.")

if __name__ == "__main__":
    shap_analysis("data/Classified_Data_HSS.csv")

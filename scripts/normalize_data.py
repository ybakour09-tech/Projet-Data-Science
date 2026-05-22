import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalize_dataset(file_path):
    print(f"Chargement des données nettoyées depuis : {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Sélectionner uniquement TV et Sales
    print("Sélection de la variable explicative (TV) et de la cible (Sales)...")
    df_selected = df[['TV', 'Sales']]
    
    # 2. Normalisation (Standardisation : Moyenne = 0, Ecart-type = 1)
    print("Application de StandardScaler (Standardisation)...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # On garde les noms de colonnes originaux mais avec les valeurs normalisées
    tv_normalized = scaler_X.fit_transform(df_selected[['TV']])
    sales_normalized = scaler_y.fit_transform(df_selected[['Sales']])
    
    df_normalized = pd.DataFrame({
        'TV_norm': tv_normalized.flatten(),
        'Sales_norm': sales_normalized.flatten()
    })
    
    # 3. Vérification
    print("\nStatistiques après normalisation (vérification) :")
    print(df_normalized.describe().round(4))
    
    print("\nAperçu des premières lignes :")
    print(df_normalized.head())
    
    # 4. Sauvegarde
    output_path = "data/Normalized_Data_HSS.csv"
    df_normalized.to_csv(output_path, index=False)
    print(f"\nJeu de données normalisé et sauvegardé avec succès : {output_path}")

if __name__ == "__main__":
    normalize_dataset("data/Clean_Data_HSS.csv")

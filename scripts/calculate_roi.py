import pandas as pd

def calculate_roi(file_path):
    print(f"Chargement des données : {file_path}")
    df = pd.read_csv(file_path)
    
    # Calcul du coût total d'investissement publicitaire
    df['Total_Investment'] = df['TV'] + df['Radio'] + df['Social Media']
    
    # Formule du ROI : (Gain - Coût) / Coût * 100
    # Gain = Sales (chiffre d'affaires généré)
    # Coût = Total_Investment (somme des budgets TV + Radio + Social Media)
    df['ROI'] = ((df['Sales'] - df['Total_Investment']) / df['Total_Investment']) * 100
    
    # Affichage des statistiques
    print("\nStatistiques du ROI calculé :")
    print(df['ROI'].describe().round(2))
    
    print(f"\nAperçu des premières lignes :")
    print(df[['TV', 'Radio', 'Social Media', 'Sales', 'Total_Investment', 'ROI', 'Performance']].head(10).to_string(index=False))
    
    # Moyenne du ROI par catégorie de performance
    print("\nROI moyen par catégorie de Performance :")
    roi_by_perf = df.groupby('Performance')['ROI'].mean().round(2)
    for perf, roi in roi_by_perf.items():
        print(f"  {perf} : {roi}%")
    
    # Sauvegarde
    df.to_csv(file_path, index=False)
    print(f"\nColonne ROI ajoutée et fichier sauvegardé : {file_path}")

if __name__ == "__main__":
    calculate_roi("data/Classified_Data_HSS.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def run_clustering(file_path):
    print(f"Chargement des données depuis : {file_path}")
    df = pd.read_csv(file_path)
    
    # The variable we cluster on (Sales non-normalisé)
    X_cluster = df[['Sales']]
    
    # Test K from 2 to 6
    best_k = 2
    best_score = -1
    scores = []
    k_range = range(2, 7)
    
    print("\nRecherche du K optimal via le Silhouette Score :")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_cluster)
        score = silhouette_score(X_cluster, labels)
        scores.append(score)
        print(f" - K={k} : Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\n=> K optimal sélectionné : {best_k} (Score : {best_score:.4f})")
    
    # Plot Silhouette Scores
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, marker='o', linestyle='--', color='b')
    plt.title('Score de Silhouette en fonction du nombre de clusters (K)')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Score de Silhouette')
    plt.axvline(x=best_k, color='r', linestyle=':', label=f'K optimal = {best_k}')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/silhouette_scores.png")
    plt.close()
    
    # Apply optimal K-Means
    kmeans_opt = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans_opt.fit_predict(X_cluster)
    
    # Re-order labels so that 0 = lowest sales, best_k-1 = highest sales
    centers = kmeans_opt.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    
    # Create mapping dictionary
    mapping = {}
    
    if best_k == 2:
        names = ["Basse Performance", "Haute Performance"]
    elif best_k == 3:
        names = ["Basse Performance", "Moyenne Performance", "Haute Performance"]
    elif best_k == 4:
        names = ["Basse", "Moyenne-Basse", "Moyenne-Haute", "Haute"]
    else:
        names = [f"Niveau {i}" for i in range(1, best_k+1)]
        
    for i, idx in enumerate(sorted_idx):
        mapping[idx] = names[i]
        
    df['Performance'] = df['Cluster'].map(mapping)
    df = df.drop(columns=['Cluster'])
    
    print("\nRépartition des classes générées :")
    print(df['Performance'].value_counts())
    
    # Save the new dataset
    output_file = "data/Classified_Data_HSS.csv"
    df.to_csv(output_file, index=False)
    print(f"\nJeu de données avec labels sauvegardé dans : {output_file}")

if __name__ == "__main__":
    run_clustering("data/Clean_Data_HSS.csv")

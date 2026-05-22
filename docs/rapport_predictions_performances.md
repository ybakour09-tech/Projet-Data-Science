# Rapport de Prédiction et Performances des Modèles

Ce rapport compare 2 modèles de Machine Learning et 2 modèles de Deep Learning sur la prédiction des ventes.

## 1. Métriques de Performance

| Modèle                    |   R2 Score |      RMSE |       MAE |
|:--------------------------|-----------:|----------:|----------:|
| Régression Linéaire (ML)  |   0.999017 | 0.0315594 | 0.0252953 |
| SVR Noyau Linéaire (ML)   |   0.999017 | 0.0315543 | 0.0252823 |
| MLP / Réseau Profond (DL) |   0.993778 | 0.0793824 | 0.0445004 |
| Réseau Linéaire SGD (DL)  |   0.999016 | 0.031571  | 0.0252981 |

## 2. Échantillon des Prédictions (5 premières lignes)

|   Actual_Sales |   TV_Input |   Régression Linéaire (ML) |   SVR Noyau Linéaire (ML) |   MLP / Réseau Profond (DL) |   Réseau Linéaire SGD (DL) |
|---------------:|-----------:|---------------------------:|--------------------------:|----------------------------:|---------------------------:|
|     -1.3827    | -1.41516   |                 -1.41478   |                -1.41258   |                   -1.42064  |                 -1.41479   |
|     -0.0967858 | -0.0718898 |                 -0.0721832 |                -0.0715288 |                   -0.06971  |                 -0.0723792 |
|      0.809732  |  0.810831  |                  0.810097  |                 0.809736  |                    0.804862 |                  0.809779  |
|      0.603652  |  0.657314  |                  0.656657  |                 0.656473  |                    0.651733 |                  0.65636   |
|      1.40807   |  1.38652   |                  1.3855    |                 1.38447   |                    1.3869   |                  1.3851    |

**Note :** Les valeurs sont normalisées (StandardScaler).

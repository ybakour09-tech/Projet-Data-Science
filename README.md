# PRISMA : Système Intelligent Multi-Modèles pour l'Optimisation du ROI Marketing

Bienvenue sur **PRISMA**, un Data Product avancé conçu pour révolutionner la façon dont les décideurs allouent leurs budgets marketing. 
Ce projet utilise une architecture d'Intelligence Artificielle multi-modèles pour non seulement prédire les performances financières, mais surtout pour **expliquer mathématiquement** l'impact et la synergie de chaque canal (TV, Radio, Réseaux Sociaux).

## Guide de Lancement Rapide (Windows)

> [!IMPORTANT]
> **Le secret de la réussite : Être dans le bon dossier !**
> L'erreur la plus courante est de lancer des commandes depuis le dossier parent. Toutes les commandes de ce guide doivent être exécutées depuis la racine du projet (là où se trouve ce fichier `README.md`).

### 1. Ouvrir le bon dossier dans le terminal

Ouvrez un terminal (PowerShell ou Invite de commandes) et naviguez vers le dossier du projet :

```powershell
cd chemin\vers\Projet-Data-Science
```

### 2. Créer l'environnement virtuel (Recommandé)

```powershell
python -m venv venv
.\venv\Scripts\Activate
```
> **Note :** Une fois activé, vous devriez voir `(venv)` apparaître au début de la ligne de votre terminal. Si vous avez une erreur de droits d'exécution de scripts sur PowerShell, tapez `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` puis réessayez.

### 3. Installer les dépendances

Installez toutes les bibliothèques requises (cette opération prendra quelques instants) :

```powershell
pip install -r requirements.txt
```

### 4. Lancer l'application PRISMA

Un script d'automatisation a été préparé pour vous. Toujours depuis le dossier `Projet-Data-Science`, tapez simplement :

```powershell
.\lancer_prisma.bat
```

*(Vous pouvez également double-cliquer sur le fichier `lancer_prisma.bat` depuis l'explorateur de fichiers Windows).*

- Une console noire va s'ouvrir pour faire tourner le backend d'Intelligence Artificielle. **Ne la fermez pas !**
- Le système va charger les modèles en mémoire.
- Votre navigateur internet s'ouvrira automatiquement sur le **Dashboard Prisma**.

---

## Architecture Multi-Modèles

Plutôt que de s'appuyer sur un algorithme unique, PRISMA déploie une cascade de modèles experts :

1. **Identification (Random Forest Classifier)** : 
   Prend en entrée la configuration budgétaire et classifie instantanément la campagne future comme étant à `Haute Performance` ou `Basse Performance`.
2. **Prédiction Exacte (Réseau de Neurones - MLPRegressor)** : 
   Ingère les budgets ainsi que la classification précédente pour prédire avec une haute précision le `ROI (%)` et le Chiffre d'Affaires estimé (Ventes).
3. **Explicabilité Combinatoire (SHAPley Values)** : 
   L'explicateur SHAP "ouvre la boîte noire" du réseau de neurones. Il calcule la contribution exacte (positive ou négative) de la TV, de la Radio et des Réseaux Sociaux dans la formation du ROI, mettant en évidence les phénomènes de cannibalisation ou de synergie.

---

## Structure du Dépôt

Le code source a été organisé selon les standards de Data Science :

```text
/
├── api.py                  # Serveur backend (FastAPI) exposant les modèles IA
├── test_api.py             # Script de test de charge/réponse de l'API
├── lancer_prisma.bat       # Script d'auto-lancement sous Windows
├── dashboard/              # Code source du Frontend web (HTML, CSS, JS)
├── models/                 # Objets de Machine Learning pré-entraînés (joblib)
├── scripts/                # Tous les scripts Python d'exploration et d'entraînement (ML Pipeline)
├── notebooks/              # Cahiers Jupyter d'exploration de données
├── docs/                   # Documentation, rapports de performances et compte-rendus textuels
├── data/                   # Données brutes et normalisées (fichiers CSV)
└── plots/                  # Graphiques d'analyse générés lors de l'entraînement
```

---

## Documentation de l'API (FastAPI)

Le moteur IA est servi par une API REST ultra-rapide tournant sur le port `8000`. 
Vous pouvez accéder à la documentation interactive (Swagger UI) en visitant : `http://127.0.0.1:8000/docs` lorsque l'API tourne.

### Endpoints principaux :
- `POST /predict/performance` : Renvoie la catégorie de performance d'un scénario budgétaire.
- `POST /predict/roi` : Renvoie l'estimation brute du ROI.
- `POST /predict/shap_impact` : Renvoie le ROI complet ainsi que la décomposition mathématique SHAP par canal.

### Exemple de Payload JSON attendu :
```json
{
  "TV": 150.0,
  "Radio": 45.0,
  "Social Media": 30.0
}
```

---

## Interface Utilisateur (Prisma Dashboard)

L'interface a été conçue sans aucun framework lourd (Vanilla HTML/CSS/JS) mais arbore un design premium et futuriste inspiré du *Glassmorphism*. 
- **Temps Réel** : Les curseurs mettent à jour dynamiquement les prédictions.
- **Data Visualization** : Intégration de `Chart.js` pour tracer l'évolution du ROI et comparer visuellement différents scénarios d'investissement.

---

## Technologies Utilisées

- **Backend & IA** : Python 3.10+, FastAPI, Uvicorn, Scikit-Learn, SHAP, Pandas.
- **Frontend** : HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Chart.js.
- **Outils** : Git, Jupyter.

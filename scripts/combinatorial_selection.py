import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

def combinatorial_feature_selection(file_path):
    print(f"Chargement des données depuis : {file_path}")
    df = pd.read_csv(file_path)
    
    features = ['TV', 'Radio', 'Social Media']
    target = 'Sales'
    
    X_full = df[features]
    y = df[target]
    
    # Generate all combinations of features
    all_combinations = []
    for r in range(1, len(features) + 1):
        combinations = list(itertools.combinations(features, r))
        all_combinations.extend(combinations)
        
    print(f"\nÉvaluation de {len(all_combinations)} combinaisons de variables...")
    
    results = []
    
    # Models to evaluate
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    scoring = {'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error'}
    
    for combo in all_combinations:
        combo_list = list(combo)
        X_subset = df[combo_list]
        
        for model_name, model in models.items():
            # 5-fold cross validation
            cv_results = cross_validate(model, X_subset, y, cv=5, scoring=scoring, return_estimator=True)
            
            mean_r2 = cv_results['test_R2'].mean()
            mean_rmse = -cv_results['test_RMSE'].mean() # negate because it's negative RMSE
            
            # If Linear Regression, check for unstable coefficients (sign of multicollinearity)
            unstable_coefs = False
            if model_name == 'Linear Regression' and len(combo_list) > 1:
                # Average coefficients across folds
                coefs = np.mean([est.coef_ for est in cv_results['estimator']], axis=0)
                # If any coefficient is negative, it's a huge red flag because all budgets should positively impact sales
                if any(c < 0 for c in coefs):
                    unstable_coefs = True
                    
            results.append({
                'Modèle': model_name,
                'Variables': " + ".join(combo_list),
                'Nombre de variables': len(combo_list),
                'R2 Moyen (CV)': mean_r2,
                'RMSE Moyen (CV)': mean_rmse,
                'Biaisé par colinéarité ?': 'Oui (Coef négatif)' if unstable_coefs else 'Non'
            })
            
    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RÉSULTATS DE L'ÉTUDE COMBINATOIRE (Validés par Validation Croisée à 5 plis)")
    print("="*80)
    
    # Separate by model
    for model_name in models.keys():
        print(f"\n--- Modèle : {model_name} ---")
        model_res = results_df[results_df['Modèle'] == model_name].sort_values(by='R2 Moyen (CV)', ascending=False)
        # Drop the Modèle column for display
        display_df = model_res.drop(columns=['Modèle'])
        print(display_df.to_string(index=False))
        
    # Feature Importance from Random Forest on ALL features
    print("\n" + "="*80)
    print("IMPORTANCE DES VARIABLES (Random Forest sur toutes les variables)")
    print("="*80)
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_full.fit(X_full, y)
    importances = rf_full.feature_importances_
    
    for feature, imp in zip(features, importances):
        print(f"{feature:15s}: {imp:.4f} ({imp*100:.1f}%)")

if __name__ == "__main__":
    combinatorial_feature_selection("data/Clean_Data_HSS.csv")

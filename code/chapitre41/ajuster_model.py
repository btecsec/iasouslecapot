import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Chargement données manchots (Palmer Penguins)
df = sns.load_dataset('penguins').dropna()
X = pd.get_dummies(df.drop(['species', 'sex'], axis=1)) # Simplification pour l'exercice
y = df['species']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Baseline ---")
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
baseline_score = accuracy_score(y_test, rf_base.predict(X_test))
print(f"Score baseline : {baseline_score:.4f}")

# 1. GridSearchCV
print("\n--- GridSearchCV ---")
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_grid_score = grid_search.best_score_
print(f"Meilleurs réglages : {grid_search.best_params_}")
print(f"Score CV : {best_grid_score:.4f}")
print(f"Gain vs baseline (CV) : {best_grid_score - baseline_score:.4f}")

# 2. RandomizedSearchCV
print("\n--- RandomizedSearchCV ---")
param_dist = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 5, 10, 20, 30, 40]
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    scoring='accuracy'
)
random_search.fit(X_train, y_train)

best_random_score = random_search.best_score_
print(f"Meilleurs réglages (Random) : {random_search.best_params_}")
print(f"Score CV : {best_random_score:.4f}")
print(f"Gain vs baseline (CV) : {best_random_score - baseline_score:.4f}")

# Vérification finale sur le test
print("\n--- Vérification Finale (Test Set) ---")
final_model = grid_search.best_estimator_
final_score = accuracy_score(y_test, final_model.predict(X_test))
print(f"Score final sur test : {final_score:.4f}")

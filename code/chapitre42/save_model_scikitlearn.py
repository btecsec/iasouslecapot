# ==============================================================================
# EXERCICE PRATIQUE : Sauvegarde et chargement Scikit-Learn (Approche Pro)
# Objectif : Utiliser les Pipelines pour packager Scaler + Modèle en un seul objet
# ==============================================================================

import joblib
import zipfile
import os
import shutil
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def main():
    # 1. Préparation des données
    print("--- Étape 1 : Préparation des données ---")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # APPROCHE PROFESSIONNELLE : Le Pipeline
    # On regroupe le preprocessing (scaler) et le modèle dans un seul objet.
    # Cela évite d'oublier le scaler lors du déploiement et garantit
    # que les données de test subissent exactement la même transformation.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Entraînement du pipeline (Scaler + Modèle)...")
    pipeline.fit(X_train, y_train)

    # Prédiction originale
    original_pred = pipeline.predict(X_test)
    print(f"Précision originale : {accuracy_score(y_test, original_pred):.4f}")

    # 2. Sauvegarde
    print("\n--- Étape 2 : Sauvegarde ---")
    # On sauvegarde l'objet pipeline complet
    joblib.dump(pipeline, "ml_pipeline.joblib")
    print("Pipeline sauvegardé sous forme de fichier .joblib")

    # Archivage
    with zipfile.ZipFile("scikit_pro_pack.zip", "w") as zipf:
        zipf.write("ml_pipeline.joblib")
    print("Résultat compressé dans : scikit_pro_pack.zip")

    os.remove("ml_pipeline.joblib")

    # 3. Chargement et Vérification
    print("\n--- Étape 3 : Chargement et Vérification ---")

    with zipfile.ZipFile("scikit_pro_pack.zip", "r") as zipf:
        zipf.extractall("temp_reload_pro")

    # On charge l'objet unique
    reloaded_pipeline = joblib.load("temp_reload_pro/ml_pipeline.joblib")

    # Utilisation directe : pas besoin de scaler manuellement !
    reloaded_pred = reloaded_pipeline.predict(X_test)

    np.testing.assert_array_equal(original_pred, reloaded_pred)
    print("✅ Succès : Le pipeline rechargé produit les mêmes prédictions !")

    shutil.rmtree("temp_reload_pro")

if __name__ == "__main__":
    main()

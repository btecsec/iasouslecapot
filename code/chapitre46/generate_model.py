import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import sys

def generate_model():
    # Import from local prepare_data.py in the same directory
    try:
        from prepare_data import prepare_penguins_data
    except ModuleNotFoundError:
        print("Erreur : impossible de trouver prepare_data.py dans le dossier courant")
        return

    model_path = "modele_manchots.joblib"

    if os.path.exists(model_path):
        print(f"Le modèle {model_path} existe déjà. Rien à faire.")
        return

    print("Génération du modèle en utilisant prepare_penguins_data...")

    # 1. Preparation des données
    X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()

    # 2. L'API ne fait pas de mise à l'échelle (scaling), donc nous devons
    # entraîner le modèle sur des données non normalisées.
    # Nous récupérons les colonnes numériques et nous "annulons" le scaling.
    colonnes_numeriques = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

    # On extrait les données scaled
    X_scaled = X_train[colonnes_numeriques]

    # On inverse la transformation pour retrouver les valeurs originales
    X_unscaled_values = scaler.inverse_transform(X_scaled)

    # On recrée un DataFrame avec les noms attendus par l'API
    mapping_noms = {
        "bill_length_mm": "longueur_bec",
        "bill_depth_mm": "profondeur_bec",
        "flipper_length_mm": "longueur_nageoire",
        "body_mass_g": "masse"
    }

    X_final = pd.DataFrame(X_unscaled_values, columns=colonnes_numeriques)
    X_final = X_final.rename(columns=mapping_noms)

    # On s'assure que y_train correspond aux indices de X_final
    y_final = y_train.values if hasattr(y_train, 'values') else y_train

    # 3. Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_final, y_final)

    # 4. Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé avec succès sous {model_path}")
    print(f"Features utilisées : {list(X_final.columns)}")

if __name__ == "__main__":
    generate_model()

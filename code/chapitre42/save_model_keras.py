# ==============================================================================
# EXERCICE PRATIQUE : Sauvegarde et chargement Keras
# Objectif : Comparer la sauvegarde du modèle complet vs poids seuls
# ==============================================================================

import os
import zipfile
import shutil
import numpy as np
import joblib
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # NOTE : Le chargement de modèles provenant de sources non fiables peut
    # exécuter du code arbitraire. Utilisez uniquement vos propres fichiers.

    # 1. Préparation des données
    print("--- Étape 1 : Préparation des données ---")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Construction et entraînement du modèle
    print("Entraînement du modèle Keras...")
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(20,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

    # Prédiction originale
    original_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # 3. Sauvegarde
    print("\n--- Étape 2 : Sauvegarde ---")
    # On sauvegarde le scaler avec joblib
    joblib.dump(scaler, "scaler_keras.joblib")

    # Option A : Modèle complet (.keras) - Architecture + Poids + Optimiseur
    model.save("full_model.keras")

    # Option B : Poids seuls (.weights.h5) + Architecture (JSON)
    model.save_weights("weights.weights.h5")
    with open("model_config.json", "w") as f:
        json.dump(model.get_config(), f)

    # Comparaison de taille
    size_full = os.path.getsize("full_model.keras") / 1024
    size_weights = os.path.getsize("weights.weights.h5") / 1024
    print(f"Taille Modèle Complet : {size_full:.2f} KB")
    print(f"Taille Poids Seuls : {size_weights:.2f} KB")
    print(f"Le fichier de poids est environ {size_full/size_weights:.1f}x plus petit.")

    # Archivage
    with zipfile.ZipFile("keras_full_pack.zip", "w") as zf:
        zf.write("full_model.keras")
        zf.write("scaler_keras.joblib")

    with zipfile.ZipFile("keras_weights_pack.zip", "w") as zf:
        zf.write("weights.weights.h5")
        zf.write("model_config.json")
        zf.write("scaler_keras.joblib")

    print("Packs ZIP créés : keras_full_pack.zip et keras_weights_pack.zip")

    # Nettoyage temporaire
    os.remove("full_model.keras")
    os.remove("weights.weights.h5")
    os.remove("model_config.json")
    os.remove("scaler_keras.joblib")

    # 4. Vérification
    print("\n--- Étape 3 : Vérification ---")

    # Cas A : Chargement modèle complet
    with zipfile.ZipFile("keras_full_pack.zip", "r") as zf:
        zf.extractall("temp_keras_full")

    reloaded_scaler_a = joblib.load("temp_keras_full/scaler_keras.joblib")
    reloaded_model_a = keras.models.load_model("temp_keras_full/full_model.keras")

    pred_a = (reloaded_model_a.predict(reloaded_scaler_a.transform(X_test)) > 0.5).astype(int)

    # Cas B : Chargement poids seuls (Architecture JSON + Poids)
    with zipfile.ZipFile("keras_weights_pack.zip", "r") as zf:
        zf.extractall("temp_keras_weights")

    reloaded_scaler_b = joblib.load("temp_keras_weights/scaler_keras.joblib")

    # On recharge l'architecture depuis le JSON
    with open("temp_keras_weights/model_config.json", "r") as f:
        config = json.load(f)
    model_b = keras.Sequential.from_config(config)

    model_b.load_weights("temp_keras_weights/weights.weights.h5")

    pred_b = (model_b.predict(reloaded_scaler_b.transform(X_test)) > 0.5).astype(int)

    # Vérification finale
    np.testing.assert_array_equal(original_pred, pred_a)
    np.testing.assert_array_equal(original_pred, pred_b)
    print("✅ Succès : Les prédictions (Full et Weights) sont identiques à l'original !")

    # Nettoyage final
    shutil.rmtree("temp_keras_full")
    shutil.rmtree("temp_keras_weights")

if __name__ == "__main__":
    main()

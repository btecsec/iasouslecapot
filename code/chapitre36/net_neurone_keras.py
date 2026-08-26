import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# prepare_penguins_data vit dans le dossier du chapitre 34, celui où l'on
# choisit un premier modèle. On ajoute ce dossier au chemin plutôt que d'en
# recopier le contenu ici.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chapitre34")))
from prepare_data import prepare_penguins_data

def main():
    # 1. Chargement des données
    # On utilise notre fonction du chapitre 34 pour récupérer des données propres et normalisées.
    print("Chargement des données...")
    X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()

    # 2. Encodage des labels
    # Keras a besoin de nombres pour la perte 'sparse_categorical_crossentropy'.
    # On transforme "Adelie", "Chinstrap", "Gentoo" en 0, 1, 2.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    print(f"Labels encodés : {encoder.classes_}")

    # 3. Construction du réseau
    # On empile les couches avec le modèle Sequential.
    print("\nConstruction du réseau de neurones...")
    modele = keras.Sequential([
        # Première couche : 16 neurones, activation ReLU.
        # input_shape est crucial : il dit au réseau combien de caractéristiques on a en entrée.
        layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),

        # Deuxième couche cachée : 8 neurones pour condenser l'information.
        layers.Dense(8, activation="relu"),

        # Couche de sortie : 3 neurones (un par espèce) avec softmax pour obtenir des probabilités.
        layers.Dense(3, activation="softmax")
    ])

    # 4. Compilation : définir les règles du jeu
    # On choisit Adam pour l'optimiseur et la cross-entropie pour la classification multiclasse.
    modele.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Entraînement du réseau
    # On lance l'apprentissage sur 50 époques avec un petit lot (batch) de 16 exemples.
    # On réserve 20 % des données d'entraînement pour surveiller la généralisation (validation).
    print("\nEntraînement en cours... Patientez un instant.")
    historique = modele.fit(
        X_train, y_train_encoded,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # 6. Lecture des courbes d'entraînement
    # On trace l'exactitude pour voir si le modèle apprend bien ou s'il surapprend (overfitting).
    print("\nGénération des courbes d'apprentissage...")
    plt.figure(figsize=(10, 6))
    plt.plot(historique.history["accuracy"], label="Entraînement")
    plt.plot(historique.history["val_accuracy"], label="Validation")
    plt.title("Évolution de l'exactitude du réseau")
    plt.xlabel("Époque")
    plt.ylabel("Exactitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 7. Évaluation sur le jeu de test
    # Le juge final : on teste sur des données que le réseau n'a JAMAIS vues.
    perte, exactitude = modele.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"\n--- Résultat Final ---")
    print(f"Exactitude sur le jeu de test : {exactitude:.2%}")
    print("Bravo ! Votre premier réseau de neurones a terminé son entraînement.")

if __name__ == "__main__":
    main()

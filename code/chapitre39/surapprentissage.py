import os
import sys
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Ajout du chemin pour importer prepare_penguins_data (depuis le chapitre 34)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chapitre34")))
from prepare_data import prepare_penguins_data

def run_experiment(name, model_fn, X_train, y_train, X_val, y_val, ax_loss, ax_acc, epochs=200):
    """Entraîne un modèle et affiche Loss et Accuracy sur les axes fournis."""
    print(f"\n--- Expérience : {name} ---")

    model = model_fn(X_train.shape[1])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # 1. Courbe de Perte (L'indicateur le plus parlant pour le surapprentissage)
    ax_loss.plot(history.history["loss"], label="Train Loss", color="blue")
    ax_loss.plot(history.history["val_loss"], label="Val Loss", color="red")
    ax_loss.set_title(f"{name} : Perte (Loss)")
    ax_loss.set_xlabel("Époque")
    ax_loss.set_ylabel("Perte")
    ax_loss.legend()
    ax_loss.grid(True)

    # 2. Courbe d'Exactitude
    ax_acc.plot(history.history["accuracy"], label="Train Acc", color="blue")
    ax_acc.plot(history.history["val_accuracy"], label="Val Acc", color="red")
    ax_acc.set_title(f"{name} : Exactitude")
    ax_acc.set_xlabel("Époque")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    return history

def main():
    # 1. Préparation des données
    X_train, X_test, y_train, y_test, _ = prepare_penguins_data()

    # Encodage manuel simple pour les labels (0, 1, 2)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # --- ASTUCE POUR LE SURAPPRENTISSAGE ---
    # On augmente légèrement l'échantillon pour que la courbe de Dropout soit plus parlante.
    X_tiny = X_train[:100]
    y_tiny = y_train_enc[:100]
    X_val = X_train[100:300]
    y_val = y_train_enc[100:300]
    print(f"Entraînement sur : {len(X_tiny)} ex. Validation sur : {len(X_val)} ex.")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # CAS 1 : Réseau trop gros (Surapprentissage massif)
    def huge_model(dim):
        return keras.Sequential([
            layers.Dense(512, activation="relu", input_shape=(dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(3, activation="softmax")
        ])
    run_experiment("Trop Gros", huge_model, X_tiny, y_tiny, X_val, y_val, axes[0, 0], axes[0, 1])

    # CAS 2 : Réseau gros + Dropout (Régularisation)
    def huge_dropout_model(dim):
        return keras.Sequential([
            layers.Dense(512, activation="relu", input_shape=(dim,)),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(3, activation="softmax")
        ])
    run_experiment("Gros + Dropout", huge_dropout_model, X_tiny, y_tiny, X_val, y_val, axes[1, 0], axes[1, 1])

    # CAS 3 : Réseau très petit (Sous-apprentissage potentiel)
    def small_model(dim):
        return keras.Sequential([
            layers.Dense(4, activation="relu", input_shape=(dim,)),
            layers.Dense(3, activation="softmax")
        ])
    run_experiment("Trop Petit", small_model, X_tiny, y_tiny, X_val, y_val, axes[2, 0], axes[2, 1])

    plt.tight_layout()
    plt.show()

    print("\n--- ANALYSE ---")
    print("1. Trop Gros : L'écart énorme entre Train (100%) et Validation (faible) indique un SURAPPRENTISSAGE.")
    print("2. Gros + Dropout : L'écart se réduit. Le modèle ne peut plus mémoriser bêtement, il généralise mieux.")
    print("3. Trop Petit : Les deux courbes restent basses. Le modèle n'a pas assez de capacité : SOUS-APPRENTISSAGE.")

    print("\n--- ANALOGIE DE L'ÉTUDIANT ---")
    print("- Surapprentissage : L'élève qui apprend le manuel par cœur mais sèche dès que la question change.")
    print("- Sous-apprentissage : L'élève qui n'a pas assez étudié et échoue même aux questions simples.")
    print("- Équilibré (Dropout) : L'élève qui comprend les concepts et sait les appliquer à de nouveaux problèmes.")

if __name__ == "__main__":
    main()

"""Chapitre 36 — corrigé de l'exercice : un réseau Keras sur les manchots.

Usage :
    python exercice_reseau_keras.py
    python exercice_reseau_keras.py --png    # écrit les courbes au lieu de les afficher
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from tensorflow import keras  # noqa: E402

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots  # noqa: E402

GRAINE = 42


# --------------------------------------------------------------- question 1
def preparer(graine: int = GRAINE):
    """Encode le label en 0/1/2 et normalise les mesures.

    Deux transformations indispensables ici :
      * `LabelEncoder` parce que Keras ne sait pas lire « Adelie » ;
      * `StandardScaler` parce qu'un réseau apprend très mal sur des
        variables d'échelles très différentes (chapitre 32).
    """
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X = df[COLONNES_NUMERIQUES].to_numpy(dtype="float32")
    encodeur = LabelEncoder()
    y = encodeur.fit_transform(df[CIBLE])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=graine, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    return X_train, X_test, y_train, y_test, encodeur, scaler


# --------------------------------------------------------------- question 2
def construire(n_entrees: int = 4, n_classes: int = 3, graine: int = GRAINE):
    """Deux couches cachées `relu`, une sortie `softmax`.

    La sortie a exactement un neurone par classe : c'est la règle.
    """
    keras.utils.set_random_seed(graine)
    return keras.Sequential(
        [
            keras.layers.Input(shape=(n_entrees,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


# --------------------------------------------------------------- question 3
def compiler(modele):
    """`sparse_categorical_crossentropy` car les labels sont 0/1/2.

    Avec des labels one-hot, ce serait `categorical_crossentropy`. Se tromper
    ici produit une erreur de forme incompréhensible : autant le savoir.
    """
    modele.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modele


# --------------------------------------------------------------- question 4
def entrainer(modele, X_train, y_train, epochs: int = 50):
    """50 époques, avec 20 % du train mis de côté pour la validation."""
    return modele.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
    )


# --------------------------------------------------------------- question 5
def courbes(historique):
    """Les deux courbes d'accuracy. Renvoie la figure, sans l'afficher."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.plot(historique.history["accuracy"], label="entraînement")
    axes.plot(historique.history["val_accuracy"], label="validation")
    axes.set_title("Accuracy au fil des époques")
    axes.set_xlabel("Époque")
    axes.set_ylabel("Accuracy")
    axes.legend()
    return figure


def ecart_final(historique) -> float:
    """L'écart entraînement - validation sur la dernière époque.

    C'est la mesure du surapprentissage : un écart qui se creuse au fil des
    époques signifie que le réseau mémorise (chapitre 39).
    """
    return float(
        historique.history["accuracy"][-1] - historique.history["val_accuracy"][-1]
    )


# --------------------------------------------------------------- question 6
def evaluer(modele, X_test, y_test) -> dict[str, float]:
    """Le score final, sur le jeu jamais vu."""
    perte, accuracy = modele.evaluate(X_test, y_test, verbose=0)
    return {"perte": float(perte), "accuracy": float(accuracy)}


def main() -> None:
    X_train, X_test, y_train, y_test, encodeur, _ = preparer()
    print(f"1. classes encodees : {dict(enumerate(encodeur.classes_))}")

    modele = compiler(construire())
    print(f"2-3. {modele.count_params()} parametres")

    historique = entrainer(modele, X_train, y_train)
    print(f"4. {len(historique.history['loss'])} epoques effectuees")

    print(f"5. ecart entrainement/validation : {ecart_final(historique):+.4f}")
    print(
        "   un ecart proche de 0 = pas de surapprentissage ;\n"
        "   un ecart qui se creuse = le reseau memorise (chapitre 39)"
    )

    resultat = evaluer(modele, X_test, y_test)
    print(f"6. test : accuracy {resultat['accuracy']:.4f}, perte {resultat['perte']:.4f}")

    figure = courbes(historique)
    if "--png" in sys.argv:
        figure.savefig("courbes_accuracy.png", dpi=120, bbox_inches="tight")
        print("\necrit : courbes_accuracy.png")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()

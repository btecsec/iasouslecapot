"""Chapitre 39 — corrigé de l'exercice : provoquer, puis soigner le surapprentissage.

Recette pour surapprendre à coup sûr :
  * peu de données d'entraînement (40 exemples) ;
  * quelques étiquettes fausses (10 sur 40) — tout dataset réel en contient ;
  * un gros réseau et beaucoup d'époques.

Le modèle finit par apprendre les erreurs par cœur. On mesure ensuite l'effet
du dropout et de la réduction du nombre de neurones.

Usage :
    python exercice_surapprentissage.py
    python exercice_surapprentissage.py --png
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


def preparer(n_exemples: int = 40, etiquettes_fausses: int = 10, graine: int = GRAINE):
    """40 exemples d'entraînement, dont 10 mal étiquetés.

    La validation, elle, garde 100 exemples **corrects** : c'est ce qui rend
    la courbe lisible. Avec les 12 exemples d'un `validation_split=0.3` sur
    40, elle sauterait par bonds de 8 points et ne dirait rien.
    """
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X = df[COLONNES_NUMERIQUES].to_numpy(dtype="float32")
    y = LabelEncoder().fit_transform(df[CIBLE])

    X_train, X_reste, y_train, y_reste = train_test_split(
        X, y, train_size=n_exemples, random_state=graine, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_reste, y_reste, train_size=100, random_state=graine, stratify=y_reste
    )

    if etiquettes_fausses:
        generateur = np.random.default_rng(graine)
        y_train = y_train.copy()
        touchees = generateur.choice(len(y_train), etiquettes_fausses, replace=False)
        y_train[touchees] = (y_train[touchees] + 1) % 3

    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train).astype("float32"),
        scaler.transform(X_val).astype("float32"),
        scaler.transform(X_test).astype("float32"),
        y_train,
        y_val,
        y_test,
    )


def construire(neurones: int = 256, dropout: float = 0.0, graine: int = GRAINE):
    """Un réseau dont on choisit la taille et le taux de dropout."""
    keras.utils.set_random_seed(graine)
    couches = [keras.layers.Input(shape=(4,))]
    for _ in range(2):
        couches.append(keras.layers.Dense(neurones, activation="relu"))
        if dropout > 0:
            couches.append(keras.layers.Dropout(dropout))
    couches.append(keras.layers.Dense(3, activation="softmax"))

    modele = keras.Sequential(couches)
    modele.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return modele


def entrainer(modele, X_train, y_train, X_val, y_val, epochs: int = 200):
    return modele.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=0,
        batch_size=8,
    )


def ecart(historique) -> float:
    """L'écart entraînement - validation en fin de course.

    C'est LA mesure du surapprentissage : le modèle réussit chez lui et
    échoue ailleurs.
    """
    return float(
        historique.history["accuracy"][-1] - historique.history["val_accuracy"][-1]
    )


def meilleure_validation(historique) -> tuple[int, float]:
    """(époque, valeur) du sommet de la courbe de validation.

    Si ce sommet est loin de la fin, un arrêt anticipé aurait sauvé le modèle.
    """
    val = historique.history["val_accuracy"]
    meilleure = int(np.argmax(val))
    return meilleure, float(val[meilleure])


# --------------------------------------------------------------- question 2
def diagnostiquer(historique, seuil_ecart: float = 0.15, seuil_bas: float = 0.70) -> str:
    """Sous-apprentissage, surapprentissage, ou équilibre."""
    accuracy = historique.history["accuracy"][-1]
    if accuracy < seuil_bas:
        return "sous-apprentissage"  # même chez lui, le modèle échoue
    if ecart(historique) > seuil_ecart:
        return "surapprentissage"  # excellent chez lui, mauvais ailleurs
    return "equilibre"


def experience(neurones, dropout, X_train, y_train, X_val, y_val, epochs: int = 200):
    """Un essai complet, résumé en quelques nombres."""
    modele = construire(neurones=neurones, dropout=dropout)
    historique = entrainer(modele, X_train, y_train, X_val, y_val, epochs=epochs)
    epoque, sommet = meilleure_validation(historique)
    return {
        "modele": modele,
        "historique": historique,
        "parametres": modele.count_params(),
        "accuracy_train": float(historique.history["accuracy"][-1]),
        "accuracy_val": float(historique.history["val_accuracy"][-1]),
        "ecart": ecart(historique),
        "meilleure_val": sommet,
        "epoque_du_sommet": epoque,
        "diagnostic": diagnostiquer(historique),
    }


def courbes(historiques: dict):
    """Les courbes de validation superposées, un tracé par expérience."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    for nom, historique in historiques.items():
        axes.plot(historique.history["val_accuracy"], label=nom)
    axes.set_title("Accuracy de validation selon la régularisation")
    axes.set_xlabel("Époque")
    axes.set_ylabel("Accuracy (validation)")
    axes.legend(fontsize=8)
    return figure


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = preparer()
    print(f"{len(X_train)} exemples d'entrainement, dont 10 mal etiquetes")
    print(f"{len(X_val)} exemples de validation, corrects\n")

    essais = {
        "1. gros reseau (256)": (256, 0.0),
        "3. gros + Dropout(0.3)": (256, 0.3),
        "3bis. gros + Dropout(0.5)": (256, 0.5),
        "4. petit reseau (8)": (8, 0.0),
    }

    resultats = {}
    for nom, (neurones, dropout) in essais.items():
        r = experience(neurones, dropout, X_train, y_train, X_val, y_val)
        resultats[nom] = r
        print(
            f"{nom:<26} {r['parametres']:>6} params | "
            f"train {r['accuracy_train']:.3f} | val {r['accuracy_val']:.3f} | "
            f"ecart {r['ecart']:+.3f} | {r['diagnostic']}"
        )
        print(
            f"{'':<26} sommet de validation {r['meilleure_val']:.3f} "
            f"a l'epoque {r['epoque_du_sommet']}"
        )

    print("\n5. L'etudiant qui correspond a chaque cas :")
    print("   surapprentissage   -> apprend les annales par coeur, y compris")
    print("                         les erreurs du corrige ; seche sur un sujet inedit")
    print("   sous-apprentissage -> n'a pas ouvert le cours ; echoue partout")
    print("   equilibre          -> a compris la methode ; reussit sur du nouveau")

    figure = courbes({nom: r["historique"] for nom, r in resultats.items()})
    if "--png" in sys.argv:
        figure.savefig("courbes_surapprentissage.png", dpi=120, bbox_inches="tight")
        print("\necrit : courbes_surapprentissage.png")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()

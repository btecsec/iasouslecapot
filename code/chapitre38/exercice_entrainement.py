"""Chapitre 38 — corrigé de l'exercice : faire varier les réglages.

On utilise le `MLPClassifier` de Scikit-learn : c'est un vrai réseau de
neurones, mais il expose directement les trois boutons du chapitre —
`learning_rate_init`, `batch_size`, `max_iter` — et il conserve sa courbe de
perte dans `loss_curve_`. Idéal pour expérimenter en quelques secondes.

Usage :
    python exercice_entrainement.py
    python exercice_entrainement.py --png
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42


def preparer(graine: int = GRAINE):
    """Les manchots, normalisés et découpés."""
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X = df[COLONNES_NUMERIQUES].to_numpy(dtype="float64")
    y = LabelEncoder().fit_transform(df[CIBLE])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=graine, stratify=y
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


def entrainer(X_train, y_train, lr=0.001, batch_size=32, epochs=200, graine=GRAINE):
    """Un entraînement, avec les trois réglages de l'exercice."""
    modele = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        batch_size=batch_size,
        max_iter=epochs,
        random_state=graine,
        n_iter_no_change=epochs,  # on veut toutes les époques, pas d'arrêt anticipé
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modele.fit(X_train, y_train)
    return modele


# --------------------------------------------------------------- question 1
def perte_du_hasard(n_classes: int = 3) -> float:
    """La perte d'un modèle qui n'a rien appris : -log(1 / n_classes).

    Repère indispensable pour juger une courbe : ici log(3) ≈ 1,0986. Une
    perte de départ très supérieure signifie que les poids ont explosé dès les
    premiers pas.
    """
    return float(np.log(n_classes))


def profil_courbe(pertes: list[float], n_classes: int = 3) -> str:
    """Nomme ce qu'on voit sur la courbe de perte.

    C'est la question 5 de l'exercice : relier un symptôme à un diagnostic.
    """
    pertes = np.asarray(list(pertes), dtype=float)
    reference = perte_du_hasard(n_classes)

    if not np.all(np.isfinite(pertes)) or pertes[-1] >= pertes[0]:
        return "divergence"  # la perte ne baisse pas : taux beaucoup trop grand
    if pertes.max() > 3 * reference:
        # Le modèle a d'abord explosé, puis s'est rattrapé tant bien que mal.
        return "explosion puis recuperation"
    if pertes[-1] > 0.5 * pertes[0]:
        return "stagnation"  # taux trop petit, ou modèle trop simple
    seconde_moitie = np.diff(pertes)[len(pertes) // 2 :]
    if (seconde_moitie > 0).mean() > 0.40:
        # Normal avec des mini-batchs : chaque lot tire la perte dans son sens.
        return "descente bruitee"
    return "descente saine"


def comparer_taux(X_train, y_train, taux=(0.001, 0.01, 1.0)) -> dict[float, dict]:
    """Question 1 : que devient la courbe selon le taux d'apprentissage ?"""
    resultats = {}
    for lr in taux:
        modele = entrainer(X_train, y_train, lr=lr, epochs=150)
        pertes = modele.loss_curve_
        resultats[lr] = {
            "perte_finale": float(pertes[-1]),
            "profil": profil_courbe(pertes),
            "courbe": [float(p) for p in pertes],
        }
    return resultats


# --------------------------------------------------------------- question 2
def comparer_batch(X_train, y_train, tailles=(8, 32, 64)) -> dict[int, dict]:
    """Question 2 : un petit batch est-il plus stable, ou plus rapide ?"""
    resultats = {}
    for taille in tailles:
        modele = entrainer(X_train, y_train, lr=0.01, batch_size=taille, epochs=100)
        pertes = np.array(modele.loss_curve_)
        resultats[taille] = {
            "perte_finale": float(pertes[-1]),
            # l'écart-type des variations mesure les à-coups de la descente
            "irregularite": float(np.std(np.diff(pertes))),
        }
    return resultats


# --------------------------------------------------------------- question 3
def effet_epoques(X_train, y_train, courtes=50, longues=100) -> dict:
    """Question 3 : doubler les époques fait-il encore baisser la perte ?"""
    court = entrainer(X_train, y_train, lr=0.01, epochs=courtes)
    long = entrainer(X_train, y_train, lr=0.01, epochs=longues)
    return {
        "perte_courte": float(court.loss_curve_[-1]),
        "perte_longue": float(long.loss_curve_[-1]),
        "gain": float(court.loss_curve_[-1] - long.loss_curve_[-1]),
    }


def courbes(resultats: dict):
    """Les courbes de perte superposées, une par taux d'apprentissage."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    for lr, detail in resultats.items():
        axes.plot(detail["courbe"], label=f"lr = {lr} ({detail['profil']})")
    axes.set_title("La perte selon le taux d'apprentissage")
    axes.set_xlabel("Époque")
    axes.set_ylabel("Perte")
    axes.set_yscale("log")
    axes.legend()
    return figure


def main() -> None:
    X_train, X_test, y_train, y_test = preparer()

    print("1. taux d'apprentissage :")
    taux = comparer_taux(X_train, y_train)
    for lr, detail in taux.items():
        print(f"   lr={lr:<7} perte finale {detail['perte_finale']:>10.4f}  {detail['profil']}")

    print("\n2. taille de batch (lr=0.01) :")
    for taille, detail in comparer_batch(X_train, y_train).items():
        print(
            f"   batch={taille:<4} perte finale {detail['perte_finale']:.4f}"
            f"   irregularite {detail['irregularite']:.4f}"
        )

    print("\n3. nombre d'epoques :")
    epoques = effet_epoques(X_train, y_train)
    print(f"   50 epoques  : {epoques['perte_courte']:.4f}")
    print(f"   100 epoques : {epoques['perte_longue']:.4f}")
    print(f"   gain        : {epoques['gain']:.4f}")

    print("\n4. Pour classer, la perte adaptee est l'entropie croisee")
    print("   (log-loss) : c'est celle qu'utilise ce modele.")

    figure = courbes(taux)
    if "--png" in sys.argv:
        figure.savefig("courbes_taux.png", dpi=120, bbox_inches="tight")
        print("\necrit : courbes_taux.png")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()

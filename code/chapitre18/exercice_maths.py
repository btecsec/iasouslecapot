"""Chapitre 18 — corrigé de l'exercice : une couche de réseau, à la main.

Chaque fonction est écrite deux fois dans votre tête : la formule mathématique
du chapitre, et sa traduction NumPy. Le fichier de tests vérifie qu'elles
donnent bien les chiffres annoncés dans le README.

Usage :
    python exercice_maths.py
"""

from __future__ import annotations

import numpy as np

# Les poids et biais du chapitre, inchangés.
W = np.array(
    [
        [0.1, 0.4, -0.2],  # ligne 1 : la taille, vue par les 3 neurones
        [0.3, -0.5, 0.6],  # ligne 2 : le poids, vu par les 3 neurones
    ]
)
B = np.array([0.1, 0.0, 0.2])

# Les plages observées dans le jeu de données.
TAILLE_MIN, TAILLE_MAX = 1.50, 2.00
POIDS_MIN, POIDS_MAX = 50.0, 110.0


# ------------------------------------------------------------ normalisation
def normaliser(valeur: float, mini: float, maxi: float) -> float:
    """Min-max : ramène une valeur dans [0, 1] selon sa plage observée."""
    return (valeur - mini) / (maxi - mini)


def vecteur_entree(taille_m: float, poids_kg: float) -> np.ndarray:
    """La personne, transformée en position relative sur ses deux échelles."""
    return np.array(
        [
            normaliser(taille_m, TAILLE_MIN, TAILLE_MAX),
            normaliser(poids_kg, POIDS_MIN, POIDS_MAX),
        ]
    )


# --------------------------------------------------------- algèbre linéaire
def produit_scalaire(a: np.ndarray, b: np.ndarray) -> float:
    """Somme des produits terme à terme : à quel point a et b vont ensemble."""
    return float(np.dot(a, b))


def norme(vecteur: np.ndarray) -> float:
    """La longueur du vecteur : racine de la somme des carrés."""
    return float(np.sqrt(np.sum(vecteur**2)))


def couche(x: np.ndarray, poids: np.ndarray = W, biais: np.ndarray = B) -> np.ndarray:
    """Y = X x W + b — les scores bruts, un par neurone."""
    return x @ poids + biais


def couche_a_la_main(x: np.ndarray) -> list[float]:
    """La même chose, neurone par neurone, comme au tableau.

    Utile pour se convaincre que `x @ W` n'est rien d'autre que trois produits
    scalaires empilés.
    """
    return [float(produit_scalaire(x, W[:, j]) + B[j]) for j in range(W.shape[1])]


# ---------------------------------------------------------- les activations
def sigmoide(y: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-y) : chaque neurone répond dans son coin, entre 0 et 1."""
    return 1 / (1 + np.exp(-y))


def softmax(y: np.ndarray) -> np.ndarray:
    """Des scores bruts vers une distribution de probabilités (somme = 1).

    Le `- y.max()` ne change pas le résultat mathématique mais évite un
    débordement quand les scores sont grands : c'est l'astuce standard.
    """
    exponentielles = np.exp(y - np.max(y))
    return exponentielles / exponentielles.sum()


# ---------------------------------------------------------------- les pertes
def erreur_quadratique(attendu: float, obtenu: float) -> float:
    """La contribution d'un seul neurone à la somme des carrés."""
    return (attendu - obtenu) ** 2


def mse(attendus: np.ndarray, obtenus: np.ndarray) -> float:
    """Erreur quadratique moyenne : la moyenne des carrés des écarts."""
    return float(np.mean((np.asarray(attendus) - np.asarray(obtenus)) ** 2))


def cross_entropy(probabilite_bonne_classe: float) -> float:
    """-log(p) : la perte des tâches de classification.

    p = 1.0  -> perte 0     (le modèle était sûr et il avait raison)
    p = 0.10 -> perte 2.30  (question 6 du quiz)
    """
    return float(-np.log(probabilite_bonne_classe))


def main() -> None:
    x = vecteur_entree(1.60, 62)
    print(f"1. X            = {x}")
    print(f"2. ||X||        = {norme(x):.4f}")

    y = couche(x)
    print(f"3. Y bruts      = {np.round(y, 4)}")
    print(f"   a la main    = {[round(v, 4) for v in couche_a_la_main(x)]}")

    s = sigmoide(y)
    print(f"4. sigmoide     = {np.round(s, 4)}  (somme = {s.sum():.4f})")

    p = softmax(y)
    print(f"5. softmax      = {np.round(p, 4)}  (somme = {p.sum():.4f})")
    print("   -> la softmax met les neurones en concurrence, pas la sigmoide")

    contribution = erreur_quadratique(1.0, s[0])
    print(f"6. ecart n1     = {1.0 - s[0]:.4f}")
    print(f"   au carre     = {contribution:.4f}  (soit {contribution / 3:.4f} en moyenne sur 3)")

    print(f"\n   cross-entropy si p=0.10 : {cross_entropy(0.10):.2f}")


if __name__ == "__main__":
    main()

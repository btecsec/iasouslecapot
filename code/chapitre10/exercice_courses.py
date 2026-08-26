"""Chapitre 10 — corrigé de l'exercice : un petit panier de courses.

Chaque question de l'énoncé devient une fonction, ce qui permet de la tester
(voir test_exercice_courses.py) au lieu de simplement la regarder s'afficher.

Usage :
    python exercice_courses.py
"""

from __future__ import annotations

import pandas as pd

PRODUITS = ["pain", "lait", "œufs"]
PRIX = [1.2, 0.9, 2.5]


# --------------------------------------------------------------- question 1
def creer_panier() -> pd.DataFrame:
    """Le DataFrame de l'énoncé : une clé du dictionnaire = une colonne."""
    return pd.DataFrame({"produit": PRODUITS, "prix": PRIX})


# --------------------------------------------------------------- question 2
def colonne_prix(df: pd.DataFrame) -> pd.Series:
    """Une seule paire de crochets renvoie une Series (une colonne)."""
    return df["prix"]


# --------------------------------------------------------------- question 3
def prix_moyen(df: pd.DataFrame) -> float:
    """La moyenne est déléguée à NumPy : aucune boucle Python ici."""
    return float(df["prix"].mean())


# --------------------------------------------------------------- question 4
def produits_chers(df: pd.DataFrame, seuil: float = 1.0) -> pd.DataFrame:
    """Filtrage booléen : df["prix"] > seuil est un masque, pas un booléen."""
    return df[df["prix"] > seuil]


def produits_entre(df: pd.DataFrame, mini: float, maxi: float) -> pd.DataFrame:
    """Deux conditions : & et non `and`, et des parenthèses obligatoires."""
    return df[(df["prix"] > mini) & (df["prix"] < maxi)]


# ------------------------------------------------------------------- bonus
def ajouter_tva(df: pd.DataFrame, taux: float = 0.055) -> pd.DataFrame:
    """Ajoute une colonne calculée, sans modifier le DataFrame d'origine."""
    resultat = df.copy()
    resultat["prix_ttc"] = (resultat["prix"] * (1 + taux)).round(2)
    return resultat


def main() -> None:
    df = creer_panier()

    print("--- le panier ---")
    print(df)

    print("\n--- la colonne prix ---")
    print(colonne_prix(df))

    print(f"\nPrix moyen : {prix_moyen(df):.2f} €")

    print("\n--- les produits à plus de 1 € ---")
    print(produits_chers(df))

    print("\n--- entre 1 € et 2 € ---")
    print(produits_entre(df, 1.0, 2.0))

    print("\n--- avec la TVA à 5,5 % ---")
    print(ajouter_tva(df))

    print("\n--- le résumé statistique ---")
    print(df.describe())


if __name__ == "__main__":
    main()

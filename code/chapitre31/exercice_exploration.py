"""Chapitre 31 — corrigé de l'exercice : explorer le dataset des manchots.

Chaque question devient une fonction qui **renvoie** son résultat au lieu de
l'imprimer : c'est ce qui permet de la tester, et de la réutiliser au
chapitre 32 pour le nettoyage.

Usage :
    python exercice_exploration.py
    python exercice_exploration.py --png   # écrit l'histogramme au lieu de l'afficher
"""

from __future__ import annotations

import sys

import pandas as pd

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots


# --------------------------------------------------------------- question 1
def dimensions(df: pd.DataFrame) -> tuple[int, int]:
    """Combien de lignes, combien de colonnes."""
    return df.shape


# --------------------------------------------------------------- question 2
def valeurs_manquantes(df: pd.DataFrame) -> dict[str, int]:
    """Les colonnes à trous, et le nombre de trous — les autres sont écartées."""
    compte = df.isnull().sum()
    return {colonne: int(n) for colonne, n in compte.items() if n > 0}


# --------------------------------------------------------------- question 3
def especes(df: pd.DataFrame) -> dict[str, int]:
    """L'effectif de chaque espèce."""
    return df[CIBLE].value_counts().to_dict()


def est_equilibre(df: pd.DataFrame, tolerance: float = 2.0) -> bool:
    """Vrai si la classe la plus fréquente ne dépasse pas `tolerance` fois la plus rare.

    Un déséquilibre modéré (ici environ 2,2) ne casse rien, mais il interdit de
    juger un modèle sur la seule accuracy (chapitre 40).
    """
    effectifs = especes(df)
    return max(effectifs.values()) <= tolerance * min(effectifs.values())


# --------------------------------------------------------------- question 4
def distribution_masse(df: pd.DataFrame, bins: int = 20):
    """L'histogramme des masses. Renvoie la figure, sans l'afficher.

    L'import est local pour que les fonctions d'analyse ci-dessus restent
    utilisables sans matplotlib installé.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.hist(df["body_mass_g"].dropna(), bins=bins, edgecolor="white")
    axes.set_title("Répartition de la masse des manchots")
    axes.set_xlabel("Masse (g)")
    axes.set_ylabel("Effectif")
    return figure


# --------------------------------------------------------------- question 5
def problemes_a_corriger(df: pd.DataFrame) -> list[str]:
    """Le diagnostic : ce qu'il faudra régler avant d'entraîner (chapitre 32)."""
    problemes = []

    manquantes = valeurs_manquantes(df)
    if manquantes:
        detail = ", ".join(f"{col} ({n})" for col, n in manquantes.items())
        problemes.append(f"valeurs manquantes : {detail}")

    doublons = int(df.duplicated().sum())
    if doublons:
        problemes.append(f"{doublons} lignes en double")

    # `dtype == object` ne suffit plus : pandas 3 a un vrai type texte.
    categorielles = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if categorielles:
        problemes.append(
            "colonnes textuelles à encoder : " + ", ".join(categorielles)
        )

    etendues = {c: df[c].max() - df[c].min() for c in COLONNES_NUMERIQUES if c in df}
    if etendues and max(etendues.values()) > 100 * min(etendues.values()):
        problemes.append(
            "échelles très différentes (la masse en grammes écrase le bec en mm) "
            "-> normalisation nécessaire"
        )

    return problemes


def main() -> None:
    df = charger_manchots()

    lignes, colonnes = dimensions(df)
    print(f"1. {lignes} lignes, {colonnes} colonnes")

    print("2. valeurs manquantes :")
    for colonne, n in valeurs_manquantes(df).items():
        print(f"     {colonne:<20} {n}")

    print("3. especes :")
    for espece, n in especes(df).items():
        print(f"     {espece:<12} {n}")
    print(f"   equilibre ? {'oui' if est_equilibre(df) else 'non'}")

    print("5. problemes a corriger avant l'entrainement :")
    for probleme in problemes_a_corriger(df):
        print(f"     - {probleme}")

    figure = distribution_masse(df)
    if "--png" in sys.argv:
        figure.savefig("distribution_masse.png", dpi=120, bbox_inches="tight")
        print("\necrit : distribution_masse.png")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()

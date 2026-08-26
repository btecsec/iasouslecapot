"""Chapitre 32 — corrigé de l'exercice : nettoyer les manchots.

Cinq étapes, cinq fonctions, chacune testée. Toutes renvoient un **nouveau**
DataFrame : on ne modifie jamais la donnée d'origine, sinon impossible de
comparer l'avant et l'après.

Usage :
    python exercice_nettoyage.py
"""

from __future__ import annotations

import pandas as pd

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots


# --------------------------------------------------------------- question 1
def compter_manquants(df: pd.DataFrame) -> dict[str, int]:
    """Le point de départ de tout nettoyage : où sont les trous ?"""
    return {colonne: int(n) for colonne, n in df.isnull().sum().items() if n > 0}


# --------------------------------------------------------------- question 2
def remplir_medianes(df: pd.DataFrame) -> pd.DataFrame:
    """Remplit les trous numériques par la médiane de leur colonne.

    Pourquoi la médiane et non la moyenne ? Parce qu'elle ne bouge pas quand
    une valeur aberrante traîne dans les données : un manchot saisi à 45 000 g
    au lieu de 4 500 g déplacerait la moyenne, pas la médiane.
    """
    propre = df.copy()
    for colonne in COLONNES_NUMERIQUES:
        if colonne in propre.columns:
            propre[colonne] = propre[colonne].fillna(propre[colonne].median())
    return propre


def remplir_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Pour le texte, la médiane n'existe pas : on prend la valeur la plus fréquente."""
    propre = df.copy()
    for colonne in propre.columns:
        if not pd.api.types.is_numeric_dtype(propre[colonne]) and propre[colonne].isnull().any():
            propre[colonne] = propre[colonne].fillna(propre[colonne].mode()[0])
    return propre


# --------------------------------------------------------------- question 3
def supprimer_doublons(df: pd.DataFrame) -> pd.DataFrame:
    """Deux lignes identiques comptent double dans l'entraînement, pour rien."""
    return df.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------- question 4
def encoder(df: pd.DataFrame, cible: str = CIBLE) -> pd.DataFrame:
    """One-hot encoding des colonnes textuelles, sauf la cible.

    `drop_first=False` garde une colonne par catégorie : plus lisible pour
    débuter. On encode `island` et `sex`, jamais `species` — c'est la réponse
    à prédire, elle reste telle quelle.
    """
    a_encoder = [
        colonne
        for colonne in df.columns
        if colonne != cible and not pd.api.types.is_numeric_dtype(df[colonne])
    ]
    return pd.get_dummies(df, columns=a_encoder, dtype=int)


# --------------------------------------------------------------- question 5
def normaliser(df: pd.DataFrame, colonnes: list[str] | None = None):
    """StandardScaler : chaque colonne devient de moyenne 0 et d'écart-type 1.

    Renvoie (DataFrame normalisé, scaler). **Gardez le scaler** : il faudra
    appliquer exactement la même transformation aux données de production
    (chapitre 42), sinon les prédictions seront fausses.
    """
    from sklearn.preprocessing import StandardScaler

    colonnes = colonnes or [c for c in COLONNES_NUMERIQUES if c in df.columns]
    resultat = df.copy()
    scaler = StandardScaler()
    resultat[colonnes] = scaler.fit_transform(resultat[colonnes])
    return resultat, scaler


def nettoyer_tout(df: pd.DataFrame):
    """La chaîne complète, dans l'ordre qui compte."""
    etape = remplir_medianes(df)
    etape = remplir_categories(etape)
    etape = supprimer_doublons(etape)
    etape = encoder(etape)
    return normaliser(etape)


def main() -> None:
    df = charger_manchots()

    print("1. valeurs manquantes au depart :")
    for colonne, n in compter_manquants(df).items():
        print(f"     {colonne:<20} {n}")

    rempli = remplir_categories(remplir_medianes(df))
    print(f"2. apres remplissage : {compter_manquants(rempli) or 'aucun trou'}")

    sans_doublon = supprimer_doublons(rempli)
    print(f"3. doublons supprimes : {len(rempli) - len(sans_doublon)}")

    encode = encoder(sans_doublon)
    print(f"4. colonnes apres encodage : {len(encode.columns)}")
    print(f"     {list(encode.columns)}")

    final, scaler = normaliser(encode)
    print("5. apres normalisation (moyennes proches de 0) :")
    print(final[COLONNES_NUMERIQUES].mean().round(6).to_string())
    print(f"\n   moyennes memorisees par le scaler : {scaler.mean_.round(1)}")


if __name__ == "__main__":
    main()

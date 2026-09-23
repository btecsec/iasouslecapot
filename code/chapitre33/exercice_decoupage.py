"""Chapitre 33 — corrigé de l'exercice : découper les manchots.

Usage :
    python exercice_decoupage.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots


def preparer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les features (X) de la cible (y).

    C'est le geste de départ de tout projet supervisé — et le premier endroit
    où l'on peut créer une fuite : si la cible reste dans X, le score sera
    parfait et le modèle inutile.
    """
    propre = df.dropna(subset=COLONNES_NUMERIQUES + [CIBLE]).copy()
    X = propre[COLONNES_NUMERIQUES]
    y = propre[CIBLE]
    return X, y


# --------------------------------------------------------------- question 1
def decouper(X, y, test_size: float = 0.2, stratifier: bool = False, graine: int = 42):
    """80 / 20, reproductible grâce à random_state."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=graine,
        stratify=y if stratifier else None,
    )


# --------------------------------------------------------------- question 2
def tailles(X_train, X_test) -> dict[str, int]:
    """Combien d'exemples de chaque côté."""
    return {"train": len(X_train), "test": len(X_test)}


# --------------------------------------------------------------- question 3
def proportions(y) -> dict[str, float]:
    """La part de chaque espèce, arrondie — pour comparer avant / après."""
    return {k: round(float(v), 3) for k, v in y.value_counts(normalize=True).items()}


def ecart_max(reference: dict[str, float], obtenu: dict[str, float]) -> float:
    """Le plus grand écart de proportion entre deux jeux.

    C'est la mesure qui montre l'intérêt de `stratify` : sans lui, l'écart
    grimpe ; avec lui, il reste minuscule.
    """
    return max(abs(reference[classe] - obtenu.get(classe, 0.0)) for classe in reference)


# --------------------------------------------------------------- question 5
def validation_croisee(X, y, plis: int = 5, graine: int = 42):
    """5 plis sur un modèle simple, dans un pipeline pour éviter la fuite.

    Le `make_pipeline` garantit que le StandardScaler est ajusté **sur le pli
    d'entraînement seulement**, à chaque tour. Normaliser avant l'appel serait
    une fuite discrète, et le score serait légèrement trop beau.
    """
    modele = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=graine)
    )
    return cross_val_score(modele, X, y, cv=plis)


def main() -> None:
    X, y = preparer(charger_manchots())

    print("--- sans stratification ---")
    X_tr, X_te, y_tr, y_te = decouper(X, y)
    print(f"1-2. tailles : {tailles(X_tr, X_te)}")
    print(f"     complet : {proportions(y)}")
    print(f"     test    : {proportions(y_te)}")
    print(f"     ecart max : {ecart_max(proportions(y), proportions(y_te)):.3f}")

    print("\n--- avec stratify=y ---")
    X_tr, X_te, y_tr, y_te = decouper(X, y, stratifier=True)
    print(f"3.   test    : {proportions(y_te)}")
    print(f"     ecart max : {ecart_max(proportions(y), proportions(y_te)):.3f}")

    print("\n4. On normalise APRES le decoupage : la moyenne et l'ecart-type")
    print("   doivent etre calcules sur le train seul, sinon le test fuite.")

    scores = validation_croisee(X, y)
    print(f"\n5. validation croisee 5 plis : {scores.round(4)}")
    print(f"   moyenne {scores.mean():.4f} (+/- {scores.std():.4f})")


if __name__ == "__main__":
    main()

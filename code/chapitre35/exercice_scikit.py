"""Chapitre 35 — corrigé de l'exercice : le projet complet avec Scikit-learn.

Charger, nettoyer, encoder, découper, entraîner, évaluer — puis tout refaire
avec un Pipeline pour supprimer le risque de fuite.

Usage :
    python exercice_scikit.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42


# --------------------------------------------------------------- question 1
def charger_et_preparer() -> tuple[pd.DataFrame, pd.Series]:
    """Charge, retire les lignes incomplètes, encode l'île et le sexe."""
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE]).copy()
    df["sex"] = df["sex"].fillna(df["sex"].mode()[0])
    encode = pd.get_dummies(df, columns=["island", "sex"], dtype=int)
    X = encode.drop(columns=[CIBLE])
    y = encode[CIBLE]
    return X, y


# --------------------------------------------------------------- question 2
def decouper(X, y, graine: int = GRAINE):
    """80 / 20 stratifié."""
    return train_test_split(X, y, test_size=0.2, random_state=graine, stratify=y)


# --------------------------------------------------------------- question 3
def entrainer_sans_pipeline(X_train, X_test, y_train, graine: int = GRAINE):
    """La version « à la main » : normaliser, puis entraîner.

    Notez le `transform` seul sur le test : c'est LA ligne qu'on oublie, et
    l'oublier crée une fuite de données invisible.
    """
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)

    modele = LogisticRegression(max_iter=1000, random_state=graine)
    modele.fit(X_train_n, y_train)
    return modele, scaler, X_test_n


# --------------------------------------------------------------- question 5
def entrainer_avec_pipeline(X_train, y_train, graine: int = GRAINE):
    """La même chose, mais impossible à rater : le scaler est dans le modèle."""
    modele = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=graine)
    )
    modele.fit(X_train, y_train)
    return modele


# --------------------------------------------------------------- question 4
def rapport(modele, X_test, y_test) -> dict:
    """Le classification_report sous forme de dictionnaire, pour le tester."""
    return classification_report(y_test, modele.predict(X_test), output_dict=True)


def classe_la_moins_bien_predite(rapport_dict: dict) -> str:
    """Celle dont le F1 est le plus faible — pas forcément la plus rare."""
    classes = {
        nom: valeurs["f1-score"]
        for nom, valeurs in rapport_dict.items()
        if isinstance(valeurs, dict) and nom not in {"macro avg", "weighted avg"}
    }
    return min(classes, key=classes.get)


def main() -> None:
    X, y = charger_et_preparer()
    X_train, X_test, y_train, y_test = decouper(X, y)
    print(f"1-2. {len(X_train)} exemples d'entrainement, {len(X_test)} de test")
    print(f"     {X.shape[1]} colonnes apres encodage")

    modele, _, X_test_n = entrainer_sans_pipeline(X_train, X_test, y_train)
    score_manuel = modele.score(X_test_n, y_test)
    print(f"3.   score (version manuelle) : {score_manuel:.4f}")

    print("4.   classification_report :")
    print(classification_report(y_test, modele.predict(X_test_n)))
    detail = rapport(modele, X_test_n, y_test)
    print(f"     classe la moins bien predite : {classe_la_moins_bien_predite(detail)}")

    pipeline = entrainer_avec_pipeline(X_train, y_train)
    score_pipeline = pipeline.score(X_test, y_test)
    print(f"5.   score (pipeline) : {score_pipeline:.4f}")
    print(
        "     identique : le pipeline ne change pas le resultat,\n"
        "     il rend l'erreur de fuite impossible."
    )


if __name__ == "__main__":
    main()

"""Chapitre 43 — projet pratique : un modèle de A à Z.

Les six étapes du livre, enchaînées dans un seul script réutilisable :

    explorer -> nettoyer -> decouper -> entrainer -> evaluer -> sauvegarder

La fonction `projet_complet` ne connaît rien aux manchots : on lui passe un
DataFrame et le nom de la colonne cible. C'est le défi de l'exercice — la
démarche ne dépend pas du dataset.

Usage :
    python projet_manchots.py
    python projet_manchots.py --iris     # le défi : le même code sur un autre jeu
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import CIBLE, charger_manchots

GRAINE = 42
MODELE = Path(__file__).parent / "modele_final.joblib"


# ------------------------------------------------------------- étape 1 : explorer
def explorer(df: pd.DataFrame, cible: str) -> dict:
    """Le diagnostic d'entrée : taille, trous, équilibre des classes."""
    return {
        "lignes": len(df),
        "colonnes": len(df.columns),
        "manquants": int(df.isnull().sum().sum()),
        "classes": df[cible].value_counts().to_dict(),
        "classe_majoritaire": float(df[cible].value_counts(normalize=True).max()),
    }


# ------------------------------------------------------------- étape 2 : nettoyer
def nettoyer(df: pd.DataFrame, cible: str) -> tuple[pd.DataFrame, pd.Series]:
    """Trous comblés, catégories encodées, cible séparée."""
    propre = df.dropna(subset=[cible]).copy()

    numeriques = [
        c
        for c in propre.columns
        if c != cible and pd.api.types.is_numeric_dtype(propre[c])
    ]
    categorielles = [c for c in propre.columns if c != cible and c not in numeriques]

    for colonne in numeriques:
        propre[colonne] = propre[colonne].fillna(propre[colonne].median())
    for colonne in categorielles:
        propre[colonne] = propre[colonne].fillna(propre[colonne].mode()[0])

    X = pd.get_dummies(propre.drop(columns=[cible]), columns=categorielles, dtype=int)
    return X, propre[cible]


# ------------------------------------------------------------- étape 3 : découper
def decouper(X, y, graine: int = GRAINE):
    return train_test_split(X, y, test_size=0.2, random_state=graine, stratify=y)


# ------------------------------------------------------------ étape 4 : entraîner
def entrainer(X_train, y_train, graine: int = GRAINE):
    """Un pipeline (donc pas de fuite possible), réglé par validation croisée."""
    recherche = GridSearchCV(
        make_pipeline(
            StandardScaler(), RandomForestClassifier(random_state=graine)
        ),
        {"randomforestclassifier__max_depth": [3, 5, None]},
        cv=5,
    )
    recherche.fit(X_train, y_train)
    return recherche


# -------------------------------------------------------------- étape 5 : évaluer
def evaluer(modele, X_test, y_test) -> dict:
    predictions = modele.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "rapport": classification_report(y_test, predictions, output_dict=True),
    }


# ---------------------------------------------------------- étape 6 : sauvegarder
def sauvegarder(modele, colonnes, chemin: Path = MODELE) -> Path:
    """Le pipeline complet + l'ordre des colonnes (chapitre 42)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modele": modele, "colonnes": list(colonnes)}, chemin)
    return chemin


def recharger_et_predire(chemin: Path, X) -> list:
    """SÉCURITÉ : ne chargez que des fichiers que vous avez produits (chapitre 42).

    joblib exécute du code en désérialisant.
    """
    paquet = joblib.load(chemin)
    return list(paquet["modele"].predict(X[paquet["colonnes"]]))


def projet_complet(df: pd.DataFrame, cible: str, graine: int = GRAINE) -> dict:
    """Les six étapes enchaînées. Aucune ligne ne parle de manchots."""
    diagnostic = explorer(df, cible)
    X, y = nettoyer(df, cible)
    X_train, X_test, y_train, y_test = decouper(X, y, graine)

    baseline = RandomForestClassifier(random_state=graine).fit(X_train, y_train)
    score_baseline = float(baseline.score(X_test, y_test))

    modele = entrainer(X_train, y_train, graine)
    resultats = evaluer(modele, X_test, y_test)

    return {
        "diagnostic": diagnostic,
        "colonnes": list(X.columns),
        "score_baseline": score_baseline,
        "meilleurs_reglages": modele.best_params_,
        "accuracy": resultats["accuracy"],
        "rapport": resultats["rapport"],
        "modele": modele,
        "X_test": X_test,
        "y_test": y_test,
    }


def main() -> None:
    if "--iris" in sys.argv:
        from sklearn.datasets import load_iris

        brut = load_iris(as_frame=True)
        df = brut.frame.rename(columns={"target": "espece"})
        cible = "espece"
        print("--- defi : le meme code sur le jeu iris ---")
    else:
        df, cible = charger_manchots(), CIBLE
        print("--- projet manchots ---")

    resultat = projet_complet(df, cible)

    print(f"1. exploration : {resultat['diagnostic']}")
    print(f"2. nettoyage   : {len(resultat['colonnes'])} colonnes apres encodage")
    print(f"3. decoupage   : 80/20 stratifie, graine {GRAINE}")
    print(f"4. baseline    : {resultat['score_baseline']:.4f}")
    print(f"   apres reglage : {resultat['accuracy']:.4f}")
    print(f"   meilleurs reglages : {resultat['meilleurs_reglages']}")
    print(
        f"   gain : {resultat['accuracy'] - resultat['score_baseline']:+.4f} "
        f"(a comparer au plancher de {resultat['diagnostic']['classe_majoritaire']:.2f})"
    )

    chemin = sauvegarder(resultat["modele"], resultat["colonnes"])
    identiques = recharger_et_predire(chemin, resultat["X_test"]) == list(
        resultat["modele"].predict(resultat["X_test"])
    )
    print(f"5-6. modele sauvegarde dans {chemin.name}, rechargement fidele : {identiques}")

    print(
        "\nLa recette generale, en trois phrases :\n"
        "  1. Regarder les donnees avant de coder, et corriger ce qui cloche.\n"
        "  2. Isoler un jeu de test des le depart, et n'y toucher qu'a la fin.\n"
        "  3. Partir du modele le plus simple, mesurer, et ne complexifier\n"
        "     que si la mesure le justifie."
    )


if __name__ == "__main__":
    main()

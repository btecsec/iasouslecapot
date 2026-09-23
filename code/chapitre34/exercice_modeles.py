"""Chapitre 34 — corrigé de l'exercice : trois modèles simples, comparés.

Usage :
    python exercice_modeles.py
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42


def donnees_decoupees(graine: int = GRAINE):
    """X/y nettoyés, découpés 80/20 avec stratification (chapitre 33)."""
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X, y = df[COLONNES_NUMERIQUES], df[CIBLE]
    return train_test_split(X, y, test_size=0.2, random_state=graine, stratify=y)


def modeles(graine: int = GRAINE) -> dict:
    """Les candidats de l'exercice, plus la baseline la plus bête possible.

    Le `DummyClassifier` répond toujours la classe majoritaire. Il ne sert à
    rien — sauf à donner le score qu'il faut battre, et c'est essentiel.
    """
    return {
        "baseline (classe majoritaire)": DummyClassifier(strategy="most_frequent"),
        "regression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=graine)
        ),
        "arbre de decision (max_depth=3)": DecisionTreeClassifier(
            max_depth=3, random_state=graine
        ),
        "foret aleatoire (100 arbres)": RandomForestClassifier(
            n_estimators=100, random_state=graine
        ),
    }


def evaluer_tous(graine: int = GRAINE) -> dict[str, float]:
    """Entraîne chaque modèle et renvoie son score sur le test."""
    X_train, X_test, y_train, y_test = donnees_decoupees(graine)
    resultats = {}
    for nom, modele in modeles(graine).items():
        modele.fit(X_train, y_train)
        resultats[nom] = float(modele.score(X_test, y_test))
    return resultats


def meilleur(resultats: dict[str, float]) -> str:
    """Le nom du modèle au meilleur score."""
    return max(resultats, key=resultats.get)


def le_simple_suffit(resultats: dict[str, float], marge: float = 0.02) -> bool:
    """Vrai si la régression logistique est à moins de `marge` du meilleur.

    C'est la question 4 de l'exercice, transformée en critère explicite :
    un gain inférieur à 2 points ne justifie pas un modèle plus lourd et
    moins explicable.
    """
    simple = resultats["regression logistique"]
    return max(resultats.values()) - simple <= marge


def main() -> None:
    resultats = evaluer_tous()

    print("Scores sur le jeu de test :")
    for nom, score in resultats.items():
        print(f"  {nom:<34} {score:.4f}")

    print(f"\nMeilleur : {meilleur(resultats)}")
    print(
        "La regression logistique suffit-elle ? "
        f"{'oui' if le_simple_suffit(resultats) else 'non'}"
    )
    print(
        "\nSans la baseline, un score de 0.97 semblerait extraordinaire.\n"
        "Avec elle, on sait que repondre toujours « Adelie » donne deja "
        f"{resultats['baseline (classe majoritaire)']:.2f}."
    )


if __name__ == "__main__":
    main()

"""Chapitre 41 — corrigé de l'exercice : régler les hyperparamètres.

Baseline -> recherche par grille -> recherche aléatoire -> vérification finale
sur le test, une seule fois.

Usage :
    python exercice_reglages.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42

GRILLE = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5, None],
}


def preparer(graine: int = GRAINE):
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X, y = df[COLONNES_NUMERIQUES], df[CIBLE]
    return train_test_split(X, y, test_size=0.2, random_state=graine, stratify=y)


# --------------------------------------------------------------- question 1
def baseline(X_train, y_train, graine: int = GRAINE) -> float:
    """Le score des réglages par défaut, mesuré en validation croisée.

    Point important : la baseline se mesure sur le **train**, comme tout le
    reste. Le test n'intervient qu'à la toute fin.
    """
    modele = RandomForestClassifier(random_state=graine)
    return float(cross_val_score(modele, X_train, y_train, cv=5).mean())


# --------------------------------------------------------------- question 2
def recherche_grille(X_train, y_train, graine: int = GRAINE):
    """Toutes les combinaisons : 3 × 4 = 12 modèles, × 5 plis = 60 entraînements."""
    recherche = GridSearchCV(
        RandomForestClassifier(random_state=graine), GRILLE, cv=5, n_jobs=1
    )
    debut = perf_counter()
    recherche.fit(X_train, y_train)
    return recherche, perf_counter() - debut


# --------------------------------------------------------------- question 4
def recherche_aleatoire(X_train, y_train, n_iter: int = 10, graine: int = GRAINE):
    """10 combinaisons tirées au hasard, au lieu des 12 possibles."""
    recherche = RandomizedSearchCV(
        RandomForestClassifier(random_state=graine),
        GRILLE,
        n_iter=n_iter,
        cv=5,
        random_state=graine,
        n_jobs=1,
    )
    debut = perf_counter()
    recherche.fit(X_train, y_train)
    return recherche, perf_counter() - debut


def nombre_de_combinaisons(grille: dict = None) -> int:
    """Le produit des longueurs : ce que teste une recherche par grille."""
    grille = grille or GRILLE
    return int(np.prod([len(valeurs) for valeurs in grille.values()]))


def gain(score_regle: float, score_baseline: float) -> float:
    """Ce que le réglage a réellement rapporté, en points."""
    return score_regle - score_baseline


# --------------------------------------------------------------- question 5
def verification_finale(recherche, X_test, y_test) -> float:
    """Le test, une seule fois, à la toute fin. Jamais avant."""
    return float(recherche.best_estimator_.score(X_test, y_test))


def main() -> None:
    X_train, X_test, y_train, y_test = preparer()

    score_baseline = baseline(X_train, y_train)
    print(f"1. baseline (reglages par defaut) : {score_baseline:.4f}")

    grille, duree_grille = recherche_grille(X_train, y_train)
    print(f"\n2-3. recherche par grille ({nombre_de_combinaisons()} combinaisons)")
    print(f"     meilleurs reglages : {grille.best_params_}")
    print(f"     score de validation : {grille.best_score_:.4f}")
    print(f"     gain : {gain(grille.best_score_, score_baseline):+.4f}")
    print(f"     duree : {duree_grille:.1f} s")

    aleatoire, duree_aleatoire = recherche_aleatoire(X_train, y_train, n_iter=10)
    print("\n4. recherche aleatoire (10 tirages)")
    print(f"     meilleurs reglages : {aleatoire.best_params_}")
    print(f"     score de validation : {aleatoire.best_score_:.4f}")
    print(f"     duree : {duree_aleatoire:.1f} s")
    print(
        f"     ecart avec la grille : "
        f"{aleatoire.best_score_ - grille.best_score_:+.4f}"
    )

    print(f"\n5. verification finale sur le test : {verification_finale(grille, X_test, y_test):.4f}")
    print("   (une seule fois, apres tous les reglages)")


if __name__ == "__main__":
    main()

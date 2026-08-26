"""Chapitre 40 — corrigé de l'exercice : évaluer pour de bon.

Deux problèmes, deux familles de métriques :
  * détecter une espèce rare (classification) -> matrice de confusion,
    précision, rappel ;
  * prédire la masse d'un manchot (régression) -> MAE, RMSE, R².

Usage :
    python exercice_metriques.py
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42
ESPECE_RARE = "Chinstrap"  # 68 manchots sur 344, soit 20 %


# ----------------------------------------------------- classification
def preparer_detection(graine: int = GRAINE):
    """Un problème binaire : ce manchot est-il un Chinstrap ?

    20 % de positifs : assez déséquilibré pour que l'accuracy commence à
    mentir, assez équilibré pour rester lisible.
    """
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X = df[COLONNES_NUMERIQUES]
    y = (df[CIBLE] == ESPECE_RARE).astype(int)
    return train_test_split(X, y, test_size=0.3, random_state=graine, stratify=y)


def entrainer_detecteur(X_train, y_train, graine: int = GRAINE):
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=graine)
    ).fit(X_train, y_train)


# --------------------------------------------------------------- question 1
def matrice(y_vrai, y_predit) -> dict[str, int]:
    """La matrice de confusion, avec des noms plutôt qu'un tableau muet.

        [[VN, FP],
         [FN, VP]]
    """
    vn, fp, fn, vp = confusion_matrix(y_vrai, y_predit, labels=[0, 1]).ravel()
    return {
        "vrais_negatifs": int(vn),
        "faux_positifs": int(fp),  # fausse alerte
        "faux_negatifs": int(fn),  # l'oubli, le plus coûteux en dépistage
        "vrais_positifs": int(vp),
    }


# --------------------------------------------------------------- question 2
def precision_rappel(y_vrai, y_predit) -> dict[str, float]:
    """Précision et rappel calculés à la main, à partir de la matrice.

    Les refaire soi-même une fois suffit à ne plus jamais les confondre.
    """
    c = matrice(y_vrai, y_predit)
    vp, fp, fn = c["vrais_positifs"], c["faux_positifs"], c["faux_negatifs"]
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {"precision": precision, "rappel": rappel, "f1": f1}


def rapport(y_vrai, y_predit) -> dict:
    return classification_report(y_vrai, y_predit, output_dict=True, zero_division=0)


# --------------------------------------------------------------- question 3
def privilegier(cout_faux_negatif: float, cout_faux_positif: float) -> str:
    """Quelle métrique viser, selon ce que coûte chaque type d'erreur.

    Rater un positif coûte cher (dépistage) -> **rappel**.
    Une fausse alerte coûte cher (anti-spam) -> **précision**.
    """
    if cout_faux_negatif > cout_faux_positif:
        return "rappel"
    if cout_faux_positif > cout_faux_negatif:
        return "precision"
    return "f1"


def seuil_ajuste(modele, X, seuil: float = 0.5) -> np.ndarray:
    """Prédire avec un autre seuil que 0,5.

    C'est le vrai levier : baisser le seuil augmente le rappel (on ose plus
    d'alertes) et fait baisser la précision. Aucun réentraînement nécessaire.
    """
    return (modele.predict_proba(X)[:, 1] >= seuil).astype(int)


# --------------------------------------------------------------- question 4
def preparer_regression(graine: int = GRAINE):
    """Prédire la masse (en grammes) à partir des trois autres mesures."""
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    features = [c for c in COLONNES_NUMERIQUES if c != "body_mass_g"]
    return train_test_split(
        df[features], df["body_mass_g"], test_size=0.3, random_state=graine
    )


def metriques_regression(y_vrai, y_predit) -> dict[str, float]:
    """MAE, RMSE, R² — et pourquoi les trois se complètent.

    MAE  : l'erreur moyenne, dans l'unité d'origine (des grammes).
    RMSE : idem, mais les grosses erreurs pèsent davantage.
    R²   : la part de variance expliquée ; 0 = aussi bon que prédire la
           moyenne, 1 = parfait.
    """
    return {
        "mae": float(mean_absolute_error(y_vrai, y_predit)),
        "rmse": float(np.sqrt(mean_squared_error(y_vrai, y_predit))),
        "r2": float(r2_score(y_vrai, y_predit)),
    }


# --------------------------------------------------------------- question 5
def accuracy_du_modele_paresseux(y_test) -> float:
    """Le score d'un modèle qui répond toujours « pas un Chinstrap ».

    C'est la démonstration que l'accuracy seule est trompeuse.
    """
    return float((y_test == 0).mean())


def main() -> None:
    X_train, X_test, y_train, y_test = preparer_detection()
    modele = entrainer_detecteur(X_train, y_train)
    y_predit = modele.predict(X_test)

    print(f"--- detection de l'espece rare ({ESPECE_RARE}) ---")
    print(f"{int(y_test.sum())} positifs sur {len(y_test)} exemples de test\n")

    print("1. matrice de confusion :")
    for nom, valeur in matrice(y_test, y_predit).items():
        print(f"     {nom:<16} {valeur}")

    print("\n2. precision et rappel :")
    for nom, valeur in precision_rappel(y_test, y_predit).items():
        print(f"     {nom:<10} {valeur:.4f}")

    print("\n3. si rater un positif coute tres cher -> on privilegie le")
    print(f"   {privilegier(cout_faux_negatif=10, cout_faux_positif=1)}.")
    for seuil in (0.5, 0.3, 0.1):
        scores = precision_rappel(y_test, seuil_ajuste(modele, X_test, seuil))
        print(
            f"     seuil {seuil:<4} -> precision {scores['precision']:.3f} "
            f"| rappel {scores['rappel']:.3f}"
        )

    Xr_train, Xr_test, yr_train, yr_test = preparer_regression()
    regresseur = RandomForestRegressor(n_estimators=100, random_state=GRAINE)
    regresseur.fit(Xr_train, yr_train)
    scores = metriques_regression(yr_test, regresseur.predict(Xr_test))
    print("\n4. regression (prevoir la masse) :")
    print(f"     MAE  {scores['mae']:.1f} g")
    print(f"     RMSE {scores['rmse']:.1f} g")
    print(f"     R2   {scores['r2']:.4f}")

    paresseux = accuracy_du_modele_paresseux(y_test)
    print("\n5. un modele qui repond toujours « non » obtient une accuracy de")
    print(f"   {paresseux:.4f} — sans detecter un seul {ESPECE_RARE}.")
    print("   Voila pourquoi l'accuracy seule ne veut rien dire sur une classe rare.")


if __name__ == "__main__":
    main()

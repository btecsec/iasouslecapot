"""Tests du projet complet du chapitre 43.

Le test le plus important du fichier est le dernier : la même chaîne, sans une
ligne de changement, doit fonctionner sur un tout autre jeu de données.
"""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import donnees  # noqa: E402
import projet_manchots as projet  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return donnees.manchots_synthetiques()


@pytest.fixture(scope="module")
def resultat(df):
    return projet.projet_complet(df, donnees.CIBLE)


# ------------------------------------------------------------------ étape 1
def test_lexploration_compte_tout(df):
    diagnostic = projet.explorer(df, donnees.CIBLE)
    assert diagnostic["lignes"] == 344
    assert diagnostic["colonnes"] == 7
    assert diagnostic["manquants"] == 19  # 4 colonnes × 2 trous + 11 sur le sexe


def test_lexploration_donne_le_plancher_a_battre(df):
    """La classe majoritaire : le score d'un modèle qui ne fait rien."""
    assert projet.explorer(df, donnees.CIBLE)["classe_majoritaire"] == pytest.approx(
        0.44, abs=0.01
    )


# ------------------------------------------------------------------ étape 2
def test_le_nettoyage_ne_laisse_aucun_trou(df):
    X, y = projet.nettoyer(df, donnees.CIBLE)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_le_nettoyage_ne_laisse_que_des_nombres(df):
    import pandas as pd

    X, _ = projet.nettoyer(df, donnees.CIBLE)
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_la_cible_nest_pas_dans_les_features(df):
    X, _ = projet.nettoyer(df, donnees.CIBLE)
    assert donnees.CIBLE not in X.columns


# ------------------------------------------------------------------ étape 3
def test_le_decoupage_est_stratifie(df):
    X, y = projet.nettoyer(df, donnees.CIBLE)
    _, _, y_train, y_test = projet.decouper(X, y)
    for classe in y.unique():
        assert abs((y_train == classe).mean() - (y_test == classe).mean()) < 0.02


# --------------------------------------------------------------- étapes 4-5
def test_le_modele_bat_tres_largement_le_plancher(resultat):
    plancher = resultat["diagnostic"]["classe_majoritaire"]
    assert resultat["accuracy"] > plancher + 0.40


def test_le_gain_du_reglage_est_mesure(resultat):
    """Question 2 de l'exercice : combien le réglage a-t-il rapporté ?"""
    gain = resultat["accuracy"] - resultat["score_baseline"]
    assert -0.05 <= gain <= 0.10  # quelques points au plus, dans un sens ou l'autre


def test_le_rapport_detaille_chaque_classe(resultat):
    for espece in ("Adelie", "Chinstrap", "Gentoo"):
        assert espece in resultat["rapport"]


def test_le_modele_choisi_vient_bien_de_la_grille(resultat):
    assert "randomforestclassifier__max_depth" in resultat["meilleurs_reglages"]


# ------------------------------------------------------------------ étape 6
def test_le_modele_recharge_predit_pareil(tmp_path, resultat):
    """Question 3 : rechargez dans un nouveau script et vérifiez."""
    chemin = projet.sauvegarder(
        resultat["modele"], resultat["colonnes"], tmp_path / "modele.joblib"
    )
    recharge = projet.recharger_et_predire(chemin, resultat["X_test"])
    assert recharge == list(resultat["modele"].predict(resultat["X_test"]))


def test_le_paquet_survit_a_un_desordre_des_colonnes(tmp_path, resultat):
    chemin = projet.sauvegarder(
        resultat["modele"], resultat["colonnes"], tmp_path / "modele.joblib"
    )
    melange = resultat["X_test"][list(reversed(resultat["colonnes"]))]
    assert projet.recharger_et_predire(chemin, melange) == projet.recharger_et_predire(
        chemin, resultat["X_test"]
    )


# ------------------------------------------------------------------- le défi
def test_la_meme_chaine_fonctionne_sur_iris():
    """Le défi de l'exercice : la démarche ne dépend pas du dataset."""
    from sklearn.datasets import load_iris

    df = load_iris(as_frame=True).frame.rename(columns={"target": "espece"})
    resultat = projet.projet_complet(df, "espece")
    assert resultat["accuracy"] > 0.85
    assert resultat["diagnostic"]["lignes"] == 150


def test_la_chaine_gere_un_jeu_avec_des_colonnes_texte():
    """Un dataset minuscule et bancal : la chaîne doit tenir debout."""
    import pandas as pd

    # 40 lignes : il en faut au moins 5 par classe dans le train, sinon la
    # validation croisée à 5 plis ne peut pas se faire.
    df = pd.DataFrame(
        {
            "mesure": [float(i) if i % 7 else None for i in range(40)],
            "couleur": (["rouge", "bleu", "rouge", None, "bleu"] * 8),
            "classe": ["a", "b"] * 20,
        }
    )
    resultat = projet.projet_complet(df, "classe")
    assert 0.0 <= resultat["accuracy"] <= 1.0
    assert "couleur_rouge" in resultat["colonnes"]

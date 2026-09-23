"""Tests du corrigé du chapitre 34."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import exercice_modeles as ex  # noqa: E402


@pytest.fixture(scope="module")
def resultats():
    return ex.evaluer_tous()


def test_les_quatre_modeles_sont_evalues(resultats):
    assert len(resultats) == 4


def test_tous_les_scores_sont_des_proportions(resultats):
    assert all(0.0 <= score <= 1.0 for score in resultats.values())


def test_la_baseline_correspond_a_la_classe_majoritaire(resultats):
    """Répondre toujours « Adelie » donne environ 44 %."""
    assert resultats["baseline (classe majoritaire)"] == pytest.approx(0.44, abs=0.05)


@pytest.mark.parametrize(
    "nom",
    [
        "regression logistique",
        "arbre de decision (max_depth=3)",
        "foret aleatoire (100 arbres)",
    ],
)
def test_chaque_vrai_modele_ecrase_la_baseline(resultats, nom):
    assert resultats[nom] > resultats["baseline (classe majoritaire)"] + 0.30


def test_la_regression_logistique_depasse_90_pourcent(resultats):
    assert resultats["regression logistique"] > 0.90


def test_le_modele_le_plus_simple_suffit(resultats):
    """La conclusion de l'exercice : l'écart avec la forêt est négligeable."""
    assert ex.le_simple_suffit(resultats, marge=0.05) is True


def test_le_meilleur_nest_pas_la_baseline(resultats):
    assert ex.meilleur(resultats) != "baseline (classe majoritaire)"


def test_larbre_est_bride_par_sa_profondeur():
    """max_depth=3 : trois questions oui/non, pas une de plus."""
    X_train, _, y_train, _ = ex.donnees_decoupees()
    arbre = ex.modeles()["arbre de decision (max_depth=3)"]
    arbre.fit(X_train, y_train)
    assert arbre.get_depth() <= 3


def test_la_foret_contient_bien_cent_arbres():
    foret = ex.modeles()["foret aleatoire (100 arbres)"]
    assert foret.n_estimators == 100


def test_les_resultats_sont_reproductibles():
    assert ex.evaluer_tous(graine=42) == ex.evaluer_tous(graine=42)


def test_les_trois_especes_sont_predites():
    """Un modèle qui n'en prédirait que deux passerait inaperçu à l'accuracy."""
    X_train, X_test, y_train, _ = ex.donnees_decoupees()
    modele = ex.modeles()["regression logistique"]
    modele.fit(X_train, y_train)
    assert len(set(modele.predict(X_test))) == 3


def test_le_decoupage_est_stratifie():
    _, _, y_train, y_test = ex.donnees_decoupees()
    part_train = (y_train == "Adelie").mean()
    part_test = (y_test == "Adelie").mean()
    assert abs(part_train - part_test) < 0.02

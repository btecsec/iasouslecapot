"""Tests du corrigé du chapitre 40."""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402

import exercice_metriques as ex  # noqa: E402


@pytest.fixture(scope="module")
def detection():
    X_train, X_test, y_train, y_test = ex.preparer_detection()
    modele = ex.entrainer_detecteur(X_train, y_train)
    return modele, X_test, y_test, modele.predict(X_test)


# --------------------------------------------- les métriques, sans modèle
def test_matrice_sur_un_cas_calcule_a_la_main():
    y_vrai = [0, 0, 1, 1, 1]
    y_predit = [0, 1, 1, 1, 0]
    assert ex.matrice(y_vrai, y_predit) == {
        "vrais_negatifs": 1,
        "faux_positifs": 1,
        "faux_negatifs": 1,
        "vrais_positifs": 2,
    }


def test_precision_et_rappel_sur_le_meme_cas():
    y_vrai = [0, 0, 1, 1, 1]
    y_predit = [0, 1, 1, 1, 0]
    scores = ex.precision_rappel(y_vrai, y_predit)
    assert scores["precision"] == pytest.approx(2 / 3)  # 2 VP sur 3 alertes
    assert scores["rappel"] == pytest.approx(2 / 3)  # 2 VP sur 3 vrais positifs


def test_un_modele_parfait_a_tout_a_un():
    scores = ex.precision_rappel([0, 1, 1], [0, 1, 1])
    assert scores == {"precision": 1.0, "rappel": 1.0, "f1": 1.0}


def test_un_modele_qui_ne_dit_jamais_oui_a_un_rappel_nul():
    scores = ex.precision_rappel([0, 1, 1], [0, 0, 0])
    assert scores["rappel"] == 0.0


def test_un_modele_qui_dit_toujours_oui_a_un_rappel_parfait():
    """Et une précision catastrophique : les deux se compensent toujours."""
    scores = ex.precision_rappel([0, 0, 1], [1, 1, 1])
    assert scores["rappel"] == 1.0
    assert scores["precision"] == pytest.approx(1 / 3)


def test_le_f1_est_la_moyenne_harmonique():
    scores = ex.precision_rappel([0, 1, 1, 1], [1, 1, 1, 0])
    p, r = scores["precision"], scores["rappel"]
    assert scores["f1"] == pytest.approx(2 * p * r / (p + r))


def test_nos_calculs_coincident_avec_ceux_de_scikit_learn():
    y_vrai = [0, 0, 1, 1, 1, 0, 1]
    y_predit = [0, 1, 1, 0, 1, 0, 1]
    maison = ex.precision_rappel(y_vrai, y_predit)
    officiel = ex.rapport(y_vrai, y_predit)["1"]
    assert maison["precision"] == pytest.approx(officiel["precision"])
    assert maison["rappel"] == pytest.approx(officiel["recall"])


# --------------------------------------------------------------- question 1
def test_la_matrice_couvre_tous_les_exemples(detection):
    _, _, y_test, y_predit = detection
    assert sum(ex.matrice(y_test, y_predit).values()) == len(y_test)


def test_le_detecteur_trouve_la_majorite_des_especes_rares(detection):
    _, _, y_test, y_predit = detection
    assert ex.matrice(y_test, y_predit)["vrais_positifs"] >= 10


# --------------------------------------------------------------- question 3
def test_privilegier_le_rappel_quand_un_oubli_coute_cher():
    assert ex.privilegier(cout_faux_negatif=10, cout_faux_positif=1) == "rappel"


def test_privilegier_la_precision_quand_une_fausse_alerte_coute_cher():
    assert ex.privilegier(cout_faux_negatif=1, cout_faux_positif=10) == "precision"


def test_baisser_le_seuil_augmente_le_rappel(detection):
    """Le levier gratuit : aucun réentraînement, juste un curseur."""
    modele, X_test, y_test, _ = detection
    haut = ex.precision_rappel(y_test, ex.seuil_ajuste(modele, X_test, 0.5))
    bas = ex.precision_rappel(y_test, ex.seuil_ajuste(modele, X_test, 0.1))
    assert bas["rappel"] >= haut["rappel"]


def test_baisser_le_seuil_fait_baisser_la_precision(detection):
    modele, X_test, y_test, _ = detection
    haut = ex.precision_rappel(y_test, ex.seuil_ajuste(modele, X_test, 0.5))
    bas = ex.precision_rappel(y_test, ex.seuil_ajuste(modele, X_test, 0.1))
    assert bas["precision"] <= haut["precision"]


def test_un_seuil_tres_bas_declenche_beaucoup_dalertes(detection):
    modele, X_test, y_test, _ = detection
    assert ex.seuil_ajuste(modele, X_test, 0.01).sum() > ex.seuil_ajuste(
        modele, X_test, 0.5
    ).sum()


# --------------------------------------------------------------- question 4
def test_les_metriques_de_regression_sont_coherentes():
    y_vrai = np.array([3000.0, 4000.0, 5000.0])
    y_predit = np.array([3100.0, 3900.0, 5200.0])
    scores = ex.metriques_regression(y_vrai, y_predit)
    assert scores["mae"] == pytest.approx((100 + 100 + 200) / 3)
    assert scores["rmse"] >= scores["mae"]  # toujours vrai
    assert 0 < scores["r2"] <= 1


def test_une_prediction_parfaite_donne_un_r2_de_un():
    y = np.array([1.0, 2.0, 3.0])
    assert ex.metriques_regression(y, y)["r2"] == pytest.approx(1.0)


def test_predire_la_moyenne_donne_un_r2_nul():
    """Le repère du R² : 0 = aussi bon que ne rien apprendre."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    moyenne = np.full_like(y, y.mean())
    assert ex.metriques_regression(y, moyenne)["r2"] == pytest.approx(0.0)


def test_la_rmse_punit_plus_fort_une_grosse_erreur_que_la_mae():
    y_vrai = np.array([0.0, 0.0, 0.0, 0.0])
    une_grosse = np.array([0.0, 0.0, 0.0, 400.0])
    scores = ex.metriques_regression(y_vrai, une_grosse)
    assert scores["rmse"] > 1.9 * scores["mae"]


def test_le_modele_de_masse_est_utile():
    Xr_train, Xr_test, yr_train, yr_test = ex.preparer_regression()
    from sklearn.ensemble import RandomForestRegressor

    modele = RandomForestRegressor(n_estimators=50, random_state=ex.GRAINE)
    modele.fit(Xr_train, yr_train)
    scores = ex.metriques_regression(yr_test, modele.predict(Xr_test))
    assert scores["r2"] > 0.70
    assert scores["mae"] < 400  # moins de 400 g d'erreur moyenne


# --------------------------------------------------------------- question 5
def test_le_modele_paresseux_obtient_une_accuracy_flatteuse(detection):
    """80 % d'accuracy sans détecter un seul Chinstrap : l'accuracy ment."""
    _, _, y_test, _ = detection
    assert ex.accuracy_du_modele_paresseux(y_test) > 0.75


def test_le_modele_paresseux_a_un_rappel_nul(detection):
    """Le même modèle, jugé sur le rappel : zéro pointé."""
    _, _, y_test, _ = detection
    toujours_non = np.zeros(len(y_test), dtype=int)
    assert ex.precision_rappel(y_test, toujours_non)["rappel"] == 0.0

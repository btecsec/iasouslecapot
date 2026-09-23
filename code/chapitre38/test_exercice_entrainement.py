"""Tests du corrigé du chapitre 38."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import exercice_entrainement as ex  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def taux(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    return ex.comparer_taux(X_train, y_train)


# ------------------------------------------------- le repère du hasard
def test_la_perte_du_hasard_sur_trois_classes():
    assert ex.perte_du_hasard(3) == pytest.approx(1.0986, abs=1e-4)


def test_plus_il_y_a_de_classes_plus_le_hasard_coute_cher():
    assert ex.perte_du_hasard(2) < ex.perte_du_hasard(3) < ex.perte_du_hasard(1000)


# ------------------------------------------------- le lecteur de courbes
def test_une_courbe_qui_descend_regulierement():
    assert ex.profil_courbe(np.linspace(1.1, 0.02, 100)) == "descente saine"


def test_une_courbe_qui_baisse_a_peine_est_une_stagnation():
    """Taux trop petit : après 50 époques, on n'a presque rien gagné."""
    assert ex.profil_courbe(np.linspace(1.10, 1.05, 50)) == "stagnation"


def test_une_courbe_parfaitement_plate_ne_passe_pas_pour_saine():
    assert ex.profil_courbe([1.10] * 50) == "divergence"


def test_une_courbe_qui_remonte_est_une_divergence():
    assert ex.profil_courbe([0.5, 0.8, 1.4, 3.0]) == "divergence"


def test_une_courbe_infinie_est_une_divergence():
    assert ex.profil_courbe([1.0, 5.0, float("inf")]) == "divergence"


def test_un_pic_initial_enorme_est_repere():
    """Perte de départ à 9,7 pour 3 classes : les poids ont explosé."""
    courbe = np.concatenate([[9.7, 5.0, 2.0], np.linspace(1.0, 0.4, 50)])
    assert ex.profil_courbe(courbe) == "explosion puis recuperation"


# --------------------------------------------------------------- question 1
def test_les_trois_taux_sont_testes(taux):
    assert set(taux) == {0.001, 0.01, 1.0}


def test_un_taux_raisonnable_donne_une_perte_tres_basse(taux):
    assert taux[0.001]["perte_finale"] < 0.05
    assert taux[0.01]["perte_finale"] < 0.05


def test_un_taux_de_1_donne_une_perte_bien_pire(taux):
    """Le cœur de la question 1 : lr=1.0 finit ~100 fois plus haut."""
    assert taux[1.0]["perte_finale"] > 10 * taux[0.01]["perte_finale"]


def test_un_taux_de_1_fait_exploser_la_perte_au_depart(taux):
    """Preuve chiffrée : la courbe démarre bien au-dessus de log(3)."""
    assert max(taux[1.0]["courbe"]) > 3 * ex.perte_du_hasard(3)
    assert taux[1.0]["profil"] == "explosion puis recuperation"


def test_un_taux_petit_donne_une_descente_propre(taux):
    assert taux[0.001]["profil"] in {"descente saine", "descente bruitee"}


def test_les_courbes_demarrent_pres_du_hasard_quand_le_taux_est_sage(taux):
    assert taux[0.001]["courbe"][0] == pytest.approx(1.1, abs=0.3)


# --------------------------------------------------------------- question 2
def test_un_gros_batch_donne_une_descente_plus_reguliere(donnees_pretes):
    """La réponse à « plus stable ou plus rapide ? » : plus stable."""
    X_train, _, y_train, _ = donnees_pretes
    resultats = ex.comparer_batch(X_train, y_train)
    assert resultats[64]["irregularite"] < resultats[8]["irregularite"]


def test_toutes_les_tailles_de_batch_apprennent(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    resultats = ex.comparer_batch(X_train, y_train)
    assert all(detail["perte_finale"] < 0.1 for detail in resultats.values())


# --------------------------------------------------------------- question 3
def test_doubler_les_epoques_fait_encore_baisser_la_perte(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    resultats = ex.effet_epoques(X_train, y_train)
    assert resultats["perte_longue"] < resultats["perte_courte"]


def test_mais_le_gain_devient_marginal(donnees_pretes):
    """La perte baisse encore, mais de miettes : c'est le plateau."""
    X_train, _, y_train, _ = donnees_pretes
    resultats = ex.effet_epoques(X_train, y_train)
    assert resultats["gain"] < 0.05


# ------------------------------------------------------------- le tracé
def test_les_courbes_sont_tracables(taux):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = ex.courbes(taux)
    assert len(figure.axes[0].get_lines()) == 3
    plt.close(figure)

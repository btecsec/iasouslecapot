"""Tests du corrigé du chapitre 39.

Ces tests entraînent réellement trois petits réseaux : comptez une trentaine
de secondes. C'est le prix à payer pour vérifier une propriété qui ne se voit
qu'à l'entraînement — qu'un modèle trop gros pour ses données surapprend.
"""

import pytest

pytest.importorskip("tensorflow", reason="pip install -r requirements.txt")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

import exercice_surapprentissage as ex  # noqa: E402

EPOQUES = 150


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def essais(donnees_pretes):
    X_train, X_val, _, y_train, y_val, _ = donnees_pretes
    return {
        "gros": ex.experience(256, 0.0, X_train, y_train, X_val, y_val, EPOQUES),
        "gros_dropout": ex.experience(256, 0.5, X_train, y_train, X_val, y_val, EPOQUES),
        "petit": ex.experience(8, 0.0, X_train, y_train, X_val, y_val, EPOQUES),
    }


# ------------------------------------------------------------ les données
def test_le_jeu_dentrainement_est_minuscule(donnees_pretes):
    X_train = donnees_pretes[0]
    assert len(X_train) == 40


def test_la_validation_est_assez_grande_pour_etre_lisible(donnees_pretes):
    X_val = donnees_pretes[1]
    assert len(X_val) == 100


def test_dix_etiquettes_dentrainement_sont_fausses(donnees_pretes):
    """Sans étiquettes fausses, ce dataset est trop facile pour surapprendre."""
    _, _, _, y_bruite, _, _ = ex.preparer(etiquettes_fausses=10)
    _, _, _, y_propre, _, _ = ex.preparer(etiquettes_fausses=0)
    assert int((y_bruite != y_propre).sum()) == 10


# --------------------------------------------------------------- question 1
def test_le_gros_reseau_apprend_tout_par_coeur(essais):
    """Y compris les 10 étiquettes fausses : train accuracy = 100 %."""
    assert essais["gros"]["accuracy_train"] > 0.98


def test_le_gros_reseau_generalise_moins_bien_quil_napprend(essais):
    assert essais["gros"]["accuracy_val"] < essais["gros"]["accuracy_train"]


# --------------------------------------------------------------- question 2
def test_le_diagnostic_du_gros_reseau_est_le_surapprentissage(essais):
    assert essais["gros"]["diagnostic"] == "surapprentissage"


def test_lecart_du_gros_reseau_est_important(essais):
    assert essais["gros"]["ecart"] > 0.15


# --------------------------------------------------------------- question 3
def test_le_dropout_reduit_lecart(essais):
    """La réponse à « l'écart se réduit-il ? » : oui."""
    assert essais["gros_dropout"]["ecart"] < essais["gros"]["ecart"]


def test_le_dropout_ne_change_pas_le_nombre_de_parametres(essais):
    """Le dropout n'ajoute aucun poids : il éteint des neurones au hasard."""
    assert essais["gros_dropout"]["parametres"] == essais["gros"]["parametres"]


def test_le_dropout_fait_baisser_laccuracy_dentrainement(essais):
    """C'est le principe : on handicape volontairement l'apprentissage."""
    assert essais["gros_dropout"]["accuracy_train"] < essais["gros"]["accuracy_train"]


# --------------------------------------------------------------- question 4
def test_le_petit_reseau_a_mille_fois_moins_de_parametres(essais):
    assert essais["petit"]["parametres"] * 100 < essais["gros"]["parametres"]


def test_le_petit_reseau_ne_surapprend_pas(essais):
    assert essais["petit"]["diagnostic"] != "surapprentissage"


def test_le_petit_reseau_generalise_mieux(essais):
    """Moins puissant chez lui, meilleur ailleurs : tout le message du chapitre."""
    assert essais["petit"]["accuracy_val"] > essais["gros"]["accuracy_val"]
    assert essais["petit"]["accuracy_train"] < essais["gros"]["accuracy_train"]


def test_les_deux_remedes_reduisent_lecart(essais):
    """Dropout et réduction du modèle marchent tous les deux ; lequel gagne
    dépend du nombre d'époques, et ce n'est pas la question."""
    assert abs(essais["petit"]["ecart"]) < abs(essais["gros"]["ecart"])
    assert abs(essais["gros_dropout"]["ecart"]) < abs(essais["gros"]["ecart"])


# ------------------------------------------------- l'argument de l'arrêt anticipé
def test_le_gros_reseau_etait_meilleur_bien_avant_la_fin(essais):
    """Son sommet de validation arrive très tôt : un early stopping l'aurait sauvé."""
    detail = essais["gros"]
    assert detail["meilleure_val"] > detail["accuracy_val"] + 0.10
    assert detail["epoque_du_sommet"] < EPOQUES // 4


# ----------------------------------------------------- le diagnostic lui-même
def test_un_modele_nul_partout_est_diagnostique_sous_apprentissage():
    class FauxHistorique:
        history = {"accuracy": [0.35], "val_accuracy": [0.34]}

    assert ex.diagnostiquer(FauxHistorique()) == "sous-apprentissage"


def test_un_modele_bon_des_deux_cotes_est_equilibre():
    class FauxHistorique:
        history = {"accuracy": [0.95], "val_accuracy": [0.93]}

    assert ex.diagnostiquer(FauxHistorique()) == "equilibre"


def test_un_modele_excellent_chez_lui_et_mauvais_ailleurs_surapprend():
    class FauxHistorique:
        history = {"accuracy": [1.00], "val_accuracy": [0.70]}

    assert ex.diagnostiquer(FauxHistorique()) == "surapprentissage"


def test_le_sommet_de_validation_est_bien_le_maximum():
    class FauxHistorique:
        history = {"accuracy": [1.0] * 4, "val_accuracy": [0.5, 0.9, 0.7, 0.6]}

    assert ex.meilleure_validation(FauxHistorique()) == (1, 0.9)


def test_les_courbes_tracent_une_ligne_par_essai(essais):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = ex.courbes({nom: d["historique"] for nom, d in essais.items()})
    assert len(figure.axes[0].get_lines()) == 3
    plt.close(figure)


def test_les_donnees_de_test_ne_servent_jamais_a_lentrainement(donnees_pretes):
    X_train, X_val, X_test, *_ = donnees_pretes
    assert len(X_test) > 0
    assert not np.array_equal(X_train[:1], X_test[:1])

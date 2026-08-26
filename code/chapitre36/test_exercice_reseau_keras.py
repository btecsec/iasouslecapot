"""Tests du corrigé du chapitre 36.

Entraîner un réseau dans un test n'est acceptable que parce que celui-ci est
minuscule : 4 entrées, 24 neurones, 273 exemples. Quelques secondes au plus.
"""

import pytest

pytest.importorskip("tensorflow", reason="pip install -r requirements.txt")
pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

import exercice_reseau_keras as ex  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def modele_entraine(donnees_pretes):
    X_train, _, y_train, _, _, _ = donnees_pretes
    modele = ex.compiler(ex.construire())
    historique = ex.entrainer(modele, X_train, y_train, epochs=30)
    return modele, historique


# --------------------------------------------------------------- question 1
def test_les_labels_sont_encodes_en_entiers(donnees_pretes):
    _, _, y_train, _, encodeur, _ = donnees_pretes
    assert set(np.unique(y_train)) <= {0, 1, 2}
    assert list(encodeur.classes_) == ["Adelie", "Chinstrap", "Gentoo"]


def test_les_donnees_sont_normalisees(donnees_pretes):
    X_train = donnees_pretes[0]
    assert X_train.mean() == pytest.approx(0, abs=1e-5)
    assert X_train.std() == pytest.approx(1, abs=1e-2)


def test_le_scaler_nest_ajuste_que_sur_le_train(donnees_pretes):
    """Le test n'est pas centré exactement sur 0 : preuve qu'il n'a pas
    servi à calculer la moyenne."""
    X_test = donnees_pretes[1]
    assert X_test.mean() != pytest.approx(0, abs=1e-9)


# --------------------------------------------------------------- question 2
def test_larchitecture_a_trois_couches_denses():
    assert len(ex.construire().layers) == 3


def test_la_sortie_a_un_neurone_par_classe():
    assert ex.construire(n_classes=3).output_shape[-1] == 3


def test_les_couches_cachees_utilisent_relu():
    couches = ex.construire().layers
    assert couches[0].activation.__name__ == "relu"
    assert couches[1].activation.__name__ == "relu"


def test_la_sortie_utilise_softmax():
    assert ex.construire().layers[-1].activation.__name__ == "softmax"


def test_les_probabilites_de_sortie_somment_a_un(donnees_pretes):
    X_test = donnees_pretes[1]
    sorties = ex.construire().predict(X_test[:5], verbose=0)
    assert np.allclose(sorties.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------- question 3
def test_le_modele_est_compile_avec_adam():
    modele = ex.compiler(ex.construire())
    assert modele.optimizer.name.lower() == "adam"


# --------------------------------------------------------------- question 4
def test_lentrainement_enregistre_chaque_epoque(modele_entraine):
    _, historique = modele_entraine
    assert len(historique.history["loss"]) == 30


def test_la_validation_est_bien_suivie(modele_entraine):
    _, historique = modele_entraine
    assert "val_accuracy" in historique.history


def test_la_perte_diminue(modele_entraine):
    """Le signe minimal que l'apprentissage fonctionne."""
    _, historique = modele_entraine
    pertes = historique.history["loss"]
    assert pertes[-1] < pertes[0]


# --------------------------------------------------------------- question 5
def test_lecart_final_est_mesurable(modele_entraine):
    _, historique = modele_entraine
    assert isinstance(ex.ecart_final(historique), float)


def test_pas_de_surapprentissage_marque_sur_ce_jeu(modele_entraine):
    """Sur un problème aussi séparable, l'écart doit rester faible."""
    _, historique = modele_entraine
    assert ex.ecart_final(historique) < 0.15


def test_les_courbes_tracent_les_deux_series(modele_entraine):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, historique = modele_entraine
    figure = ex.courbes(historique)
    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)


# --------------------------------------------------------------- question 6
def test_le_score_de_test_est_bon(modele_entraine, donnees_pretes):
    modele, _ = modele_entraine
    _, X_test, _, y_test, _, _ = donnees_pretes
    assert ex.evaluer(modele, X_test, y_test)["accuracy"] > 0.90


def test_le_reseau_bat_largement_la_classe_majoritaire(modele_entraine, donnees_pretes):
    modele, _ = modele_entraine
    _, X_test, _, y_test, _, _ = donnees_pretes
    majoritaire = np.bincount(y_test).max() / len(y_test)
    assert ex.evaluer(modele, X_test, y_test)["accuracy"] > majoritaire + 0.30

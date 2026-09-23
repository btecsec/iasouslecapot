"""Tests du corrigé du chapitre 25."""

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow requis : pip install tensorflow")

import exercice_keras as ex  # noqa: E402


def test_la_version_est_lisible():
    assert ex.version().startswith("2.")


def test_le_tenseur_contient_les_bonnes_valeurs():
    assert ex.creer_tenseur().numpy().tolist() == [2, 4, 6]


def test_la_forme_est_un_vecteur_de_trois_elements():
    assert ex.forme(ex.creer_tenseur()) == (3,)


def test_tripler():
    assert ex.tripler(ex.creer_tenseur()).numpy().tolist() == [6, 12, 18]


def test_gpu_disponible_renvoie_un_booleen():
    assert isinstance(ex.gpu_disponible(), bool)


def test_le_reseau_a_deux_couches_denses():
    modele = ex.petit_reseau()
    denses = [c for c in modele.layers if isinstance(c, tf.keras.layers.Dense)]
    assert len(denses) == 2


def test_la_sortie_a_un_neurone_par_classe():
    modele = ex.petit_reseau(n_entrees=4, n_classes=3)
    assert modele.output_shape[-1] == 3


def test_la_sortie_est_une_distribution_de_probabilites():
    """softmax : chaque ligne de sortie doit sommer à 1."""
    numpy = pytest.importorskip("numpy")
    modele = ex.petit_reseau(n_entrees=4, n_classes=3)
    entree = numpy.zeros((5, 4), dtype="float32")
    sorties = modele.predict(entree, verbose=0)
    assert sorties.shape == (5, 3)
    assert numpy.allclose(sorties.sum(axis=1), 1.0, atol=1e-5)


def test_le_reseau_sadapte_au_nombre_de_classes():
    assert ex.petit_reseau(n_entrees=10, n_classes=7).output_shape[-1] == 7


def test_meme_resultat_quen_pytorch_pour_la_vectorisation():
    """Le message du chapitre : seuls les noms changent."""
    numpy = pytest.importorskip("numpy")
    assert ex.tripler(ex.creer_tenseur()).numpy().tolist() == (
        numpy.array([2, 4, 6]) * 3
    ).tolist()

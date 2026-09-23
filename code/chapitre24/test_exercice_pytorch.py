"""Tests du corrigé du chapitre 24."""

import pytest

torch = pytest.importorskip(
    "torch",
    reason="PyTorch requis : pip install torch --index-url https://download.pytorch.org/whl/cpu",
)

import exercice_pytorch as ex  # noqa: E402


def test_la_version_est_lisible():
    assert isinstance(ex.version(), str)
    assert ex.version()[0].isdigit()


def test_le_tenseur_contient_les_bonnes_valeurs():
    assert ex.creer_tenseur().tolist() == [5, 10, 15]


def test_la_forme_est_un_vecteur_de_trois_elements():
    assert ex.forme(ex.creer_tenseur()) == (3,)


def test_le_type_deduit_est_entier():
    # Piège utile : sans dtype explicite, PyTorch garde des entiers.
    assert ex.creer_tenseur().dtype == torch.int64


def test_doubler():
    assert ex.doubler(ex.creer_tenseur()).tolist() == [10, 20, 30]


def test_doubler_ne_modifie_pas_le_tenseur_dorigine():
    t = ex.creer_tenseur()
    ex.doubler(t)
    assert t.tolist() == [5, 10, 15]


def test_gpu_disponible_renvoie_un_booleen():
    # On ne teste pas *la valeur* : elle dépend de la machine.
    assert isinstance(ex.gpu_disponible(), bool)


def test_appareil_est_coherent_avec_la_disponibilite_du_gpu():
    assert ex.appareil() == ("cuda" if ex.gpu_disponible() else "cpu")


def test_le_tenseur_se_deplace_sur_lappareil_choisi():
    t = ex.creer_tenseur().to(ex.appareil())
    assert t.device.type == ex.appareil()


def test_equivalence_avec_numpy():
    """Le message du chapitre : c'est du NumPy avec des super-pouvoirs."""
    numpy = pytest.importorskip("numpy")
    tableau = numpy.array([5, 10, 15])
    assert ex.doubler(ex.creer_tenseur()).tolist() == (tableau * 2).tolist()

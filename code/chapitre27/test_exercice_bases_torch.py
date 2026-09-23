"""Tests du corrigé du chapitre 27."""

import pytest

torch = pytest.importorskip(
    "torch",
    reason="PyTorch requis : pip install torch --index-url https://download.pytorch.org/whl/cpu",
)

import exercice_bases_torch as ex  # noqa: E402


def test_la_suite_va_de_0_a_23():
    t = ex.suite()
    assert tuple(t.shape) == (24,)
    assert t[0].item() == 0 and t[-1].item() == 23


def test_les_deux_reshapes_gardent_les_memes_valeurs():
    t = ex.suite()
    assert tuple(ex.en_cube(t).shape) == (2, 3, 4)
    assert tuple(ex.en_tableau(t).shape) == (6, 4)
    # Démonstration du cours : reshape ne crée ni ne détruit de valeur.
    assert ex.en_cube(t).sum().item() == t.sum().item()
    assert ex.en_tableau(t).flatten().tolist() == t.tolist()


def test_une_forme_impossible_est_refusee():
    # 24 valeurs ne rentrent pas dans une boîte de 5 × 3.
    with pytest.raises(RuntimeError):
        ex.suite().reshape(5, 3)


def test_les_decoupes_prennent_la_bonne_tranche():
    tableau = ex.en_tableau(ex.suite())
    assert ex.premiere_colonne(tableau).tolist() == [0, 4, 8, 12, 16, 20]
    assert ex.deuxieme_rangee(tableau).tolist() == [4, 5, 6, 7]


def test_une_decoupe_est_une_vue_pas_une_copie():
    # Le piège du chapitre : modifier la vue modifie l'original.
    tableau = ex.en_tableau(ex.suite())
    vue = ex.deuxieme_rangee(tableau)
    vue[0] = -1
    assert tableau[1, 0].item() == -1


def test_stack_ajoute_une_dimension_et_cat_en_allonge_une():
    a, b = torch.randn(4, 5), torch.randn(4, 5)
    formes = ex.formes_assemblage(a, b)
    assert formes["stack"] == (2, 4, 5)
    assert formes["cat_dim0"] == (8, 5)
    assert formes["cat_dim1"] == (4, 10)


def test_stack_refuse_des_formes_differentes():
    with pytest.raises(RuntimeError):
        torch.stack([torch.randn(4, 5), torch.randn(4, 6)])


def test_le_lot_a_la_forme_taille_lot_par_contexte():
    donnees = torch.arange(100)
    lot = ex.lot_dexemples(donnees, debuts=[0, 10, 20], taille_contexte=8)
    assert tuple(lot.shape) == (3, 8)
    assert lot[1].tolist() == list(range(10, 18))


def test_pytorch_retrouve_la_derivee_du_cube():
    # 3x² en x = 2 vaut 12 : PyTorch le calcule sans qu'on lui donne la formule.
    assert ex.derivee_du_cube(2.0) == pytest.approx(12.0)
    assert ex.derivee_du_cube(3.0) == pytest.approx(27.0)

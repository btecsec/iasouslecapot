"""Tests du corrigé du chapitre 37."""

import pytest

torch = pytest.importorskip("torch", reason="pip install -r requirements.txt")
pytest.importorskip("sklearn")

import exercice_reseau_pytorch as ex  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def modele_entraine(donnees_pretes):
    torch.manual_seed(ex.GRAINE)
    X_train, _, y_train, _, _ = donnees_pretes
    modele = ex.ReseauManchots()
    historique = ex.entrainer(modele, X_train, y_train, epochs=200)
    return modele, historique


# --------------------------------------------------------------- question 1
def test_les_features_sont_des_flottants(donnees_pretes):
    X_train = donnees_pretes[0]
    assert X_train.dtype == torch.float32


def test_les_labels_sont_des_entiers_longs(donnees_pretes):
    """Le piège du chapitre : CrossEntropyLoss refuse des labels float."""
    y_train = donnees_pretes[2]
    assert y_train.dtype == torch.long


def test_les_trois_classes_sont_presentes(donnees_pretes):
    y_train = donnees_pretes[2]
    assert set(y_train.unique().tolist()) == {0, 1, 2}


def test_les_donnees_sont_normalisees(donnees_pretes):
    X_train = donnees_pretes[0]
    assert float(X_train.mean()) == pytest.approx(0, abs=1e-5)


# --------------------------------------------------------------- question 2
def test_le_reseau_a_le_bon_nombre_de_parametres():
    modele = ex.ReseauManchots()
    assert sum(p.numel() for p in modele.parameters()) == 243


def test_la_sortie_a_une_valeur_par_classe():
    sortie = ex.ReseauManchots()(torch.zeros(5, 4))
    assert sortie.shape == (5, 3)


def test_la_sortie_nest_pas_un_softmax():
    """Ce sont des logits : ils ne somment pas à 1, et c'est voulu."""
    sortie = ex.ReseauManchots()(torch.randn(4, 4))
    assert not torch.allclose(sortie.sum(dim=1), torch.ones(4), atol=1e-3)


def test_deux_couches_relu():
    couches = ex.ReseauManchots().couches
    assert sum(isinstance(c, torch.nn.ReLU) for c in couches) == 2


# --------------------------------------------------------------- question 3
def test_la_perte_et_loptimiseur():
    modele = ex.ReseauManchots()
    critere, optimiseur = ex.outils(modele)
    assert isinstance(critere, torch.nn.CrossEntropyLoss)
    assert isinstance(optimiseur, torch.optim.Adam)


def test_la_perte_initiale_est_proche_de_log_3():
    """Un modèle non entraîné hésite entre 3 classes : -log(1/3) ≈ 1,0986."""
    torch.manual_seed(0)
    modele = ex.ReseauManchots()
    critere, _ = ex.outils(modele)
    perte = critere(modele(torch.randn(64, 4)), torch.randint(0, 3, (64,)))
    assert perte.item() == pytest.approx(1.0986, abs=0.25)


# --------------------------------------------------------------- question 4
def test_la_boucle_enregistre_une_perte_par_epoque(modele_entraine):
    _, historique = modele_entraine
    assert len(historique) == 200


def test_la_perte_diminue_nettement(modele_entraine):
    _, historique = modele_entraine
    assert historique[-1] < historique[0] / 2


def test_sans_zero_grad_les_gradients_saccumulent():
    """Pourquoi `zero_grad` est indispensable — démontré en 6 lignes."""
    torch.manual_seed(0)
    modele = ex.ReseauManchots()
    critere, _ = ex.outils(modele)
    X, y = torch.randn(16, 4), torch.randint(0, 3, (16,))

    critere(modele(X), y).backward()
    premier = modele.couches[0].weight.grad.clone()

    critere(modele(X), y).backward()  # sans zero_grad() entre les deux
    second = modele.couches[0].weight.grad

    assert torch.allclose(second, premier * 2, atol=1e-5)


# --------------------------------------------------------------- question 5
def test_le_score_de_test_est_bon(modele_entraine, donnees_pretes):
    modele, _ = modele_entraine
    _, X_test, _, y_test, _ = donnees_pretes
    assert ex.evaluer(modele, X_test, y_test) > 0.90


def test_levaluation_ne_cree_pas_de_gradients(donnees_pretes):
    _, X_test, _, y_test, _ = donnees_pretes
    modele = ex.ReseauManchots()
    ex.evaluer(modele, X_test, y_test)
    assert all(p.grad is None for p in modele.parameters())


def test_le_modele_revient_en_mode_entrainement(donnees_pretes):
    _, X_test, _, y_test, _ = donnees_pretes
    modele = ex.ReseauManchots()
    ex.evaluer(modele, X_test, y_test)
    assert modele.training is True


def test_lentrainement_est_reproductible(donnees_pretes):
    """Même graine, même parcours : indispensable pour comparer deux essais."""
    X_train, _, y_train, _, _ = donnees_pretes

    def perte_finale():
        torch.manual_seed(123)
        return ex.entrainer(ex.ReseauManchots(), X_train, y_train, epochs=20)[-1]

    assert perte_finale() == pytest.approx(perte_finale())

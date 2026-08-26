"""Tests du corrigé du chapitre 18.

Chaque chiffre annoncé dans le README est vérifié ici. C'est la meilleure
façon de faire un exercice de maths en informatique : si votre calcul à la
main diffère du test, l'un des deux se trompe — et vous saurez lequel.
"""

import pytest

np = pytest.importorskip("numpy")

import exercice_maths as ex  # noqa: E402


# --------------------------------------------------------------- question 1
def test_normalisation_de_la_taille():
    assert ex.normaliser(1.60, 1.50, 2.00) == pytest.approx(0.2)


def test_normalisation_du_poids():
    assert ex.normaliser(62, 50, 110) == pytest.approx(0.2)


def test_le_vecteur_dentree():
    assert ex.vecteur_entree(1.60, 62) == pytest.approx([0.2, 0.2])


def test_la_personne_du_chapitre_redonne_le_vecteur_du_chapitre():
    """Contrôle de non-régression : 1,75 m / 98 kg doit donner [0.5, 0.8]."""
    assert ex.vecteur_entree(1.75, 98) == pytest.approx([0.5, 0.8])


@pytest.mark.parametrize(
    "valeur, attendu",
    [(1.50, 0.0), (2.00, 1.0), (1.75, 0.5)],
)
def test_les_bornes_donnent_zero_et_un(valeur, attendu):
    assert ex.normaliser(valeur, 1.50, 2.00) == pytest.approx(attendu)


# --------------------------------------------------------------- question 2
def test_la_norme_du_vecteur():
    x = ex.vecteur_entree(1.60, 62)
    assert ex.norme(x) == pytest.approx(0.2828, abs=1e-4)


def test_la_nouvelle_personne_est_plus_proche_de_lorigine():
    """Elle activera donc plus faiblement la couche."""
    assert ex.norme(ex.vecteur_entree(1.60, 62)) < ex.norme(ex.vecteur_entree(1.75, 98))


# --------------------------------------------------------------- question 3
def test_les_trois_scores_bruts():
    y = ex.couche(ex.vecteur_entree(1.60, 62))
    assert y == pytest.approx([0.18, -0.02, 0.28])


def test_le_calcul_a_la_main_donne_le_meme_resultat():
    x = ex.vecteur_entree(1.60, 62)
    assert ex.couche_a_la_main(x) == pytest.approx(list(ex.couche(x)))


def test_la_personne_du_chapitre_redonne_les_scores_du_chapitre():
    y = ex.couche(ex.vecteur_entree(1.75, 98))
    assert y == pytest.approx([0.39, -0.20, 0.58])


def test_une_entree_nulle_ne_laisse_que_le_biais():
    """Ce que le biais sert à démontrer."""
    assert ex.couche(np.zeros(2)) == pytest.approx(ex.B)


# --------------------------------------------------------------- question 4
def test_les_sorties_sigmoides():
    s = ex.sigmoide(ex.couche(ex.vecteur_entree(1.60, 62)))
    assert s == pytest.approx([0.5449, 0.4950, 0.5695], abs=1e-4)


def test_la_sigmoide_ne_somme_pas_a_un():
    """Le point clé de la question 5 : ce ne sont pas des probabilités."""
    s = ex.sigmoide(ex.couche(ex.vecteur_entree(1.60, 62)))
    assert s.sum() == pytest.approx(1.6094, abs=1e-4)


def test_la_sigmoide_de_zero_vaut_un_demi():
    assert ex.sigmoide(np.array([0.0])) == pytest.approx([0.5])


def test_la_sigmoide_reste_entre_zero_et_un():
    valeurs = ex.sigmoide(np.array([-100.0, -1.0, 0.0, 1.0, 100.0]))
    assert np.all((valeurs >= 0) & (valeurs <= 1))


# --------------------------------------------------------------- question 5
def test_la_softmax_des_scores_bruts():
    p = ex.softmax(ex.couche(ex.vecteur_entree(1.60, 62)))
    assert p == pytest.approx([0.3420, 0.2800, 0.3780], abs=1e-4)


def test_la_softmax_somme_toujours_a_un():
    for scores in ([0.18, -0.02, 0.28], [10.0, -3.0, 0.5], [0.0, 0.0, 0.0]):
        assert ex.softmax(np.array(scores)).sum() == pytest.approx(1.0)


def test_la_softmax_conserve_lordre_de_la_sigmoide():
    y = ex.couche(ex.vecteur_entree(1.60, 62))
    assert np.argmax(ex.softmax(y)) == np.argmax(ex.sigmoide(y))


def test_la_softmax_ne_deborde_pas_sur_de_grands_scores():
    """Sans l'astuce du max, exp(1000) donnerait inf puis nan."""
    p = ex.softmax(np.array([1000.0, 999.0, 998.0]))
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)


# --------------------------------------------------------------- question 6
def test_la_contribution_du_premier_neurone_a_la_mse():
    s = ex.sigmoide(ex.couche(ex.vecteur_entree(1.60, 62)))
    assert ex.erreur_quadratique(1.0, s[0]) == pytest.approx(0.2071, abs=1e-4)


def test_la_mse_sur_les_trois_neurones():
    s = ex.sigmoide(ex.couche(ex.vecteur_entree(1.60, 62)))
    attendus = np.array([1.0, s[1], s[2]])  # seul le 1er neurone se trompe
    assert ex.mse(attendus, s) == pytest.approx(0.0690, abs=1e-4)


def test_la_mse_punit_plus_fort_une_grosse_erreur_que_deux_moyennes():
    grosse = ex.mse(np.array([1.0, 1.0]), np.array([0.1, 1.0]))
    deux_moyennes = ex.mse(np.array([1.0, 1.0]), np.array([0.55, 0.55]))
    assert grosse > deux_moyennes


def test_la_mse_dun_modele_parfait_vaut_zero():
    assert ex.mse(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


# --------------------------------------------------------- quiz, question 6
def test_cross_entropy_de_dix_pourcent():
    assert ex.cross_entropy(0.10) == pytest.approx(2.3026, abs=1e-4)


def test_cross_entropy_dune_certitude_juste_vaut_zero():
    assert ex.cross_entropy(1.0) == pytest.approx(0.0)


def test_cross_entropy_augmente_quand_la_confiance_baisse():
    assert ex.cross_entropy(0.9) < ex.cross_entropy(0.5) < ex.cross_entropy(0.01)


# ---------------------------------------------------------- produit scalaire
def test_produit_scalaire_de_vecteurs_alignes_est_maximal():
    aligne = ex.produit_scalaire(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    perpendiculaire = ex.produit_scalaire(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    oppose = ex.produit_scalaire(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert oppose < perpendiculaire < aligne
    assert perpendiculaire == 0.0

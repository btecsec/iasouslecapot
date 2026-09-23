"""Tests du corrigé du chapitre 33."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import donnees  # noqa: E402
import exercice_decoupage as ex  # noqa: E402


@pytest.fixture(scope="module")
def xy():
    return ex.preparer(donnees.manchots_synthetiques())


# ---------------------------------------------------------------- préparer
def test_la_cible_nest_jamais_dans_les_features(xy):
    """La fuite la plus grossière — et la plus fréquente."""
    X, _ = xy
    assert donnees.CIBLE not in X.columns


def test_les_lignes_incompletes_sont_ecartees(xy):
    X, y = xy
    assert X.isnull().sum().sum() == 0
    assert len(X) == len(y)


# --------------------------------------------------------------- question 1
def test_le_decoupage_respecte_les_80_20(xy):
    X, y = xy
    X_tr, X_te, _, _ = ex.decouper(X, y, test_size=0.2)
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    assert len(X_tr) + len(X_te) == len(X)


def test_aucun_exemple_nest_dans_les_deux_jeux(xy):
    X, y = xy
    X_tr, X_te, _, _ = ex.decouper(X, y)
    assert set(X_tr.index).isdisjoint(set(X_te.index))


def test_le_decoupage_est_reproductible(xy):
    """C'est tout l'intérêt de random_state : deux appels, deux fois pareil."""
    X, y = xy
    premier = ex.decouper(X, y, graine=42)[1].index.tolist()
    second = ex.decouper(X, y, graine=42)[1].index.tolist()
    assert premier == second


def test_une_autre_graine_donne_un_autre_decoupage(xy):
    X, y = xy
    assert (
        ex.decouper(X, y, graine=42)[1].index.tolist()
        != ex.decouper(X, y, graine=7)[1].index.tolist()
    )


# --------------------------------------------------------------- question 2
def test_les_tailles(xy):
    X, y = xy
    X_tr, X_te, _, _ = ex.decouper(X, y)
    assert ex.tailles(X_tr, X_te) == {"train": len(X_tr), "test": len(X_te)}


# --------------------------------------------------------------- question 3
def test_stratify_conserve_les_proportions(xy):
    X, y = xy
    _, _, _, y_te = ex.decouper(X, y, stratifier=True)
    assert ex.ecart_max(ex.proportions(y), ex.proportions(y_te)) < 0.01


def test_sans_stratify_lecart_est_plus_grand(xy):
    """Le cœur de la question 3 : la démonstration, chiffrée."""
    X, y = xy
    _, _, _, sans = ex.decouper(X, y, stratifier=False)
    _, _, _, avec = ex.decouper(X, y, stratifier=True)
    reference = ex.proportions(y)
    assert ex.ecart_max(reference, ex.proportions(avec)) < ex.ecart_max(
        reference, ex.proportions(sans)
    )


def test_les_trois_especes_sont_presentes_des_deux_cotes(xy):
    X, y = xy
    _, _, y_tr, y_te = ex.decouper(X, y, stratifier=True)
    assert set(y_tr.unique()) == set(y_te.unique()) == set(y.unique())


def test_les_proportions_somment_a_un(xy):
    _, y = xy
    # abs=5e-3 et non 1e-3 : les proportions sont arrondies à 3 décimales,
    # donc leur somme peut valoir 0,999 au lieu de 1,000.
    assert sum(ex.proportions(y).values()) == pytest.approx(1.0, abs=5e-3)


# --------------------------------------------------------------- question 5
def test_la_validation_croisee_rend_cinq_scores(xy):
    X, y = xy
    assert len(ex.validation_croisee(X, y, plis=5)) == 5


def test_les_scores_sont_des_proportions(xy):
    X, y = xy
    scores = ex.validation_croisee(X, y)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_le_modele_apprend_vraiment_quelque_chose(xy):
    """Il doit battre largement la classe majoritaire (~44 %)."""
    X, y = xy
    assert ex.validation_croisee(X, y).mean() > 0.80


def test_la_validation_croisee_est_stable(xy):
    """Un écart-type énorme entre les plis signalerait un dataset instable."""
    X, y = xy
    assert ex.validation_croisee(X, y).std() < 0.15

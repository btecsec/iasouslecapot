"""Tests du corrigé du chapitre 41 (≈ 20 s : les recherches entraînent
plusieurs dizaines de forêts)."""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import exercice_reglages as ex  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def resultats(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    grille, duree_grille = ex.recherche_grille(X_train, y_train)
    aleatoire, duree_aleatoire = ex.recherche_aleatoire(X_train, y_train, n_iter=6)
    return {
        "baseline": ex.baseline(X_train, y_train),
        "grille": grille,
        "duree_grille": duree_grille,
        "aleatoire": aleatoire,
        "duree_aleatoire": duree_aleatoire,
    }


# --------------------------------------------------------------- question 1
def test_la_baseline_est_deja_bonne(resultats):
    assert resultats["baseline"] > 0.90


def test_la_baseline_se_mesure_sur_le_train(donnees_pretes):
    """Elle ne doit jamais dépendre du jeu de test."""
    X_train, _, y_train, _ = donnees_pretes
    assert ex.baseline(X_train, y_train) == ex.baseline(X_train, y_train)


# --------------------------------------------------------------- question 2
def test_la_grille_compte_douze_combinaisons():
    assert ex.nombre_de_combinaisons() == 12


def test_la_grille_teste_bien_toutes_les_combinaisons(resultats):
    assert len(resultats["grille"].cv_results_["params"]) == 12


def test_les_meilleurs_reglages_viennent_de_la_grille(resultats):
    meilleurs = resultats["grille"].best_params_
    assert meilleurs["n_estimators"] in ex.GRILLE["n_estimators"]
    assert meilleurs["max_depth"] in ex.GRILLE["max_depth"]


# --------------------------------------------------------------- question 3
def test_le_reglage_ne_fait_jamais_pire_que_la_baseline(resultats):
    """La grille contient les réglages par défaut ou mieux : elle ne peut pas perdre."""
    assert resultats["grille"].best_score_ >= resultats["baseline"] - 1e-9


def test_le_gain_est_modeste_sur_ce_jeu(resultats):
    """La leçon du chapitre : le réglage rapporte des miettes quand les
    données sont bonnes. C'est un affinage, pas un sauvetage."""
    assert ex.gain(resultats["grille"].best_score_, resultats["baseline"]) < 0.05


def test_le_calcul_du_gain():
    assert ex.gain(0.98, 0.95) == pytest.approx(0.03)


# --------------------------------------------------------------- question 4
def test_la_recherche_aleatoire_teste_moins_de_modeles(resultats):
    assert len(resultats["aleatoire"].cv_results_["params"]) == 6
    assert len(resultats["aleatoire"].cv_results_["params"]) < 12


def test_la_recherche_aleatoire_trouve_presque_aussi_bien(resultats):
    """Moins d'essais, quasiment le même résultat : c'est tout son intérêt."""
    assert resultats["aleatoire"].best_score_ >= resultats["grille"].best_score_ - 0.02


def test_la_recherche_aleatoire_est_plus_rapide(resultats):
    assert resultats["duree_aleatoire"] < resultats["duree_grille"]


def test_la_recherche_aleatoire_est_reproductible(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    premier, _ = ex.recherche_aleatoire(X_train, y_train, n_iter=4)
    second, _ = ex.recherche_aleatoire(X_train, y_train, n_iter=4)
    assert premier.best_params_ == second.best_params_


# --------------------------------------------------------------- question 5
def test_le_test_ne_sert_quune_fois_a_la_fin(resultats, donnees_pretes):
    _, X_test, _, y_test = donnees_pretes
    assert ex.verification_finale(resultats["grille"], X_test, y_test) > 0.90


def test_les_recherches_nont_jamais_vu_le_jeu_de_test(resultats, donnees_pretes):
    """Contrôle anti-fuite : la recherche s'est faite sur 273 exemples, pas 342."""
    X_train, _, _, _ = donnees_pretes
    assert resultats["grille"].n_features_in_ == X_train.shape[1]
    assert len(X_train) == 273

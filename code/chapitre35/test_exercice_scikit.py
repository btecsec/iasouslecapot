"""Tests du corrigé du chapitre 35."""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import exercice_scikit as ex  # noqa: E402
from donnees import CIBLE  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    X, y = ex.charger_et_preparer()
    return ex.decouper(X, y)


# --------------------------------------------------------------- question 1
def test_la_cible_a_ete_retiree_des_features():
    X, _ = ex.charger_et_preparer()
    assert CIBLE not in X.columns


def test_toutes_les_features_sont_numeriques():
    """Un modèle Scikit-learn refuse le texte."""
    import pandas as pd

    X, _ = ex.charger_et_preparer()
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_il_ne_reste_aucune_valeur_manquante():
    X, y = ex.charger_et_preparer()
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_lencodage_a_ajoute_les_colonnes_attendues():
    X, _ = ex.charger_et_preparer()
    assert "island_Biscoe" in X.columns
    assert "sex_Male" in X.columns


# --------------------------------------------------------------- question 2
def test_le_decoupage_est_stratifie(donnees_pretes):
    _, _, y_train, y_test = donnees_pretes
    for espece in y_train.unique():
        assert abs((y_train == espece).mean() - (y_test == espece).mean()) < 0.02


# --------------------------------------------------------------- question 3
def test_le_score_depasse_95_pourcent(donnees_pretes):
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    assert modele.score(X_test_n, y_test) > 0.95


def test_le_scaler_nest_ajuste_que_sur_le_train(donnees_pretes):
    """La vérification anti-fuite : les moyennes viennent du train seul."""
    X_train, X_test, y_train, _ = donnees_pretes
    _, scaler, _ = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    assert scaler.mean_[0] == pytest.approx(X_train.iloc[:, 0].mean())


# --------------------------------------------------------------- question 4
def test_le_rapport_couvre_les_trois_especes(donnees_pretes):
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    detail = ex.rapport(modele, X_test_n, y_test)
    for espece in ("Adelie", "Chinstrap", "Gentoo"):
        assert espece in detail


def test_le_rapport_donne_precision_rappel_et_f1(donnees_pretes):
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    detail = ex.rapport(modele, X_test_n, y_test)
    assert {"precision", "recall", "f1-score"} <= set(detail["Adelie"])


def test_la_classe_la_moins_bien_predite_est_identifiable(donnees_pretes):
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    detail = ex.rapport(modele, X_test_n, y_test)
    assert ex.classe_la_moins_bien_predite(detail) in {"Adelie", "Chinstrap", "Gentoo"}


def test_les_moyennes_ne_sont_pas_prises_pour_des_classes(donnees_pretes):
    """Piège du classification_report : il contient aussi 'macro avg'."""
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    detail = ex.rapport(modele, X_test_n, y_test)
    assert ex.classe_la_moins_bien_predite(detail) != "macro avg"


# --------------------------------------------------------------- question 5
def test_le_pipeline_donne_le_meme_score_que_la_version_manuelle(donnees_pretes):
    """La réponse à « le score change-t-il ? » : non, et c'est le message."""
    X_train, X_test, y_train, y_test = donnees_pretes
    modele, _, X_test_n = ex.entrainer_sans_pipeline(X_train, X_test, y_train)
    pipeline = ex.entrainer_avec_pipeline(X_train, y_train)
    assert pipeline.score(X_test, y_test) == pytest.approx(
        modele.score(X_test_n, y_test)
    )


def test_le_pipeline_predit_a_partir_des_donnees_brutes(donnees_pretes):
    """Son vrai apport : plus besoin de penser à normaliser à la prédiction."""
    X_train, X_test, y_train, _ = donnees_pretes
    pipeline = ex.entrainer_avec_pipeline(X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions) <= {"Adelie", "Chinstrap", "Gentoo"}


def test_le_pipeline_contient_bien_les_deux_etapes(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    pipeline = ex.entrainer_avec_pipeline(X_train, y_train)
    assert list(pipeline.named_steps) == ["standardscaler", "logisticregression"]

"""Tests du corrigé du chapitre 42.

Tout se passe dans `tmp_path` : les tests n'écrivent jamais dans le dépôt.
"""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import exercice_sauvegarde as ex  # noqa: E402


@pytest.fixture(scope="module")
def donnees_pretes():
    return ex.preparer()


@pytest.fixture(scope="module")
def modele_et_scaler(donnees_pretes):
    X_train, _, y_train, _ = donnees_pretes
    scaler = StandardScaler().fit(X_train)
    modele = RandomForestClassifier(n_estimators=50, random_state=ex.GRAINE)
    modele.fit(scaler.transform(X_train), y_train)
    return modele, scaler


# --------------------------------------------------------------- question 1
def test_le_fichier_est_bien_ecrit(tmp_path, modele_et_scaler):
    modele, _ = modele_et_scaler
    chemin = ex.sauver_scikit(modele, tmp_path / "foret.joblib")
    assert chemin.exists() and chemin.stat().st_size > 0


def test_le_modele_recharge_predit_exactement_pareil(tmp_path, modele_et_scaler, donnees_pretes):
    """Le seul critère qui compte : mêmes entrées, mêmes sorties."""
    modele, scaler = modele_et_scaler
    _, X_test, _, _ = donnees_pretes
    chemin = ex.sauver_scikit(modele, tmp_path / "foret.joblib")
    recharge = ex.charger_scikit(chemin)

    X_norme = scaler.transform(X_test)
    assert np.array_equal(modele.predict(X_norme), recharge.predict(X_norme))


def test_les_probabilites_aussi_sont_identiques(tmp_path, modele_et_scaler, donnees_pretes):
    modele, scaler = modele_et_scaler
    _, X_test, _, _ = donnees_pretes
    recharge = ex.charger_scikit(ex.sauver_scikit(modele, tmp_path / "f.joblib"))
    X_norme = scaler.transform(X_test)
    assert np.allclose(modele.predict_proba(X_norme), recharge.predict_proba(X_norme))


# ------------------------------------------------------- le piège du scaler
def test_oublier_le_scaler_change_les_predictions(modele_et_scaler, donnees_pretes):
    """La question 4 du quiz, démontrée : aucune erreur levée, résultats faux."""
    modele, scaler = modele_et_scaler
    _, X_test, _, _ = donnees_pretes

    correct = modele.predict(scaler.transform(X_test))
    sans_scaler = modele.predict(X_test.to_numpy())

    assert (correct != sans_scaler).sum() > 0


# --------------------------------------------------------------- question 4
def test_le_paquet_contient_les_trois_elements(tmp_path, modele_et_scaler, donnees_pretes):
    modele, scaler = modele_et_scaler
    X_train, _, _, _ = donnees_pretes
    chemin = ex.sauver_avec_scaler(modele, scaler, X_train.columns, tmp_path / "p.joblib")
    paquet = ex.charger_scikit(chemin)
    assert set(paquet) == {"modele", "scaler", "colonnes"}


def test_le_paquet_predit_a_partir_des_donnees_brutes(tmp_path, modele_et_scaler, donnees_pretes):
    modele, scaler = modele_et_scaler
    X_train, X_test, _, _ = donnees_pretes
    chemin = ex.sauver_avec_scaler(modele, scaler, X_train.columns, tmp_path / "p.joblib")
    paquet = ex.charger_scikit(chemin)

    attendu = modele.predict(scaler.transform(X_test))
    assert np.array_equal(ex.predire_avec_paquet(paquet, X_test), attendu)


def test_le_paquet_remet_les_colonnes_dans_le_bon_ordre(tmp_path, modele_et_scaler, donnees_pretes):
    """Le troisième oubli classique : des colonnes reçues dans le désordre."""
    modele, scaler = modele_et_scaler
    X_train, X_test, _, _ = donnees_pretes
    chemin = ex.sauver_avec_scaler(modele, scaler, X_train.columns, tmp_path / "p.joblib")
    paquet = ex.charger_scikit(chemin)

    melange = X_test[list(reversed(list(X_test.columns)))]
    assert np.array_equal(
        ex.predire_avec_paquet(paquet, melange), ex.predire_avec_paquet(paquet, X_test)
    )


# -------------------------------------------------------- variante pipeline
def test_le_pipeline_sauvegarde_tout_dun_bloc(tmp_path, donnees_pretes):
    X_train, X_test, y_train, _ = donnees_pretes
    pipeline = make_pipeline(
        StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=ex.GRAINE)
    ).fit(X_train, y_train)

    recharge = ex.charger_scikit(ex.sauver_scikit(pipeline, tmp_path / "pipe.joblib"))
    assert np.array_equal(pipeline.predict(X_test), recharge.predict(X_test))


# --------------------------------------------------------------- question 5
def test_le_paquet_complet_ne_pese_presque_rien_de_plus(tmp_path, modele_et_scaler, donnees_pretes):
    """Le scaler n'ajoute que quelques moyennes : aucune raison de l'oublier."""
    modele, scaler = modele_et_scaler
    X_train, _, _, _ = donnees_pretes
    seul = ex.sauver_scikit(modele, tmp_path / "seul.joblib")
    complet = ex.sauver_avec_scaler(modele, scaler, X_train.columns, tmp_path / "c.joblib")
    assert ex.taille_ko(complet) < ex.taille_ko(seul) * 1.1


# ------------------------------------------------- les autres frameworks
def test_pytorch_state_dict_aller_retour(tmp_path):
    torch = pytest.importorskip("torch")

    reseau = torch.nn.Sequential(torch.nn.Linear(4, 3))
    chemin = tmp_path / "poids.pt"
    torch.save(reseau.state_dict(), chemin)

    recharge = torch.nn.Sequential(torch.nn.Linear(4, 3))
    # weights_only=True : on ne charge que des nombres, jamais du code.
    recharge.load_state_dict(torch.load(chemin, weights_only=True))
    recharge.eval()

    entree = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.allclose(reseau(entree), recharge(entree))


def test_keras_aller_retour(tmp_path):
    keras = pytest.importorskip("tensorflow.keras")
    numpy = pytest.importorskip("numpy")

    modele = keras.Sequential(
        [keras.layers.Input(shape=(4,)), keras.layers.Dense(3, activation="softmax")]
    )
    chemin = tmp_path / "modele.keras"
    modele.save(chemin)

    recharge = keras.models.load_model(chemin)
    entree = numpy.zeros((2, 4), dtype="float32")
    assert numpy.allclose(modele.predict(entree, verbose=0), recharge.predict(entree, verbose=0))

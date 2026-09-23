"""Chapitre 49 — les deux tests demandés par l'exercice, plus le filet.

Question 1 : « une prédiction valide ; une entrée mal formée rejetée. »
Ce sont exactement les deux premiers tests ci-dessous. Les suivants sont ce
qu'on ajoute quand on découvre, en production, ce qui casse vraiment.
"""

import pytest

pytest.importorskip("fastapi", reason="pip install -r requirements.txt")
pytest.importorskip("sklearn")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)

MANCHOT = {
    "longueur_bec": 39.1,
    "profondeur_bec": 18.7,
    "longueur_nageoire": 181.0,
    "masse": 3750.0,
}


# ------------------------------------------------ les deux tests demandés
def test_une_prediction_valide_aboutit():
    reponse = client.post("/predire", json=MANCHOT)
    assert reponse.status_code == 200
    assert reponse.json()["espece"] in {"Adelie", "Chinstrap", "Gentoo"}


def test_une_entree_mal_formee_est_rejetee():
    reponse = client.post("/predire", json=MANCHOT | {"masse": "beaucoup"})
    assert reponse.status_code == 422


# ------------------------------------------------------------ le filet
def test_une_masse_negative_est_refusee():
    """`Field(gt=0)` : la validation attrape l'absurde avant le modèle."""
    assert client.post("/predire", json=MANCHOT | {"masse": -100}).status_code == 422


def test_un_champ_manquant_est_refuse():
    incomplet = {k: v for k, v in MANCHOT.items() if k != "masse"}
    assert client.post("/predire", json=incomplet).status_code == 422


def test_la_sonde_de_vie_repond():
    """C'est cette route que la plateforme d'hébergement interroge."""
    assert client.get("/sante").json() == {"statut": "ok"}


def test_la_reponse_est_serialisable_en_json():
    """Piège classique : `numpy.str_` sort de scikit-learn et casse la
    sérialisation. D'où le `str(...)` dans l'API."""
    espece = client.post("/predire", json=MANCHOT).json()["espece"]
    assert type(espece) is str


def test_lordre_des_champs_envoyes_na_aucune_importance():
    desordre = dict(reversed(list(MANCHOT.items())))
    assert (
        client.post("/predire", json=desordre).json()
        == client.post("/predire", json=MANCHOT).json()
    )


def test_le_modele_est_charge_une_fois_au_demarrage():
    assert api.modele is not None


# ------------------------------------------------- non-régression du modèle
def test_le_modele_ne_regresse_pas_sous_le_seuil():
    """Le test de non-régression du chapitre 9 : la chaîne CI doit **échouer**
    si un réentraînement fait chuter la performance sous le seuil convenu."""
    from sklearn.model_selection import cross_val_score

    from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    scores = cross_val_score(api.modele, df[COLONNES_NUMERIQUES], df[CIBLE], cv=3)
    assert scores.mean() > 0.90, f"performance tombee a {scores.mean():.3f}"

"""Tests de l'API du chapitre 45.

`TestClient` appelle l'application **en mémoire** : aucun serveur à lancer,
aucun port à ouvrir. C'est ce qui rend ces tests utilisables dans une chaîne
CI/CD (chapitre 49).

Le modèle est entraîné une fois au premier import de `mon_api` ; comptez
quelques secondes au démarrage de la session de tests.
"""

import pytest

pytest.importorskip("fastapi", reason="pip install -r requirements.txt")
pytest.importorskip("sklearn")

from fastapi.testclient import TestClient  # noqa: E402

import mon_api  # noqa: E402

client = TestClient(mon_api.app)

MANCHOT_VALIDE = {
    "longueur_bec": 39.1,
    "profondeur_bec": 18.7,
    "longueur_nageoire": 181.0,
    "masse": 3750.0,
}


# ----------------------------------------------------------- les routes
def test_la_route_daccueil_repond():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "message" in reponse.json()


def test_la_route_de_sante_repond_ok():
    """Le défi de l'exercice : /sante, utile pour les sondes du cloud."""
    reponse = client.get("/sante")
    assert reponse.status_code == 200
    assert reponse.json() == {"statut": "ok"}


def test_une_route_inexistante_renvoie_404():
    assert client.get("/inconnue").status_code == 404


# ------------------------------------------------------------ /predire
def test_la_prediction_renvoie_une_espece():
    reponse = client.post("/predire", json=MANCHOT_VALIDE)
    assert reponse.status_code == 200
    assert reponse.json()["espece"] in {"Adelie", "Chinstrap", "Gentoo"}


def test_un_manchot_typique_adelie_est_reconnu():
    """Bec court, nageoires courtes, masse moyenne : profil Adelie."""
    reponse = client.post("/predire", json=MANCHOT_VALIDE)
    assert reponse.json()["espece"] == "Adelie"


def test_un_manchot_typique_gentoo_est_reconnu():
    """Nageoires longues et masse élevée : profil Gentoo."""
    gentoo = {
        "longueur_bec": 47.5,
        "profondeur_bec": 15.0,
        "longueur_nageoire": 217.0,
        "masse": 5100.0,
    }
    assert client.post("/predire", json=gentoo).json()["espece"] == "Gentoo"


def test_la_prediction_est_deterministe():
    """Deux appels identiques doivent donner la même réponse."""
    premier = client.post("/predire", json=MANCHOT_VALIDE).json()
    second = client.post("/predire", json=MANCHOT_VALIDE).json()
    assert premier == second


# ------------------------------------------------- la validation Pydantic
def test_du_texte_a_la_place_dun_nombre_est_rejete():
    """Question 4 de l'exercice : l'API répond 422, avant d'atteindre le modèle."""
    invalide = MANCHOT_VALIDE | {"masse": "beaucoup"}
    reponse = client.post("/predire", json=invalide)
    assert reponse.status_code == 422


def test_un_champ_manquant_est_rejete():
    incomplet = {k: v for k, v in MANCHOT_VALIDE.items() if k != "masse"}
    assert client.post("/predire", json=incomplet).status_code == 422


def test_le_message_derreur_designe_le_champ_fautif():
    """Pydantic ne dit pas seulement « non » : il dit où et pourquoi."""
    reponse = client.post("/predire", json=MANCHOT_VALIDE | {"masse": "beaucoup"})
    detail = reponse.json()["detail"][0]
    assert "masse" in detail["loc"]


def test_un_nombre_ecrit_en_texte_est_converti():
    """Pydantic est tolérant sur ce qui est convertible sans ambiguïté."""
    reponse = client.post("/predire", json=MANCHOT_VALIDE | {"masse": "3750"})
    assert reponse.status_code == 200


def test_un_corps_vide_est_rejete():
    assert client.post("/predire", json={}).status_code == 422


def test_la_route_predire_refuse_le_get():
    """405 : la route existe, mais pas avec ce verbe HTTP."""
    assert client.get("/predire").status_code == 405


# --------------------------------------------------------- la documentation
def test_la_documentation_interactive_existe():
    assert client.get("/docs").status_code == 200


def test_le_schema_openapi_decrit_le_modele_manchot():
    schema = client.get("/openapi.json").json()
    assert "Manchot" in schema["components"]["schemas"]


# ------------------------------------------------ le modèle chargé une fois
def test_le_modele_est_charge_au_demarrage():
    """Question 3 du quiz : il est en mémoire, pas rechargé à chaque requête."""
    assert mon_api.modele is not None
    assert hasattr(mon_api.modele, "predict")

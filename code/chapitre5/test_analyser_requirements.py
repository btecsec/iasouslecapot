"""Tests du lecteur de requirements.txt du chapitre 5."""

import pytest

import analyser_requirements as ar

EXEMPLE = """
# Les bibliothèques de l'exercice
requests==2.32.3
rich>=13.0
pandas          # oups, aucune version
-r autre.txt
"""


def test_nettoyer_retire_commentaires_vides_et_options():
    lignes = ar.nettoyer(EXEMPLE.splitlines())
    assert lignes == ["requests==2.32.3", "rich>=13.0", "pandas"]


def test_ligne_figee():
    dependance = ar.analyser_ligne("requests==2.32.3")
    assert dependance.nom == "requests"
    assert dependance.operateur == "=="
    assert dependance.version == "2.32.3"
    assert dependance.figee is True


def test_ligne_souple_nest_pas_figee():
    dependance = ar.analyser_ligne("rich>=13.0")
    assert dependance.figee is False
    assert dependance.sans_version is False


def test_ligne_sans_version_est_signalee():
    dependance = ar.analyser_ligne("pandas")
    assert dependance.sans_version is True
    assert dependance.figee is False


@pytest.mark.parametrize(
    "ligne, nom",
    [
        ("uvicorn[standard]==0.30.1", "uvicorn"),
        ("scikit-learn>=1.5", "scikit-learn"),
        ("python-dotenv~=1.0", "python-dotenv"),
        ("torch != 2.0.0", "torch"),
    ],
)
def test_variantes_de_syntaxe_acceptees(ligne, nom):
    assert ar.analyser_ligne(ligne).nom == nom


def test_ligne_illisible_renvoie_none():
    assert ar.analyser_ligne("=== ???") is None


def test_analyser_separe_les_illisibles():
    dependances, illisibles = ar.analyser(EXEMPLE + "\n=== ???\n")
    assert [d.nom for d in dependances] == ["requests", "rich", "pandas"]
    assert illisibles == ["=== ???"]


def test_fichier_vide_ne_plante_pas():
    dependances, illisibles = ar.analyser("")
    assert dependances == []
    assert illisibles == []


def test_rapport_alerte_sur_les_dependances_sans_version():
    dependances, illisibles = ar.analyser(EXEMPLE)
    texte = ar.rapport(dependances, illisibles)
    assert "3 dépendances lues" in texte
    assert "pandas" in texte
    assert "Attention" in texte

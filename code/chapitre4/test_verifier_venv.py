"""Tests du détecteur d'environnement virtuel du chapitre 4.

On ne crée pas de vrai venv dans les tests : on injecte les deux chemins,
ce qui rend les tests instantanés et valables sur n'importe quelle machine.
"""

from pathlib import Path

import verifier_venv as vv


def test_prefixes_differents_signifie_venv_actif():
    assert vv.dans_un_venv("/projet/venv", "/usr/local") is True


def test_prefixes_identiques_signifie_python_global():
    assert vv.dans_un_venv("/usr/local", "/usr/local") is False


def test_nom_environnement_extrait_le_dernier_dossier():
    assert vv.nom_environnement("/projet/mon-super-venv") == "mon-super-venv"


def test_chemin_pip_est_dans_le_prefixe_courant():
    chemin = vv.chemin_pip()
    assert isinstance(chemin, Path)
    assert chemin.name in {"Scripts", "bin"}


def test_detection_sans_argument_ne_leve_pas():
    # Doit fonctionner quel que soit l'environnement d'exécution des tests.
    assert isinstance(vv.dans_un_venv(), bool)

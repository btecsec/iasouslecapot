"""Tests du corrigé de l'exercice du chapitre 8.

L'exercice compare quatre écritures d'un même filtrage. La question la plus
importante n'est pas « laquelle est la plus rapide » mais « donnent-elles
toutes exactement le même résultat ? ». C'est précisément ce qu'on teste ici :
une optimisation qui change le résultat n'est pas une optimisation, c'est un
bug.
"""

import sys
from types import GeneratorType

import pytest

import exercice_astuces as ex

VERSIONS = [
    ex.restantes_naif,
    ex.restantes_avec_set,
    ex.restantes_comprehension,
]


@pytest.fixture
def petit_jeu():
    """Un cas minuscule, vérifiable à l'œil nu."""
    commandes = [1, 2, 3, 4, 5]
    deja = [2, 4]
    return commandes, deja


@pytest.mark.parametrize("version", VERSIONS)
def test_chaque_version_filtre_correctement(version, petit_jeu):
    commandes, deja = petit_jeu
    assert version(commandes, deja) == [1, 3, 5]


def test_le_generateur_donne_le_meme_resultat(petit_jeu):
    commandes, deja = petit_jeu
    assert list(ex.restantes_generateur(commandes, deja)) == [1, 3, 5]


def test_le_generateur_est_bien_un_generateur(petit_jeu):
    commandes, deja = petit_jeu
    resultat = ex.restantes_generateur(commandes, deja)
    assert isinstance(resultat, GeneratorType)


def test_le_generateur_occupe_moins_de_memoire_quune_liste():
    """C'est l'étape 4 de l'exercice, transformée en test."""
    commandes, deja = ex.fabriquer_donnees()
    generateur = ex.restantes_generateur(commandes, deja)
    liste = ex.restantes_comprehension(commandes, deja)
    assert sys.getsizeof(generateur) < sys.getsizeof(liste)


def test_les_quatre_versions_saccordent_sur_le_vrai_jeu():
    """Le test qui compte : même entrée d'un million, même sortie."""
    commandes, deja = ex.fabriquer_donnees()
    attendu = ex.restantes_avec_set(commandes, deja)
    assert ex.restantes_comprehension(commandes, deja) == attendu
    assert list(ex.restantes_generateur(commandes, deja)) == attendu


def test_liste_deja_traitee_vide_renvoie_tout(petit_jeu):
    commandes, _ = petit_jeu
    assert ex.restantes_avec_set(commandes, []) == commandes


def test_tout_deja_traite_renvoie_liste_vide(petit_jeu):
    commandes, _ = petit_jeu
    assert ex.restantes_avec_set(commandes, commandes) == []


def test_ordre_preserve():
    """Le passage par un set ne doit pas mélanger les commandes restantes."""
    commandes = [9, 1, 8, 2, 7]
    assert ex.restantes_avec_set(commandes, [1, 2]) == [9, 8, 7]


def test_chrono_renvoie_resultat_et_duree(petit_jeu):
    commandes, deja = petit_jeu
    resultat, duree = ex.chrono(ex.restantes_avec_set, commandes, deja)
    assert resultat == [1, 3, 5]
    assert duree >= 0

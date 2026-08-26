# -*- coding: utf-8 -*-
"""Étapes 1 à 3 de l'exercice : tester `normaliser`.

On teste le cas normal, le cas vide, et le cas limite (toutes les
valeurs identiques) — c'est ce dernier qui fait tomber les débutants.
"""

import pytest

from calculs import moyenne, normaliser


# --------------------------------------------------------------- moyenne
def test_moyenne_cas_normal():
    # Arrange / Act / Assert
    resultat = moyenne([1, 2, 3])
    assert resultat == 2


def test_moyenne_liste_vide_leve_erreur():
    with pytest.raises(ValueError):
        moyenne([])


# ------------------------------------------------------------ normaliser
def test_normaliser_cas_normal():
    assert normaliser([0, 5, 10]) == [0.0, 0.5, 1.0]


def test_normaliser_liste_vide_leve_erreur():
    # On vérifie aussi le MESSAGE : une erreur muette ne sert personne.
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        normaliser([])


def test_normaliser_valeurs_identiques():
    """Le cas limite : étendue nulle, donc division impossible.

    Décision retenue (voir la docstring de `normaliser`) : renvoyer des
    zéros, comme le `MinMaxScaler` de scikit-learn. Ce qui compte n'est
    pas le choix lui-même, mais qu'il soit explicite et testé.
    """
    assert normaliser([7, 7, 7]) == [0.0, 0.0, 0.0]


def test_normaliser_ne_modifie_pas_l_entree():
    """Un piège classique : une fonction qui abîme la liste de l'appelant."""
    original = [0, 5, 10]
    copie = list(original)
    normaliser(original)
    assert original == copie


@pytest.mark.parametrize("valeurs,attendu", [
    ([0, 5, 10], [0.0, 0.5, 1.0]),
    ([10, 5, 0], [1.0, 0.5, 0.0]),        # ordre inversé
    ([-10, 0, 10], [0.0, 0.5, 1.0]),      # valeurs négatives
    ([1, 2], [0.0, 1.0]),                 # deux éléments seulement
    ([42], [0.0]),                        # un seul élément : étendue nulle
])
def test_normaliser_plusieurs_cas(valeurs, attendu):
    """Le même test rejoué sur cinq jeux de données.

    pytest les compte comme cinq tests distincts : si le cas négatif
    casse, le rapport nomme précisément ce cas-là.
    """
    assert normaliser(valeurs) == attendu


# --------------------------------------------------------------- étape 3
@pytest.mark.xfail(reason="échec volontaire : retirez le décorateur xfail")
def test_qui_echoue_volontairement():
    """Étape 3 de l'exercice : lire un rapport d'échec.

    Marqué `xfail` pour que la suite reste verte. Retirez le décorateur
    et lancez `pytest -q` : pytest affiche la ligne fautive, la valeur
    attendue et la valeur obtenue.
    """
    assert normaliser([0, 5, 10]) == [0.0, 0.4, 1.0]

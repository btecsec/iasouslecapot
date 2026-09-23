# -*- coding: utf-8 -*-
"""Fixtures partagées par tous les tests du dossier.

pytest découvre `conftest.py` tout seul : aucun import à écrire dans les
fichiers de test, il suffit de demander la fixture par son nom en
paramètre.
"""

import random

import pytest

# Attention : un `pytest.importorskip("pandas")` au niveau du module ferait
# échouer la collecte de TOUTE la suite si pandas manquait. L'import est
# donc fait à l'intérieur des fixtures qui en ont besoin : seuls les tests
# de données sont alors ignorés.


@pytest.fixture(autouse=True)
def graines_fixes():
    """Fige le hasard avant CHAQUE test (`autouse=True`).

    Un test qui échoue une fois sur cinq est pire que pas de test : on
    finit par l'ignorer, et il finit par cacher un vrai bug.
    """
    random.seed(0)
    # Avec NumPy : np.random.seed(0)
    # Avec PyTorch : torch.manual_seed(0)


@pytest.fixture
def donnees_client():
    """Un petit jeu de données propre, reconstruit à chaque test.

    pytest rappelle la fonction pour chaque test qui la demande : aucun
    test ne peut donc polluer le suivant en modifiant le DataFrame.
    """
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        "age": [25, 40, 33],
        "revenu": [1800.0, 3200.0, 2500.0],
        "label": [0, 1, 1],
    })


@pytest.fixture
def donnees_client_abimees():
    """Le même jeu, avec les défauts que l'on veut voir détectés."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        "age": [25, 240, None],          # âge impossible + valeur manquante
        "revenu": [1800.0, 3200.0, 2500.0],
        "label": [0, 1, 7],              # label hors du domaine {0, 1}
    })

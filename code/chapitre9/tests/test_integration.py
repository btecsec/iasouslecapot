# -*- coding: utf-8 -*-
"""Étape 5 de l'exercice : le test lent, marqué et exclu par défaut.

Ces tests appellent (ou appelleraient) le vrai modèle, la vraie base
vectorielle, la vraie API : lents, payants, parfois instables. On les
marque pour pouvoir les tenir à l'écart du lot quotidien.

    pytest -m "not integration"   # à chaque commit : rapide et gratuit
    pytest -m integration         # une fois par nuit, avec la vraie clé
"""

import os

import pytest

pytestmark = pytest.mark.integration      # marque TOUT le fichier


def test_marqueur_actif():
    """Le test « bidon » de l'énoncé : il sert à vérifier le filtrage.

    Lancez `pytest -m "not integration" -v` : cette ligne doit être
    absente du rapport.
    """
    assert True


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="pas de clé d'API : test ignoré plutôt qu'en échec",
)
def test_pipeline_rag_reel():
    """Le vrai test d'intégration, ignoré tant qu'aucune clé n'est fournie.

    `skipif` plutôt qu'un échec : une clé absente n'est pas un bug, et
    une suite rouge pour une mauvaise raison finit par être ignorée.
    """
    pytest.skip("Branchez ici votre pipeline RAG (voir chapitre 53).")

    # Le squelette à remplir le jour venu :
    #
    #   from rag_pipeline import repondre
    #   reponse = repondre("Quelle est la capitale de la France ?")
    #   assert "paris" in reponse.lower()

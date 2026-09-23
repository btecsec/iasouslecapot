# -*- coding: utf-8 -*-
"""Étape 4 de l'exercice : tester les données, pas seulement le code.

La moitié des « bugs de modèle » sont en réalité des bugs de données.
Ces tests coûtent trois minutes à écrire et font gagner des nuits.
"""

import pytest

pd = pytest.importorskip("pandas")

COLONNES_ATTENDUES = {"age", "revenu", "label"}


def valider(df) -> None:
    """Les règles métier du jeu de données, en un seul endroit.

    Écrire la validation dans une fonction plutôt que dans le test
    permet de la réutiliser en production, au moment de l'ingestion.
    """
    assert COLONNES_ATTENDUES.issubset(df.columns), "colonne manquante"
    assert df["age"].between(0, 120).all(), "âge hors plage"
    assert df["label"].isin([0, 1]).all(), "label non binaire"
    assert df.isnull().sum().sum() == 0, "valeurs manquantes"


def test_schema_donnees_propres(donnees_client):
    valider(donnees_client)


def test_schema_detecte_les_donnees_abimees(donnees_client_abimees):
    """Un test de validation qui ne sait pas échouer ne vaut rien.

    On vérifie donc aussi que le garde-fou se déclenche sur un jeu
    volontairement cassé.
    """
    with pytest.raises(AssertionError):
        valider(donnees_client_abimees)


def test_pas_de_doublon(donnees_client):
    assert not donnees_client.duplicated().any()


def test_types_de_colonnes(donnees_client):
    """Une colonne numérique lue comme du texte casse tout en silence."""
    assert pd.api.types.is_numeric_dtype(donnees_client["age"])
    assert pd.api.types.is_numeric_dtype(donnees_client["revenu"])


# --------------------------------------------------------------- variante
# Sur un vrai projet, on déclare le schéma plutôt que de l'écrire à la
# main : Pandera produit des messages d'erreur bien plus parlants.
#
#   import pandera.pandas as pa
#
#   schema = pa.DataFrameSchema({
#       "age": pa.Column(int, pa.Check.in_range(0, 120)),
#       "revenu": pa.Column(float, pa.Check.ge(0)),
#       "label": pa.Column(int, pa.Check.isin([0, 1])),
#   })
#
#   def test_schema_pandera(donnees_client):
#       schema.validate(donnees_client)

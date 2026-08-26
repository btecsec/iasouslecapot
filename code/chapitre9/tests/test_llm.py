# -*- coding: utf-8 -*-
"""Tester le code qui entoure un LLM, sans jamais appeler le vrai modèle.

Le principe du chapitre : le modèle est non déterministe et payant, mais
tout ce qui l'entoure — parsing, routage, outils — est du Python
ordinaire, testable strictement et gratuitement.
"""

from unittest.mock import MagicMock

import pytest

from agent import LLMClient, executer_agent, extraire_infos, parser_sortie


# ------------------------------------------------- 1. l'interface isolée
def test_le_client_abstrait_refuse_d_etre_utilise_tel_quel():
    """`LLMClient` est un contrat, pas une implémentation."""
    with pytest.raises(NotImplementedError):
        LLMClient().generer("bonjour")


# ------------------------------------------------------- 2. le mock
def test_extraction_apres_reponse_llm():
    faux_client = MagicMock()
    faux_client.generer.return_value = '{"nom": "Ali", "age": 30}'

    resultat = extraire_infos(faux_client, "Voici mon profil...")

    assert resultat == {"nom": "Ali", "age": 30}
    faux_client.generer.assert_called_once()


def test_le_prompt_contient_bien_le_texte_source():
    """On vérifie ce qu'on ENVOIE au modèle, pas seulement ce qu'on reçoit."""
    faux_client = MagicMock()
    faux_client.generer.return_value = "{}"

    extraire_infos(faux_client, "Je m'appelle Ali")

    prompt_envoye = faux_client.generer.call_args.args[0]
    assert "Je m'appelle Ali" in prompt_envoye


# --------------------------------------------- 3. le parsing, tous les cas
@pytest.mark.parametrize("brut,attendu", [
    ('{"score": 8}', {"score": 8}),
    ('Voici le résultat : {"score": 8}', {"score": 8}),
    ('```json\n{"score": 8}\n```', {"score": 8}),   # entouré de balises
    ('{"score": 8', None),                          # tronqué
    ('', None),                                     # réponse vide
    ('Je ne sais pas.', None),                      # pas de JSON du tout
])
def test_parseur_sortie_llm(brut, attendu):
    """Le modèle finira par renvoyer chacun de ces cas. Aucun ne doit planter."""
    assert parser_sortie(brut) == attendu


# --------------------------------------------------- 4. l'agent et ses outils
def test_agent_declenche_le_bon_outil():
    faux_llm = MagicMock()
    faux_llm.generer.return_value = (
        '{"outil": "meteo", "args": {"ville": "Paris"}}'
    )
    outil = MagicMock(return_value="20°C, ensoleillé")

    resultat = executer_agent(
        faux_llm, {"meteo": outil}, "Quel temps à Paris ?")

    outil.assert_called_once_with(ville="Paris")
    assert "20°C" in resultat


def test_agent_refuse_un_outil_inconnu():
    """Le modèle peut inventer un nom d'outil : l'agent ne doit pas planter."""
    faux_llm = MagicMock()
    faux_llm.generer.return_value = '{"outil": "lance_missile", "args": {}}'

    resultat = executer_agent(faux_llm, {"meteo": MagicMock()}, "…")

    assert "inconnu" in resultat.lower()


def test_agent_survit_a_une_reponse_illisible():
    faux_llm = MagicMock()
    faux_llm.generer.return_value = "je crois qu'il faut regarder la météo"

    resultat = executer_agent(faux_llm, {"meteo": MagicMock()}, "…")

    assert "comprendre" in resultat.lower()

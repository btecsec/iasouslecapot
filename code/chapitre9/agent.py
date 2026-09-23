# -*- coding: utf-8 -*-
"""Le code déterministe qui entoure un LLM — c'est LUI qu'on teste.

Rien ici n'appelle un vrai modèle : `LLMClient` est une interface, et
`ClientReel` est le seul endroit qui toucherait l'API. Dans les tests, on
lui substitue un faux client qui répond instantanément et gratuitement.
"""

import json
import re


class LLMClient:
    """Le contrat : tout client de modèle sait générer du texte."""

    def generer(self, prompt: str) -> str:
        raise NotImplementedError


class ClientReel(LLMClient):
    """L'implémentation de production. Jamais appelée dans les tests.

    La clé d'API vit dans une variable d'environnement, jamais en clair
    dans le code (rappel du chapitre 29).
    """

    def generer(self, prompt: str) -> str:      # pragma: no cover
        raise RuntimeError(
            "Branchez ici votre client d'API (openai, anthropic, "
            "mistralai…). Les tests, eux, utilisent un mock."
        )


def parser_sortie(brut: str) -> dict | None:
    """Extrait le premier objet JSON d'une réponse de modèle.

    Renvoie None plutôt que de lever une exception : une réponse
    illisible est un cas de fonctionnement normal pour un LLM, pas un
    bug du programme. Les cas gérés sont listés dans
    `tests/test_llm.py` — JSON pur, JSON précédé de bavardage, JSON dans
    un bloc de code, JSON tronqué, réponse vide.
    """
    if not brut:
        return None

    # Le modèle encadre souvent sa réponse de ```json … ```
    sans_balises = re.sub(r"```(?:json)?|```", "", brut).strip()

    debut = sans_balises.find("{")
    if debut == -1:
        return None

    try:
        return json.loads(sans_balises[debut:])
    except json.JSONDecodeError:
        return None


def extraire_infos(client: LLMClient, texte: str) -> dict | None:
    """Demande au modèle d'extraire un profil, puis relit sa réponse."""
    prompt = (
        "Extrais le nom et l'âge de ce texte, au format JSON "
        'strict {"nom": ..., "age": ...}.\n\n'
        f"Texte : {texte}"
    )
    return parser_sortie(client.generer(prompt))


def executer_agent(client: LLMClient, outils: dict, question: str) -> str:
    """Un agent minimal : le modèle choisit un outil, on l'exécute.

    Trois issues possibles, toutes testées :
    réponse illisible, outil inconnu, exécution réussie.
    """
    prompt = (
        "Choisis un outil parmi " + ", ".join(outils) + " et réponds en "
        'JSON : {"outil": ..., "args": {...}}.\n\n'
        f"Question : {question}"
    )
    decision = parser_sortie(client.generer(prompt))

    if decision is None or "outil" not in decision:
        return "Je n'ai pas réussi à comprendre la demande."

    nom = decision["outil"]
    if nom not in outils:
        return f"Outil inconnu : {nom}."

    return str(outils[nom](**decision.get("args", {})))

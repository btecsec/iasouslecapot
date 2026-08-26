"""Chapitre 29 — LangChain et LangGraph, démontés en Python pur.

Aucune dépendance, aucune clé d'API. L'idée n'est pas de réécrire LangChain
mais de montrer qu'il n'y a pas de magie :

  * une chaîne `prompt | modele | parser` est une composition de fonctions ;
  * l'opérateur `|` est la méthode `__or__` ;
  * un graphe est un dictionnaire de nœuds et une fonction d'aiguillage ;
  * un garde-fou est un compteur, comparé à un maximum.

Usage :
    python mini_chaine.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------- la chaîne
class Etape:
    """Une brique composable : elle sait s'exécuter et se chaîner.

    C'est le `Runnable` de LangChain, réduit à son squelette.
    """

    def __init__(self, fonction: Callable[[Any], Any], nom: str = "étape"):
        self.fonction = fonction
        self.nom = nom

    def invoke(self, entree: Any) -> Any:
        return self.fonction(entree)

    def __or__(self, suivante: "Etape") -> "Etape":
        """prompt | modele : la sortie de l'une devient l'entrée de l'autre."""
        return Etape(
            lambda entree: suivante.invoke(self.invoke(entree)),
            nom=f"{self.nom} | {suivante.nom}",
        )

    def batch(self, entrees: list[Any]) -> list[Any]:
        """500 avis à classer : la même chaîne, appliquée en série."""
        return [self.invoke(entree) for entree in entrees]


@dataclass
class Message:
    """Ce que renvoie un vrai modèle : du texte ET des métadonnées.

    C'est tout l'objet de la question 1 de l'exercice : sans parser, c'est
    cela que vous récupérez, et `print` afficherait l'objet entier.
    """

    content: str
    jetons: int = 0


def gabarit(modele_de_texte: str) -> Etape:
    """Le ChatPromptTemplate : remplit les trous {...} d'un texte."""
    return Etape(lambda variables: modele_de_texte.format(**variables), "prompt")


def modele_factice(reponse: Callable[[str], str] | None = None) -> Etape:
    """Un faux LLM déterministe : il renvoie un Message, comme le vrai."""
    reponse = reponse or (lambda prompt: f"réponse à : {prompt}")

    def appeler(prompt: str) -> Message:
        texte = reponse(prompt)
        return Message(content=texte, jetons=len(prompt.split()) + len(texte.split()))

    return Etape(appeler, "modele")


def parser() -> Etape:
    """Le StrOutputParser : il extrait .content, rien de plus."""
    return Etape(lambda message: message.content, "parser")


# ---------------------------------------------------------------- le graphe
@dataclass
class Etat:
    """L'état transporté d'un nœud à l'autre (question 4 du quiz)."""

    question: str
    documents: list[str] = field(default_factory=list)
    reponse: str | None = None
    tentative: int = 0
    appels_modele: int = 0


MAX_TENTATIVES = 10


def chercher(etat: Etat) -> Etat:
    """Nœud « recherche » : ajoute un document et compte le tour."""
    etat.tentative += 1
    etat.appels_modele += 1
    etat.documents.append(f"document {etat.tentative} sur « {etat.question} »")
    return etat


def evaluer(etat: Etat, satisfait_au_tour: int | None) -> str:
    """Aiguillage : « fin » si la réponse convient, sinon « chercher ».

    Le garde-fou est ici, et nulle part ailleurs : sans le test sur
    MAX_TENTATIVES, un modèle jamais satisfait boucle indéfiniment.
    """
    if satisfait_au_tour is not None and etat.tentative >= satisfait_au_tour:
        etat.reponse = f"trouvé au tour {etat.tentative}"
        return "fin"
    if etat.tentative >= MAX_TENTATIVES:
        etat.reponse = "abandon : trop de tentatives"
        return "fin"
    return "chercher"


def executer_graphe(question: str, satisfait_au_tour: int | None = 3) -> Etat:
    """Fait tourner la boucle chercher → évaluer jusqu'à une sortie.

    `satisfait_au_tour=None` simule le cas de l'exercice : un modèle qui
    n'est *jamais* content. Le graphe s'arrête quand même, grâce au compteur.
    """
    etat = Etat(question=question)
    while True:
        etat = chercher(etat)
        if evaluer(etat, satisfait_au_tour) == "fin":
            return etat


def cout_estime(etat: Etat, euros_par_appel: float = 0.01) -> float:
    """De quoi chiffrer la question 3 : la facture d'une boucle."""
    return round(etat.appels_modele * euros_par_appel, 4)


def main() -> None:
    chaine_sans_parser = gabarit("Résume {texte} pour {public}.") | modele_factice()
    chaine_complete = chaine_sans_parser | parser()

    variables = {"texte": "un long rapport", "public": "un débutant"}

    print("--- sans parser (question 1) ---")
    print(chaine_sans_parser.invoke(variables))
    print("-> il faut aller chercher .content")

    print("\n--- avec parser ---")
    print(chaine_complete.invoke(variables))

    print("\n--- 3 avis classés d'un coup (LangChain, cas c) ---")
    for sortie in chaine_complete.batch([variables] * 3):
        print(" ", sortie)

    print("\n--- le graphe qui aboutit (cas b) ---")
    etat = executer_graphe("où trouver la doc ?", satisfait_au_tour=3)
    print(f"{etat.reponse} — {etat.appels_modele} appels, {cout_estime(etat)} €")

    print("\n--- le graphe jamais satisfait (question 3) ---")
    etat = executer_graphe("question impossible", satisfait_au_tour=None)
    print(f"{etat.reponse} — {etat.appels_modele} appels, {cout_estime(etat)} €")
    print("Sans le garde-fou, ces deux nombres n'auraient pas de limite.")


if __name__ == "__main__":
    main()

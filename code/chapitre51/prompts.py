"""Chapitre 51 — corrigé de l'exercice : construire des prompts, et les tester.

Un prompt est du **texte assemblé par du code**. À ce titre, il se teste comme
n'importe quelle chaîne de caractères — sans clé d'API, sans réseau, sans un
centime dépensé. C'est le principe du chapitre 9 appliqué aux LLM : on teste le
code déterministe qui entoure le modèle.

Usage :
    python prompts.py
    python prompts.py --appel-reel     # si ANTHROPIC_API_KEY est définie
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


# --------------------------------------------------------------- question 1
@dataclass(frozen=True)
class Prompt:
    """Les quatre ingrédients d'une consigne précise.

    C'est la différence entre « parle-moi des tests » et une demande à
    laquelle un humain lui-même saurait répondre.
    """

    sujet: str
    format: str
    longueur: str
    public: str
    role: str | None = None

    def rendre(self) -> str:
        lignes = []
        if self.role:
            lignes.append(self.role)
        lignes.append(f"Rédige {self.format} sur : {self.sujet}.")
        lignes.append(f"Longueur : {self.longueur}.")
        lignes.append(f"Public visé : {self.public}.")
        return "\n".join(lignes)


def prompt_vague(sujet: str) -> str:
    """Ce qu'on écrit spontanément — et qui donne une réponse passe-partout."""
    return f"parle-moi de {sujet}"


# --------------------------------------------------------------- question 3
def prompt_few_shot(exemples: list[tuple[str, str]], entree: str) -> str:
    """Quelques exemples valent mieux qu'une longue explication.

    Le modèle n'apprend rien de durable : il imite le motif présent dans le
    contexte. Fermez la conversation, tout disparaît.
    """
    if not exemples:
        raise ValueError("Le few-shot exige au moins un exemple.")

    lignes = ["Classe chaque avis en positif ou negatif.", ""]
    for texte, etiquette in exemples:
        lignes.append(f"Avis : {texte}")
        lignes.append(f"Classe : {etiquette}")
        lignes.append("")
    lignes.append(f"Avis : {entree}")
    lignes.append("Classe :")
    return "\n".join(lignes)


def extraire_classe(reponse: str, classes=("positif", "negatif")) -> str | None:
    """Le parsing : la partie la plus fragile d'une application à base de LLM.

    Un modèle peut répondre « Positif », « positif.», « Classe : positif » ou
    faire une phrase entière. On normalise avant de comparer, et on renvoie
    None plutôt que de deviner.
    """
    normalisee = reponse.strip().lower().replace("é", "e")
    trouvees = [classe for classe in classes if classe in normalisee]
    return trouvees[0] if len(trouvees) == 1 else None


# --------------------------------------------------------------- question 4
def cle_api_presente(nom: str = "ANTHROPIC_API_KEY") -> bool:
    """La clé se lit dans l'environnement. Jamais en dur dans le fichier :
    un dépôt Git garde l'historique, même après suppression."""
    return bool(os.environ.get(nom))


def appeler_modele(prompt: str, modele: str = "claude-sonnet-5") -> str:
    """L'appel réel — hors des tests, toujours.

    Un test unitaire qui appelle un vrai modèle est lent, payant et non
    reproductible. On l'isole donc derrière un marqueur `integration`
    (chapitre 9) ou, ici, derrière une option en ligne de commande.
    """
    from anthropic import Anthropic

    client = Anthropic()  # lit ANTHROPIC_API_KEY dans l'environnement
    reponse = client.messages.create(
        model=modele,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return reponse.content[0].text


def main() -> None:
    print("--- question 1 : du vague au precis ---")
    print(f"AVANT : {prompt_vague('les tests')}\n")

    precis = Prompt(
        sujet="les tests automatises en Python",
        format="une explication structuree en trois parties",
        longueur="300 mots environ",
        public="un developpeur debutant qui n'a jamais ecrit de test",
    )
    print(f"APRES :\n{precis.rendre()}\n")

    print("--- question 2 : ajouter un role ---")
    severe = Prompt(
        sujet=precis.sujet,
        format=precis.format,
        longueur=precis.longueur,
        public=precis.public,
        role="Tu es un examinateur exigeant : signale ce qui manque.",
    )
    print(f"{severe.rendre()}\n")

    print("--- question 3 : few-shot ---")
    exemples = [
        ("Livraison rapide, produit conforme.", "positif"),
        ("Colis abime et service client injoignable.", "negatif"),
    ]
    print(prompt_few_shot(exemples, "Correct sans plus, mais cher pour ce que c'est."))

    print("\n--- question 4 : appel reel ---")
    if "--appel-reel" in sys.argv and cle_api_presente():
        print(appeler_modele(precis.rendre())[:400])
    else:
        print(
            "Cle absente ou option --appel-reel non fournie : aucun appel.\n"
            "C'est le comportement voulu — les tests ne doivent jamais\n"
            "dependre d'un service payant."
        )


if __name__ == "__main__":
    main()

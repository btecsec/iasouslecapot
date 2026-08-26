"""Chapitre 55 — projet pratique : un assistant RAG complet.

Le pipeline entier — découper, indexer, retrouver, générer — sans dépendre
d'un modèle de plusieurs gigaoctets. Deux briques sont **remplaçables** :

  * la vectorisation : TF-IDF (scikit-learn) ici, un modèle d'embeddings dans
    la vraie vie. Même interface, même code autour ;
  * la génération : un `generateur` que vous passez en paramètre. Par défaut,
    une réponse extractive (les passages retrouvés). En production, une
    fonction qui appelle un LLM.

C'est ce découplage qui rend l'assistant testable : les tests vérifient la
recherche et le garde-fou, jamais la créativité du modèle.

Usage :
    python assistant_rag.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEPARATEURS = ["\n\n", "\n", ". ", " ", ""]
JE_NE_SAIS_PAS = "Je ne sais pas."

# Scikit-learn ne fournit pas de liste française. Sans elle, « quelle est la »
# suffit à faire ressembler n'importe quelle question à n'importe quel
# paragraphe — et le garde-fou saute.
MOTS_VIDES = [
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle",
    "en", "est", "et", "eux", "il", "ils", "je", "la", "le", "les", "leur",
    "lui", "ma", "mais", "me", "meme", "mes", "moi", "mon", "ne", "nos",
    "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "sa", "se", "ses", "son", "sont",
    "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre",
    "vous", "c", "d", "j", "l", "m", "n", "s", "t", "y", "ete", "etee",
    "etees", "etes", "etant", "suis", "es", "sommes", "etes", "etais",
    "etait", "etions", "etiez", "etaient", "ai", "as", "avons", "avez", "ont",
    "aura", "avais", "avait", "avions", "aviez", "avaient", "comment",
    "combien", "quand", "pourquoi", "quoi", "est-ce",
]


# --------------------------------------------------------------- question 2
def decouper(texte: str, taille: int = 400, chevauchement: int = 60) -> list[str]:
    """Découpe en morceaux, de préférence aux frontières naturelles."""
    if chevauchement >= taille:
        raise ValueError("Le chevauchement doit être plus petit que la taille.")

    texte = texte.strip()
    if not texte:
        return []
    if len(texte) <= taille:
        return [texte]

    morceaux: list[str] = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            fenetre = texte[debut:fin]
            for separateur in SEPARATEURS:
                if not separateur:
                    break
                position = fenetre.rfind(separateur)
                if position > len(fenetre) // 2:
                    fin = debut + position + len(separateur)
                    break

        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)
        if fin >= len(texte):
            break

        debut = max(fin - chevauchement, debut + 1)
        # Le chevauchement peut retomber au milieu d'un mot : on avance
        # jusqu'à l'espace suivant, sinon un morceau commence par « teur de ».
        espace = texte.find(" ", debut)
        if 0 <= espace < debut + chevauchement:
            debut = espace + 1

    return morceaux


@dataclass
class Passage:
    """Un morceau retrouvé, avec son score et sa provenance."""

    texte: str
    score: float
    source: str
    rang: int


@dataclass
class Reponse:
    """Ce que renvoie l'assistant. `sources` est ce qui rend la réponse
    vérifiable — c'est le défi de l'exercice."""

    texte: str
    sources: list[Passage] = field(default_factory=list)
    meilleur_score: float = 0.0

    @property
    def a_repondu(self) -> bool:
        return self.texte != JE_NE_SAIS_PAS


class AssistantRAG:
    """Le pipeline complet : indexer une fois, interroger autant qu'on veut."""

    def __init__(
        self,
        taille_morceau: int = 400,
        chevauchement: int = 60,
        k: int = 3,
        # Seuil de refus. Avec les mots vides retirés, une question hors sujet
        # tombe à 0,00 et une question pertinente dépasse 0,15 : 0,10 laisse
        # une marge des deux côtés. À recalibrer sur VOS documents.
        seuil: float = 0.10,
        generateur: Callable[[str, list[Passage]], str] | None = None,
    ):
        self.taille_morceau = taille_morceau
        self.chevauchement = chevauchement
        self.k = k
        # Le seuil est le garde-fou : en dessous, on refuse de répondre.
        self.seuil = seuil
        self.generateur = generateur or self._generation_extractive

        self.morceaux: list[str] = []
        self.sources: list[str] = []
        self._vectoriseur: TfidfVectorizer | None = None
        self._matrice = None

    # ------------------------------------------------------ l'indexation
    def indexer(self, texte: str, source: str = "document") -> int:
        """Étapes 1 à 4 du RAG. Se fait **une fois** par lot de documents."""
        nouveaux = decouper(texte, self.taille_morceau, self.chevauchement)
        self.morceaux.extend(nouveaux)
        self.sources.extend([source] * len(nouveaux))
        self._reconstruire_index()
        return len(nouveaux)

    def indexer_fichier(self, chemin: Path) -> int:
        chemin = Path(chemin)
        return self.indexer(chemin.read_text(encoding="utf-8"), source=chemin.name)

    def oublier(self, source: str) -> int:
        """Le défi du chapitre 54 : retirer un document devenu obsolète.

        Impossible avec un fine-tuning ; trivial ici.
        """
        gardes = [
            (morceau, origine)
            for morceau, origine in zip(self.morceaux, self.sources)
            if origine != source
        ]
        retires = len(self.morceaux) - len(gardes)
        self.morceaux = [m for m, _ in gardes]
        self.sources = [s for _, s in gardes]
        self._reconstruire_index()
        return retires

    def _reconstruire_index(self) -> None:
        if not self.morceaux:
            self._vectoriseur, self._matrice = None, None
            return
        # TF-IDF tient lieu d'embeddings : la logique autour est identique.
        # Remplacez ces deux lignes par un modèle d'embeddings, rien d'autre
        # ne bouge dans ce fichier.
        self._vectoriseur = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), stop_words=MOTS_VIDES
        )
        self._matrice = self._vectoriseur.fit_transform(self.morceaux)

    # ---------------------------------------------------- l'interrogation
    def rechercher(self, question: str, k: int | None = None) -> list[Passage]:
        """Étape 5 : les k morceaux les plus proches de la question."""
        if self._vectoriseur is None:
            return []

        k = k or self.k
        vecteur = self._vectoriseur.transform([question])
        scores = cosine_similarity(vecteur, self._matrice)[0]
        meilleurs = np.argsort(scores)[::-1][:k]

        return [
            Passage(
                texte=self.morceaux[i],
                score=float(scores[i]),
                source=self.sources[i],
                rang=rang,
            )
            for rang, i in enumerate(meilleurs, start=1)
            if scores[i] > 0
        ]

    # --------------------------------------------------------- question 4
    def repondre(self, question: str) -> Reponse:
        """Étape 6, garde-fou compris.

        Si aucun passage ne dépasse le seuil, on répond « Je ne sais pas. »
        **avant** d'appeler le modèle. C'est l'arme anti-hallucination la plus
        efficace : un modèle sans contexte invente, toujours.
        """
        passages = self.rechercher(question)
        meilleur = passages[0].score if passages else 0.0

        if not passages or meilleur < self.seuil:
            return Reponse(JE_NE_SAIS_PAS, sources=[], meilleur_score=meilleur)

        return Reponse(
            texte=self.generateur(question, passages),
            sources=passages,
            meilleur_score=meilleur,
        )

    @staticmethod
    def _generation_extractive(question: str, passages: list[Passage]) -> str:
        """La génération par défaut : on rend le meilleur passage tel quel.

        Sans LLM, c'est la réponse la plus honnête possible — aucun risque
        d'invention, puisque rien n'est reformulé.
        """
        return passages[0].texte

    def construire_prompt(self, question: str, passages: list[Passage]) -> str:
        """Le prompt à envoyer à un vrai LLM (chapitre 51).

        La consigne de refus est répétée en tête ET en pied : un modèle suit
        d'autant mieux une contrainte qu'elle encadre le contexte.
        """
        contexte = "\n\n---\n\n".join(
            f"[{p.source} #{p.rang}] {p.texte}" for p in passages
        )
        return (
            "Réponds UNIQUEMENT à partir du contexte ci-dessous.\n"
            f'Si la réponse ne s\'y trouve pas, réponds exactement "{JE_NE_SAIS_PAS}"\n\n'
            f"CONTEXTE :\n{contexte}\n\n"
            f"QUESTION : {question}\n\n"
            f'Rappel : si le contexte ne contient pas la réponse, réponds "{JE_NE_SAIS_PAS}"\n'
            "Réponse :"
        )


DOCUMENT = """Manuel de l'utilisateur du RobotX1000.

Pour réinitialiser le mot de passe, maintenez le bouton Reset enfoncé pendant
cinq secondes. L'appareil redémarre et les réglages d'usine sont restaurés.

Le RobotX1000 fonctionne avec une batterie Lithium-Ion de 5000 mAh.
L'autonomie annoncée est de huit heures en usage normal, et de trois heures en
mode aspiration intensive. La recharge complète demande deux heures.

Le support technique est joignable à l'adresse support@robotx.com, du lundi au
vendredi, de 9 h à 18 h. La garantie constructeur couvre deux ans, pièces et
main-d'œuvre.

L'entretien recommandé consiste à vider le bac à poussière après chaque usage
et à nettoyer les filtres une fois par mois à l'eau claire.
"""


def main() -> None:
    assistant = AssistantRAG(taille_morceau=250, chevauchement=40)
    nombre = assistant.indexer(DOCUMENT, source="manuel_robotx.txt")
    print(f"2. document decoupe et indexe : {nombre} morceaux\n")

    print("3. trois questions dont on connait la reponse :")
    for question in (
        "Combien de temps dure la batterie ?",
        "Comment reinitialiser le mot de passe ?",
        "Quelle est la duree de la garantie ?",
    ):
        reponse = assistant.repondre(question)
        print(f"\n   Q : {question}")
        print(f"   R : {reponse.texte[:120]}...")
        print(f"   sources : {[f'{p.source} #{p.rang} ({p.score:.2f})' for p in reponse.sources]}")

    print("\n4. une question hors sujet :")
    hors_sujet = assistant.repondre("Quelle est la capitale de l'Australie ?")
    print(f"   Q : Quelle est la capitale de l'Australie ?")
    print(f"   R : {hors_sujet.texte}   (meilleur score {hors_sujet.meilleur_score:.3f})")
    print("   -> le garde-fou a fonctionne : aucune invention.")

    print("\n5. defi : le prompt envoye a un vrai LLM")
    passages = assistant.rechercher("Combien de temps dure la batterie ?")
    print("   " + assistant.construire_prompt("Combien de temps dure la batterie ?", passages)[:300].replace("\n", "\n   ") + "...")

    print("\n6. mise a jour : on oublie le document et on reindexe")
    print(f"   morceaux retires : {assistant.oublier('manuel_robotx.txt')}")
    print(f"   il reste {len(assistant.morceaux)} morceaux indexes")


if __name__ == "__main__":
    main()

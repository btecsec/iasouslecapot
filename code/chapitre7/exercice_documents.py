"""Corrigé exécutable du chapitre 7 — les classes.

On modélise les documents d'un futur moteur de recherche : une classe
abstraite qui pose le contrat, deux implémentations concrètes, et une boucle
qui les traite sans jamais tester leur type.
"""

from abc import ABC, abstractmethod

LONGUEUR_RESUME = 50


class Document(ABC):
    """Le contrat : tout document sait se résumer et dire ce qu'il est."""

    # Attribut de CLASSE : une seule valeur, partagée par toutes les instances
    # et par toutes les sous-classes. Il compte les documents créés.
    compteur = 0

    def __init__(self, titre, contenu):
        self.titre = titre        # attributs d'INSTANCE : propres à chaque objet
        self.contenu = contenu
        # On incrémente sur Document, pas sur self : `self.compteur += 1`
        # créerait un attribut d'instance qui masquerait celui de la classe,
        # et le compteur global resterait à zéro.
        Document.compteur += 1

    @abstractmethod
    def resumer(self):
        """Les 50 premiers caractères, suivis de « … » si le texte est coupé."""

    @abstractmethod
    def type_document(self):
        """Le nom court du type, en minuscules."""

    @classmethod
    def nombre_crees(cls):
        """Combien de documents ont été construits depuis le lancement.

        `cls` reçoit la classe sur laquelle on appelle la méthode ; on lit
        volontairement `Document.compteur` pour renvoyer le total global.
        """
        return cls.compteur


class Article(Document):
    def resumer(self):
        if len(self.contenu) <= LONGUEUR_RESUME:
            return self.contenu
        return self.contenu[:LONGUEUR_RESUME] + "..."

    def type_document(self):
        return "article"


class PageWeb(Document):
    def __init__(self, titre, contenu, url):
        # super() appelle le constructeur du parent : on ne recopie pas ses
        # lignes. Le jour où Document gagne un attribut, PageWeb en hérite
        # sans être modifiée.
        super().__init__(titre, contenu)
        self.url = url

    def resumer(self):
        if len(self.contenu) <= LONGUEUR_RESUME:
            return self.contenu
        return self.contenu[:LONGUEUR_RESUME] + "..."

    def type_document(self):
        return "page web"


def catalogue():
    """Trois documents de types mélangés."""
    return [
        Article(
            "Les Transformers",
            "Le mécanisme d'attention permet au modèle de pondérer "
            "l'importance de chaque mot du contexte.",
        ),
        PageWeb(
            "Documentation PyTorch",
            "torch.nn contient les briques de base des réseaux de neurones.",
            "https://pytorch.org/docs",
        ),
        Article("Note courte", "Trop court pour être coupé."),
    ]


def lignes_catalogue(documents):
    """['1. [article] Titre — résumé', ...]

    Le polymorphisme est ici : la boucle appelle `type_document()` et
    `resumer()` sur chaque objet sans jamais demander de quel type il s'agit.
    Ajoutez une classe `Video(Document)` demain, cette fonction n'a pas à
    changer d'une ligne.
    """
    return [
        f"{numero}. [{doc.type_document()}] {doc.titre} — {doc.resumer()}"
        for numero, doc in enumerate(documents, start=1)
    ]


def main():
    documents = catalogue()

    for ligne in lignes_catalogue(documents):
        print(ligne)

    print(f"\nDocuments créés : {Document.nombre_crees()}")

    print("\nLe contrat est vérifié dès l'instanciation :")
    try:
        Document("Titre", "Contenu")
    except TypeError as erreur:
        print(f"  TypeError -> {erreur}")


if __name__ == "__main__":
    main()

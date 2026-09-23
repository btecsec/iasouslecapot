# -*- coding: utf-8 -*-
"""Chapitre 54 — la base de données vectorielle, pour de vrai.

Au chapitre 53, les vecteurs vivaient dans une liste Python et la recherche
comparait la question à **tous** les morceaux, un par un. Cela marche jusqu'à
quelques milliers de passages, puis cela s'écroule.

Une base vectorielle règle trois problèmes d'un coup :

  * elle **range** les vecteurs sur le disque, pour ne pas tout recalculer à
    chaque démarrage ;
  * elle **cherche vite**, par un index approché (HNSW), au lieu de comparer
    la question à toute la collection ;
  * elle garde des **métadonnées** à côté de chaque vecteur, ce qui permet de
    filtrer avant de chercher (une rubrique, une date, un client).

Le moteur retenu ici est **Chroma** : il s'installe avec un `pip install`, il
n'a pas de serveur à démarrer, et il écrit dans un simple dossier.

Usage :
    python base_vectorielle.py
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Le modèle qui transforme le texte en vecteur. Celui-ci est **multilingue** :
# c'est le détail qui change tout. Le très populaire « all-MiniLM-L6-v2 » n'a
# vu que de l'anglais, et sur un corpus français il retrouve n'importe quoi.
# La variable d'environnement permet de pointer un modèle déjà téléchargé.
MODELE_EMBEDDINGS = os.environ.get(
    "MODELE_EMBEDDINGS",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

DOSSIER_BASE = Path(__file__).parent / "ma_base"
NOM_COLLECTION = "manuel_robotx"

# Au-delà de cette distance, on considère que rien ne correspond. Une distance
# cosinus va de 0 (identique) à 2 (opposé) ; sur ce corpus, une question
# pertinente descend sous 0,75 et une question hors sujet reste au-dessus de
# 0,90. Le seuil est à recalibrer sur VOS documents, jamais recopié.
SEUIL_DISTANCE = 0.85

# Le corpus : le manuel du chapitre 53, déjà découpé, avec ses métadonnées.
MORCEAUX = [
    {
        "id": "manuel-01",
        "texte": "Pour reinitialiser le mot de passe, maintenez le bouton "
                 "Reset enfonce pendant cinq secondes. L'appareil redemarre.",
        "rubrique": "securite",
        "page": 4,
    },
    {
        "id": "manuel-02",
        "texte": "Le RobotX1000 fonctionne avec une batterie Lithium-Ion de "
                 "5000 mAh. L'autonomie annoncee est de huit heures.",
        "rubrique": "batterie",
        "page": 7,
    },
    {
        "id": "manuel-03",
        "texte": "La recharge complete de la batterie demande deux heures sur "
                 "la base de charge fournie.",
        "rubrique": "batterie",
        "page": 8,
    },
    {
        "id": "manuel-04",
        "texte": "Videz le bac a poussiere apres chaque usage et nettoyez les "
                 "filtres une fois par mois a l'eau claire.",
        "rubrique": "entretien",
        "page": 15,
    },
    {
        "id": "manuel-05",
        "texte": "Le support technique est joignable a support@robotx.com, du "
                 "lundi au vendredi, de 9 h a 18 h.",
        "rubrique": "support",
        "page": 21,
    },
    {
        "id": "manuel-06",
        "texte": "La garantie constructeur couvre deux ans, pieces et "
                 "main-d'oeuvre, sur presentation de la facture.",
        "rubrique": "garantie",
        "page": 22,
    },
]


@dataclass
class Resultat:
    """Un voisin retrouvé dans la base, avec sa provenance."""

    identifiant: str
    document: str
    rubrique: str
    page: int
    distance: float

    @property
    def similarite(self) -> float:
        """Distance cosinus -> similarité cosinus, plus parlante à l'oral."""
        return 1.0 - self.distance


def charger_embeddings(modele: str | None = None):
    """La fonction d'embedding : du texte en entrée, un vecteur en sortie.

    Chroma l'appelle tout seul, à l'ajout comme à la recherche. C'est
    important : les documents et les questions **doivent** passer par le même
    modèle, sinon les vecteurs ne vivent pas dans le même espace.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        modele or MODELE_EMBEDDINGS
    )


def ouvrir_collection(
    dossier: Path | str = DOSSIER_BASE,
    nom: str = NOM_COLLECTION,
    embeddings=None,
    remise_a_zero: bool = False,
):
    """Ouvre (ou crée) la collection, stockée dans un dossier du disque.

    `hnsw:space` choisit la distance de l'index. Par défaut Chroma prend la
    distance euclidienne ; pour des embeddings de texte on veut le **cosinus**,
    qui compare la direction des vecteurs et ignore leur longueur.
    """
    dossier = Path(dossier)
    if remise_a_zero and dossier.exists():
        shutil.rmtree(dossier)

    client = chromadb.PersistentClient(path=str(dossier))
    return client.get_or_create_collection(
        name=nom,
        embedding_function=embeddings or charger_embeddings(),
        metadata={"hnsw:space": "cosine"},
    )


def indexer(collection, morceaux=MORCEAUX) -> int:
    """Range les morceaux dans la base. `upsert` remplace au lieu de doubler.

    C'est la différence pratique avec `add` : relancer le script deux fois ne
    crée pas six doublons, il réécrit les six mêmes identifiants.
    """
    collection.upsert(
        ids=[m["id"] for m in morceaux],
        documents=[m["texte"] for m in morceaux],
        metadatas=[
            {"rubrique": m["rubrique"], "page": m["page"]} for m in morceaux
        ],
    )
    return collection.count()


def inspecter(collection, limite: int = 3) -> dict:
    """Ce que contient vraiment la base : identifiants, texte, métadonnées et
    les premiers nombres du vecteur — la partie qu'on ne voit jamais."""
    contenu = collection.get(include=["documents", "metadatas", "embeddings"])
    vecteurs = contenu["embeddings"]
    return {
        "nom": collection.name,
        "nombre": collection.count(),
        "dimension": len(vecteurs[0]) if len(vecteurs) else 0,
        "distance": (collection.metadata or {}).get("hnsw:space", "l2"),
        "lignes": [
            {
                "id": identifiant,
                "document": document,
                "rubrique": metadonnees["rubrique"],
                "page": metadonnees["page"],
                "vecteur": [float(x) for x in vecteur[:limite]],
            }
            for identifiant, document, metadonnees, vecteur in zip(
                contenu["ids"],
                contenu["documents"],
                contenu["metadatas"],
                vecteurs,
            )
        ],
    }


def rechercher(collection, question: str, k: int = 3, rubrique: str | None = None):
    """Les k voisins les plus proches de la question.

    `where` filtre **sur les métadonnées avant la recherche** : c'est le
    superpouvoir qu'une liste Python n'a pas. Chercher « batterie » dans les
    seuls documents publics, ou dans ceux d'un seul client, coûte une ligne.
    """
    reponse = collection.query(
        query_texts=[question],
        n_results=k,
        where={"rubrique": rubrique} if rubrique else None,
    )
    return [
        Resultat(
            identifiant=identifiant,
            document=document,
            rubrique=metadonnees["rubrique"],
            page=metadonnees["page"],
            distance=float(distance),
        )
        for identifiant, document, metadonnees, distance in zip(
            reponse["ids"][0],
            reponse["documents"][0],
            reponse["metadatas"][0],
            reponse["distances"][0],
        )
    ]


def repondre(collection, question: str, k: int = 3):
    """Le garde-fou du chapitre 53, appliqué à la base vectorielle.

    Si le voisin le plus proche reste au-delà du seuil, on rend une liste vide
    plutôt que des passages hors sujet : un LLM nourri de contexte non
    pertinent invente une réponse avec aplomb.
    """
    resultats = rechercher(collection, question, k=k)
    if not resultats or resultats[0].distance > SEUIL_DISTANCE:
        return []
    return [r for r in resultats if r.distance <= SEUIL_DISTANCE]


def court(texte: str, largeur: int) -> str:
    """Coupe un extrait à la largeur voulue, sans casser le dernier mot.

    Les points de suspension sont écrits en trois points ASCII, pas en « … » :
    la console Windows par défaut ne sait pas encoder ce caractère et le
    script planterait sur un `print`.
    """
    if len(texte) <= largeur:
        return texte
    tronque = texte[: largeur - 3]
    espace = tronque.rfind(" ")
    return (tronque[:espace] if espace > largeur // 2 else tronque) + "..."


def main() -> None:
    # Tout l'affichage tient en 62 colonnes : c'est la largeur d'un bloc de
    # code sur une page de livre, et celle d'un terminal étroit.
    collection = ouvrir_collection(remise_a_zero=True)
    modele = MODELE_EMBEDDINGS.replace("\\", "/").rstrip("/").split("/")[-1]
    print(f"1. base ouverte : {DOSSIER_BASE.name}/")
    print(f"   modele : {modele[:44]}")

    nombre = indexer(collection)
    print(f"2. {nombre} morceaux indexes\n")

    vue = inspecter(collection)
    print(f"3. collection \"{vue['nom']}\"")
    print(f"   {vue['nombre']} vecteurs, dimension {vue['dimension']}, "
          f"distance {vue['distance']}\n")
    print("   id         rubrique   page  document (extrait)")
    print("   ---------  ---------  ----  " + "-" * 26)
    for ligne in vue["lignes"]:
        print(f"   {ligne['id']:9}  {ligne['rubrique']:9}  "
              f"{ligne['page']:>4}  {court(ligne['document'], 26)}")

    premier = vue["lignes"][0]
    nombres = "  ".join(f"{x:+.4f}" for x in premier["vecteur"])
    print(f"\n   {premier['id']} vu de l'interieur "
          f"({vue['dimension']} nombres) :")
    print(f"   [ {nombres}  ... ]\n")

    print("4. recherche par le sens (0 = identique)")
    for question in ("j'ai oublie mon code d'acces",
                     "combien de temps tient la batterie ?",
                     "comment nettoyer l'appareil ?"):
        print(f"\n   Q : {question}")
        for rang, r in enumerate(rechercher(collection, question, k=2), 1):
            print(f"     {rang}. {r.identifiant}  {r.distance:.4f}  "
                  f"p.{r.page:<3} {court(r.document, 24)}")

    print("\n5. filtre : rubrique = batterie")
    resultats = rechercher(
        collection, "combien de temps tient la batterie ?", k=2,
        rubrique="batterie",
    )
    for rang, r in enumerate(resultats, start=1):
        print(f"     {rang}. {r.identifiant}  {r.distance:.4f}  "
              f"{court(r.document, 30)}")

    print("\n6. garde-fou : une question hors sujet")
    hors_sujet = "quelle est la capitale de l'Australie ?"
    voisins = rechercher(collection, hors_sujet, k=1)
    print(f"   Q : {hors_sujet}")
    print(f"   plus proche voisin : {voisins[0].distance:.4f} "
          f"> seuil {SEUIL_DISTANCE}")
    print(f"   passages retenus : {len(repondre(collection, hors_sujet))} "
          "-> rien a inventer")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Tests du chapitre 54 — la base vectorielle.

Deux principes, hérités du chapitre 9 :

  * on ne teste **jamais** la qualité sémantique d'un modèle d'embeddings.
    Elle dépend des poids téléchargés, elle change d'une version à l'autre, et
    un test qui échoue pour cette raison n'apprend rien ;
  * on teste tout ce qui **nous** appartient : le rangement, les métadonnées,
    le filtre, le seuil, l'absence de doublons.

D'où la `FonctionEmbeddingFactice` : des vecteurs déterministes, calculés sans
réseau ni gigaoctet de poids. Les tests tournent en une seconde, hors ligne.
"""

from __future__ import annotations

import math

import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb n'est pas installé")

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings  # noqa: E402

import base_vectorielle as bv  # noqa: E402

# Un vocabulaire minuscule : chaque mot est une dimension. Deux textes qui
# partagent des mots donnent des vecteurs proches — c'est un embedding de
# pauvre, mais il suffit à vérifier la mécanique.
VOCABULAIRE = [
    "batterie", "recharge", "autonomie", "heures",
    "mot", "passe", "reset", "bouton",
    "filtres", "poussiere", "nettoyez", "garantie",
    "support", "technique",
]


class FonctionEmbeddingFactice(EmbeddingFunction):
    """Sac de mots normalisé. Déterministe, hors ligne, instantané."""

    def __init__(self) -> None:
        self.vocabulaire = VOCABULAIRE

    def __call__(self, input: Documents) -> Embeddings:
        vecteurs = []
        for texte in input:
            mots = texte.lower().split()
            brut = [
                float(sum(1 for m in mots if terme in m))
                for terme in self.vocabulaire
            ]
            # Une dernière dimension « hors vocabulaire » évite le vecteur nul,
            # que la distance cosinus ne sait pas comparer. Elle ne s'allume
            # que si aucun terme connu n'apparaît : un texte hors sujet devient
            # alors orthogonal à tous les autres, distance 1.
            brut.append(0.0 if any(brut) else 1.0)
            norme = math.sqrt(sum(x * x for x in brut))
            vecteurs.append([x / norme for x in brut])
        return vecteurs

    @staticmethod
    def name() -> str:
        return "factice"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "FonctionEmbeddingFactice":
        return FonctionEmbeddingFactice()


@pytest.fixture
def collection(tmp_path):
    """Une base neuve par test, dans un dossier temporaire."""
    col = bv.ouvrir_collection(
        dossier=tmp_path / "base",
        nom="collection_de_test",
        embeddings=FonctionEmbeddingFactice(),
    )
    bv.indexer(col)
    return col


def test_indexation_range_tous_les_morceaux(collection):
    assert collection.count() == len(bv.MORCEAUX)


def test_upsert_ne_cree_pas_de_doublons(collection):
    """Relancer l'indexation deux fois est le geste le plus courant du
    lecteur. Avec `add`, la base doublerait ; avec `upsert`, elle ne bouge."""
    bv.indexer(collection)
    bv.indexer(collection)
    assert collection.count() == len(bv.MORCEAUX)


def test_inspecter_expose_vecteurs_et_metadonnees(collection):
    vue = bv.inspecter(collection)

    assert vue["nombre"] == len(bv.MORCEAUX)
    assert vue["dimension"] == len(VOCABULAIRE) + 1
    assert vue["distance"] == "cosine"

    ligne = next(l for l in vue["lignes"] if l["id"] == "manuel-02")
    assert ligne["rubrique"] == "batterie"
    assert ligne["page"] == 7
    assert len(ligne["vecteur"]) == 3


def test_recherche_rend_le_bon_nombre_de_voisins(collection):
    resultats = bv.rechercher(collection, "batterie autonomie heures", k=2)

    assert len(resultats) == 2
    # La base rend toujours les voisins du plus proche au plus lointain.
    assert resultats[0].distance <= resultats[1].distance


def test_recherche_retrouve_le_passage_attendu(collection):
    """Avec le vocabulaire factice, « reset bouton » ne peut désigner que le
    morceau sur la réinitialisation : la mécanique de recherche est bien
    branchée, indépendamment de la finesse du vrai modèle."""
    resultats = bv.rechercher(collection, "reset bouton mot passe", k=1)

    assert resultats[0].identifiant == "manuel-01"


def test_filtre_sur_les_metadonnees(collection):
    """`where` restreint la recherche AVANT de comparer les vecteurs."""
    resultats = bv.rechercher(
        collection, "garantie", k=5, rubrique="batterie"
    )

    assert {r.identifiant for r in resultats} == {"manuel-02", "manuel-03"}
    assert all(r.rubrique == "batterie" for r in resultats)


def test_similarite_est_le_complement_de_la_distance(collection):
    resultat = bv.rechercher(collection, "batterie recharge", k=1)[0]

    assert resultat.similarite == pytest.approx(1.0 - resultat.distance)


def test_garde_fou_refuse_les_questions_hors_sujet(collection, monkeypatch):
    """Aucun mot du vocabulaire : tous les vecteurs sont orthogonaux à la
    question, la distance vaut 1, et le seuil doit trancher."""
    monkeypatch.setattr(bv, "SEUIL_DISTANCE", 0.85)

    assert bv.repondre(collection, "la capitale de l'Australie") == []


def test_garde_fou_laisse_passer_une_question_pertinente(collection, monkeypatch):
    monkeypatch.setattr(bv, "SEUIL_DISTANCE", 0.85)
    retenus = bv.repondre(collection, "batterie autonomie heures")

    assert retenus
    assert all(r.distance <= 0.85 for r in retenus)


def test_la_base_survit_a_la_fermeture(tmp_path):
    """L'intérêt d'une base persistante : rouvrir sans réencoder."""
    dossier = tmp_path / "base"
    premiere = bv.ouvrir_collection(
        dossier=dossier, nom="persistance",
        embeddings=FonctionEmbeddingFactice(),
    )
    bv.indexer(premiere)
    del premiere

    seconde = bv.ouvrir_collection(
        dossier=dossier, nom="persistance",
        embeddings=FonctionEmbeddingFactice(),
    )
    assert seconde.count() == len(bv.MORCEAUX)

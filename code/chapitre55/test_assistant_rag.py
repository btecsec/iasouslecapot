"""Tests de l'assistant RAG du chapitre 55.

Aucun modèle de langage n'est appelé : on teste la **tuyauterie** — découpage,
recherche, garde-fou, sources, mise à jour. C'est elle qui décide de la qualité
d'un RAG, et c'est elle qui casse.
"""

import pytest

pytest.importorskip("sklearn")

import assistant_rag as rag  # noqa: E402


@pytest.fixture
def assistant():
    a = rag.AssistantRAG(taille_morceau=250, chevauchement=40)
    a.indexer(rag.DOCUMENT, source="manuel_robotx.txt")
    return a


# ------------------------------------------------------------- découpage
def test_le_document_est_decoupe_en_plusieurs_morceaux(assistant):
    assert len(assistant.morceaux) > 1


def test_aucun_morceau_ne_depasse_la_taille(assistant):
    for morceau in assistant.morceaux:
        assert len(morceau) <= 250


def test_aucun_morceau_ne_commence_au_milieu_dun_mot(assistant):
    """Le chevauchement est réaligné sur un espace : sinon un morceau
    commencerait par « teur de 5000 mAh »."""
    debut_document = rag.DOCUMENT.strip()[:20]
    for morceau in assistant.morceaux:
        if morceau[:20] == debut_document:
            continue
        premier_mot = morceau.split()[0].strip(".,;:!?'\"")
        # Un vrai mot fait au moins deux lettres, ou est un chiffre.
        assert len(premier_mot) >= 2 or premier_mot.isdigit()


def test_indexer_renvoie_le_nombre_de_morceaux_ajoutes():
    a = rag.AssistantRAG()
    assert a.indexer("Un texte assez long. " * 60) == len(a.morceaux)


# -------------------------------------------------------------- recherche
@pytest.mark.parametrize(
    "question, attendu",
    [
        ("Combien de temps dure la batterie ?", "autonomie"),
        ("Comment reinitialiser le mot de passe ?", "Reset"),
        ("Quelle est la duree de la garantie ?", "garantie"),
        ("Comment contacter le support ?", "support@robotx.com"),
    ],
)
def test_la_recherche_trouve_le_bon_passage(assistant, question, attendu):
    passages = assistant.rechercher(question)
    assert passages
    assert any(attendu in p.texte for p in passages)


def test_les_passages_sont_classes_du_meilleur_au_moins_bon(assistant):
    scores = [p.score for p in assistant.rechercher("batterie autonomie")]
    assert scores == sorted(scores, reverse=True)


def test_la_recherche_respecte_le_k_demande(assistant):
    assert len(assistant.rechercher("robot", k=2)) <= 2


def test_la_recherche_sur_un_index_vide_ne_plante_pas():
    assert rag.AssistantRAG().rechercher("quelque chose") == []


def test_chaque_passage_porte_sa_source_et_son_rang(assistant):
    passages = assistant.rechercher("garantie")
    assert passages[0].source == "manuel_robotx.txt"
    assert passages[0].rang == 1


# ------------------------------------------------------------ garde-fou
def test_une_question_hors_sujet_declenche_je_ne_sais_pas(assistant):
    """Question 4 de l'exercice, et arme anti-hallucination n° 1."""
    reponse = assistant.repondre("Quelle est la capitale de l'Australie ?")
    assert reponse.texte == rag.JE_NE_SAIS_PAS
    assert reponse.a_repondu is False


def test_une_question_hors_sujet_ne_cite_aucune_source(assistant):
    assert assistant.repondre("Qui a gagne la coupe du monde 1998 ?").sources == []


def test_une_question_pertinente_obtient_une_reponse(assistant):
    reponse = assistant.repondre("Combien de temps dure la batterie ?")
    assert reponse.a_repondu is True
    assert reponse.meilleur_score >= assistant.seuil


def test_le_garde_fou_agit_avant_la_generation(assistant):
    """Le modèle n'est même pas appelé : c'est ce qui rend le refus fiable."""
    appels = []
    assistant.generateur = lambda question, passages: appels.append(question) or "réponse"
    assistant.repondre("Quelle est la capitale de l'Australie ?")
    assert appels == []


def test_un_seuil_severe_refuse_davantage(assistant):
    question = "Combien de temps dure la batterie ?"
    assert assistant.repondre(question).a_repondu is True
    assistant.seuil = 0.95
    assert assistant.repondre(question).a_repondu is False


# --------------------------------------------------------------- sources
def test_la_reponse_cite_ses_sources(assistant):
    """Le défi de l'exercice : afficher les passages utilisés."""
    reponse = assistant.repondre("Comment contacter le support ?")
    assert reponse.sources
    assert all(p.source == "manuel_robotx.txt" for p in reponse.sources)


def test_la_reponse_extractive_ne_contient_que_du_texte_du_document(assistant):
    """Sans LLM, aucune invention n'est possible : c'est vérifiable."""
    reponse = assistant.repondre("Quelle est la duree de la garantie ?")
    assert reponse.texte in assistant.morceaux


# ------------------------------------------------------------ génération
def test_le_generateur_est_remplacable(assistant):
    """En production, on branche ici l'appel au LLM (chapitre 51)."""
    assistant.generateur = lambda question, passages: f"[{len(passages)} passages]"
    reponse = assistant.repondre("Comment contacter le support ?")
    assert reponse.texte.startswith("[")


def test_le_prompt_contient_la_question_et_le_contexte(assistant):
    passages = assistant.rechercher("garantie")
    prompt = assistant.construire_prompt("Quelle garantie ?", passages)
    assert "Quelle garantie ?" in prompt
    assert passages[0].texte[:30] in prompt


def test_le_prompt_repete_la_consigne_de_refus(assistant):
    """Répétée en tête ET en pied : un modèle suit mieux une contrainte qui
    encadre le contexte."""
    prompt = assistant.construire_prompt("x", assistant.rechercher("garantie"))
    assert prompt.count(rag.JE_NE_SAIS_PAS) == 2


def test_le_prompt_identifie_chaque_passage(assistant):
    prompt = assistant.construire_prompt("x", assistant.rechercher("garantie"))
    assert "[manuel_robotx.txt #1]" in prompt


# ------------------------------------------------------- mise à jour
def test_oublier_retire_les_morceaux_dune_source(assistant):
    """Le défi du chapitre 54 : mettre à jour sans réentraîner."""
    avant = len(assistant.morceaux)
    retires = assistant.oublier("manuel_robotx.txt")
    assert retires == avant
    assert assistant.morceaux == []


def test_oublier_une_source_inconnue_ne_retire_rien(assistant):
    avant = len(assistant.morceaux)
    assert assistant.oublier("autre_document.txt") == 0
    assert len(assistant.morceaux) == avant


def test_apres_oubli_lassistant_ne_repond_plus(assistant):
    assistant.oublier("manuel_robotx.txt")
    assert assistant.repondre("Comment contacter le support ?").texte == rag.JE_NE_SAIS_PAS


def test_on_peut_indexer_plusieurs_documents(assistant):
    assistant.indexer("Le chat de la maison dort sur le canape rouge.", source="notes.txt")
    passages = assistant.rechercher("canape rouge")
    assert passages[0].source == "notes.txt"


def test_la_reindexation_remplace_bien_lancienne_version(assistant):
    assistant.oublier("manuel_robotx.txt")
    assistant.indexer(
        "La garantie constructeur couvre desormais trois ans.", source="manuel_robotx.txt"
    )
    reponse = assistant.repondre("Quelle est la duree de la garantie ?")
    assert "trois ans" in reponse.texte


# ---------------------------------------------------------- chunk_size
def test_chunk_size_change_le_nombre_de_morceaux():
    """Question 2 : quel effet ?"""
    petit = rag.AssistantRAG(taille_morceau=100, chevauchement=20)
    gros = rag.AssistantRAG(taille_morceau=500, chevauchement=20)
    petit.indexer(rag.DOCUMENT)
    gros.indexer(rag.DOCUMENT)
    assert len(petit.morceaux) > len(gros.morceaux)


def test_un_chevauchement_trop_grand_est_refuse():
    with pytest.raises(ValueError):
        rag.decouper("un texte", taille=10, chevauchement=10)

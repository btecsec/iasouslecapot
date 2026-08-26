"""Tests du corrigé du chapitre 7."""

import pytest

import exercice_documents as ex


@pytest.fixture(autouse=True)
def compteur_remis_a_zero():
    """Le compteur est un attribut de CLASSE : il survit d'un test à l'autre.
    On le remet à zéro avant chaque test pour qu'ils restent indépendants."""
    ex.Document.compteur = 0
    yield
    ex.Document.compteur = 0


def test_la_classe_abstraite_refuse_d_etre_instanciee():
    """Le contrat est vérifié à la construction, pas au premier appel."""
    with pytest.raises(TypeError, match="abstract"):
        ex.Document("Titre", "Contenu")


def test_une_sous_classe_incomplete_est_refusee_aussi():
    class DocumentBancal(ex.Document):
        def resumer(self):        # il manque type_document()
            return ""

    with pytest.raises(TypeError, match="abstract"):
        DocumentBancal("Titre", "Contenu")


def test_le_resume_coupe_a_50_caracteres():
    long_texte = "a" * 120
    article = ex.Article("Titre", long_texte)
    resume = article.resumer()
    assert resume == "a" * ex.LONGUEUR_RESUME + "..."
    assert len(resume) == ex.LONGUEUR_RESUME + 3


def test_le_resume_ne_coupe_pas_un_texte_court():
    article = ex.Article("Titre", "Court.")
    assert article.resumer() == "Court."
    assert not article.resumer().endswith("...")


def test_les_types_de_documents():
    assert ex.Article("t", "c").type_document() == "article"
    assert ex.PageWeb("t", "c", "http://x").type_document() == "page web"


def test_super_init_remplit_bien_les_attributs_du_parent():
    page = ex.PageWeb("Titre", "Contenu", "https://exemple.fr")
    assert page.titre == "Titre"        # vient de Document.__init__
    assert page.contenu == "Contenu"    # idem
    assert page.url == "https://exemple.fr"


def test_le_compteur_de_classe_compte_toutes_les_sous_classes():
    ex.Article("a", "x")
    ex.PageWeb("b", "y", "http://z")
    ex.Article("c", "w")
    assert ex.Document.nombre_crees() == 3


def test_la_boucle_polymorphe_ne_teste_aucun_type():
    lignes = ex.lignes_catalogue(ex.catalogue())
    assert len(lignes) == 3
    assert lignes[0].startswith("1. [article] Les Transformers — ")
    assert lignes[1].startswith("2. [page web] Documentation PyTorch — ")
    assert lignes[2] == "3. [article] Note courte — Trop court pour être coupé."


def test_une_nouvelle_classe_s_integre_sans_toucher_a_la_boucle():
    """La démonstration de l'intérêt du polymorphisme."""

    class Video(ex.Document):
        def resumer(self):
            return "(vidéo)"

        def type_document(self):
            return "vidéo"

    lignes = ex.lignes_catalogue([Video("Cours", "…")])
    assert lignes == ["1. [vidéo] Cours — (vidéo)"]

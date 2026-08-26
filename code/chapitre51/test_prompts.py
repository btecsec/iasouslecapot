"""Tests du corrigé du chapitre 51.

Aucun appel de modèle : ces tests coûtent zéro euro et tournent en 50 ms.
Ils vérifient exactement ce qui casse en vrai — l'assemblage du prompt et le
parsing de la réponse.
"""

import pytest

import prompts as p


# --------------------------------------------------------------- question 1
def test_le_prompt_precis_contient_les_quatre_ingredients():
    rendu = p.Prompt(
        sujet="les tests",
        format="un resume en trois points",
        longueur="200 mots",
        public="un debutant",
    ).rendre()
    for morceau in ("les tests", "trois points", "200 mots", "debutant"):
        assert morceau in rendu


def test_le_prompt_precis_est_bien_plus_long_que_le_vague():
    vague = p.prompt_vague("les tests")
    precis = p.Prompt("les tests", "un resume", "200 mots", "un debutant").rendre()
    assert len(precis) > 3 * len(vague)


def test_le_prompt_est_immuable():
    """Un prompt figé est un prompt reproductible : deux appels, même texte."""
    prompt = p.Prompt("a", "b", "c", "d")
    with pytest.raises(Exception):
        prompt.sujet = "autre chose"


# --------------------------------------------------------------- question 2
def test_le_role_apparait_en_tete():
    rendu = p.Prompt("x", "y", "z", "w", role="Tu es un examinateur exigeant.").rendre()
    assert rendu.startswith("Tu es un examinateur exigeant.")


def test_sans_role_le_prompt_commence_par_la_consigne():
    assert p.Prompt("x", "y", "z", "w").rendre().startswith("Rédige")


# --------------------------------------------------------------- question 3
def test_le_few_shot_contient_tous_les_exemples():
    exemples = [("bien", "positif"), ("mauvais", "negatif")]
    rendu = p.prompt_few_shot(exemples, "correct sans plus")
    assert rendu.count("Avis :") == 3  # 2 exemples + l'entrée à classer
    assert rendu.count("Classe :") == 3


def test_le_few_shot_se_termine_par_une_reponse_a_completer():
    """Le prompt doit s'arrêter pile là où le modèle doit écrire."""
    rendu = p.prompt_few_shot([("bien", "positif")], "a tester")
    assert rendu.rstrip().endswith("Classe :")


def test_le_few_shot_place_lentree_en_dernier():
    rendu = p.prompt_few_shot([("bien", "positif")], "ENTREE_A_CLASSER")
    assert rendu.index("ENTREE_A_CLASSER") > rendu.index("bien")


def test_un_few_shot_sans_exemple_est_refuse():
    with pytest.raises(ValueError):
        p.prompt_few_shot([], "quelque chose")


# ---------------------------------------------------------- le parsing
@pytest.mark.parametrize(
    "reponse",
    ["positif", "Positif", "  positif.  ", "Classe : positif", "POSITIF"],
)
def test_le_parsing_supporte_les_variantes(reponse):
    assert p.extraire_classe(reponse) == "positif"


def test_le_parsing_gere_les_accents():
    assert p.extraire_classe("Négatif") == "negatif"


def test_le_parsing_refuse_de_deviner():
    """Deux classes citées : on renvoie None plutôt que d'inventer.

    Un parsing qui devine produit des erreurs silencieuses — le pire type de
    bug dans une application à base de LLM.
    """
    assert p.extraire_classe("c'est entre positif et negatif") is None


def test_le_parsing_renvoie_none_sur_une_reponse_hors_sujet():
    assert p.extraire_classe("je ne sais pas") is None


def test_le_parsing_dune_phrase_complete():
    assert p.extraire_classe("Cet avis me semble plutot positif.") == "positif"


# --------------------------------------------------------------- question 4
def test_la_cle_api_est_lue_dans_lenvironnement(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert p.cle_api_presente() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert p.cle_api_presente() is True


def test_aucune_cle_nest_ecrite_dans_le_fichier():
    """Contrôle de sécurité : un secret dans le code finit dans Git,
    et l'historique le garde même après suppression."""
    from pathlib import Path

    source = Path(p.__file__).read_text(encoding="utf-8")
    for motif in ("sk-ant-", "sk-proj-", "api_key ="):
        assert motif not in source


# ------------------------------------------ l'appel réel, isolé et mocké
def test_lappel_reel_est_teste_avec_un_mock(monkeypatch):
    """On vérifie le code qui entoure l'appel, jamais le modèle lui-même."""

    class FausseReponse:
        content = [type("Bloc", (), {"text": "positif"})()]

    class FauxClient:
        def __init__(self, *a, **k):
            self.messages = self

        def create(self, **kwargs):
            assert kwargs["model"]
            assert kwargs["messages"][0]["content"]
            return FausseReponse()

    faux_module = type("anthropic", (), {"Anthropic": FauxClient})
    monkeypatch.setitem(__import__("sys").modules, "anthropic", faux_module)

    assert p.appeler_modele("classe cet avis") == "positif"

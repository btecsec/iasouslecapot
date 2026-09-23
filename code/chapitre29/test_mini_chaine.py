"""Tests du chapitre 29 : la chaîne, le parser et le garde-fou.

Ces tests ne coûtent rien et ne dépendent d'aucun réseau — c'est exactement la
leçon du chapitre 9 appliquée à une application à base de LLM : on teste le
code déterministe qui entoure le modèle, pas le modèle.
"""

import pytest

import mini_chaine as mc


# ------------------------------------------------------------ la chaîne
def test_le_gabarit_remplit_les_trous():
    prompt = mc.gabarit("Résume {texte} pour {public}.")
    assert prompt.invoke({"texte": "un rapport", "public": "un débutant"}) == (
        "Résume un rapport pour un débutant."
    )


def test_un_trou_manquant_leve_une_erreur():
    prompt = mc.gabarit("Résume {texte} pour {public}.")
    with pytest.raises(KeyError):
        prompt.invoke({"texte": "un rapport"})


def test_sans_parser_on_obtient_un_message_pas_du_texte():
    """Question 1 de l'exercice."""
    chaine = mc.gabarit("Bonjour {nom}") | mc.modele_factice()
    resultat = chaine.invoke({"nom": "Ada"})
    assert isinstance(resultat, mc.Message)
    assert not isinstance(resultat, str)
    assert resultat.content.endswith("Bonjour Ada")


def test_avec_parser_on_obtient_une_chaine_de_caracteres():
    chaine = mc.gabarit("Bonjour {nom}") | mc.modele_factice() | mc.parser()
    assert isinstance(chaine.invoke({"nom": "Ada"}), str)


def test_le_parser_extrait_exactement_le_contenu():
    message = mc.Message(content="texte utile", jetons=42)
    assert mc.parser().invoke(message) == "texte utile"


def test_la_composition_est_associative():
    """(a | b) | c doit donner le même résultat que a | (b | c)."""
    prompt, modele, parser = mc.gabarit("{x}"), mc.modele_factice(), mc.parser()
    gauche = (prompt | modele) | parser
    droite = prompt | (modele | parser)
    assert gauche.invoke({"x": "test"}) == droite.invoke({"x": "test"})


def test_batch_traite_tous_les_elements():
    """Cas (c) : 500 avis, la même chaîne appliquée à chacun."""
    chaine = mc.gabarit("avis : {texte}") | mc.modele_factice() | mc.parser()
    sorties = chaine.batch([{"texte": f"avis {i}"} for i in range(5)])
    assert len(sorties) == 5
    assert all("avis" in sortie for sortie in sorties)


def test_le_modele_factice_est_deterministe():
    chaine = mc.gabarit("{x}") | mc.modele_factice() | mc.parser()
    assert chaine.invoke({"x": "a"}) == chaine.invoke({"x": "a"})


# ------------------------------------------------------------- le graphe
def test_le_graphe_sarrete_quand_il_a_trouve():
    etat = mc.executer_graphe("une question", satisfait_au_tour=3)
    assert etat.tentative == 3
    assert etat.reponse == "trouvé au tour 3"


def test_le_graphe_accumule_ses_documents():
    etat = mc.executer_graphe("une question", satisfait_au_tour=3)
    assert len(etat.documents) == 3


def test_le_garde_fou_borne_un_modele_jamais_satisfait():
    """Question 3 : sans ce test, la boucle serait infinie."""
    etat = mc.executer_graphe("question impossible", satisfait_au_tour=None)
    assert etat.tentative == mc.MAX_TENTATIVES
    assert etat.reponse == "abandon : trop de tentatives"


def test_le_nombre_dappels_reste_borne():
    etat = mc.executer_graphe("question impossible", satisfait_au_tour=None)
    assert etat.appels_modele <= mc.MAX_TENTATIVES


def test_le_cout_du_pire_cas_est_previsible():
    """Le garde-fou transforme une facture inconnue en facture plafonnée."""
    etat = mc.executer_graphe("question impossible", satisfait_au_tour=None)
    assert mc.cout_estime(etat, euros_par_appel=0.01) == pytest.approx(0.10)


@pytest.mark.parametrize("tour", [1, 2, 5, 9])
def test_le_graphe_sarrete_au_tour_demande(tour):
    assert mc.executer_graphe("q", satisfait_au_tour=tour).tentative == tour


def test_letat_transporte_bien_la_question():
    etat = mc.executer_graphe("ma question", satisfait_au_tour=1)
    assert etat.question == "ma question"
    assert "ma question" in etat.documents[0]

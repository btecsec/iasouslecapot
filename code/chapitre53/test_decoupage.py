"""Tests du découpage du chapitre 53 (aucune dépendance, 20 ms)."""

import pytest

import decoupage as d

TEXTE = (
    "Le manuel de l'utilisateur du RobotX1000.\n\n"
    "Pour reinitialiser le mot de passe, maintenez le bouton Reset "
    "pendant cinq secondes. Cette operation efface les reglages.\n\n"
    "Le RobotX1000 fonctionne avec une batterie Lithium-Ion de 5000mAh. "
    "L'autonomie annoncee est de huit heures en usage normal.\n\n"
    "Le support technique est joignable a l'adresse support@robotx.com."
)


# ------------------------------------------------------------ le découpage
def test_un_texte_court_reste_entier():
    assert d.decouper("Trois mots seulement.", taille=500) == ["Trois mots seulement."]


def test_un_texte_vide_ne_donne_aucun_morceau():
    assert d.decouper("") == []
    assert d.decouper("      ") == []


def test_un_texte_long_est_bien_decoupe():
    morceaux = d.decouper(TEXTE, taille=100, chevauchement=20)
    assert len(morceaux) > 1


def test_aucun_morceau_ne_depasse_la_taille_demandee():
    for morceau in d.decouper(TEXTE, taille=120, chevauchement=20):
        assert len(morceau) <= 120


def test_le_texte_entier_est_couvert():
    """Rien ne doit disparaître au découpage : la réponse pourrait s'y trouver."""
    morceaux = d.decouper(TEXTE, taille=100, chevauchement=20)
    reconstitue = " ".join(morceaux)
    for mot in ("RobotX1000", "Reset", "5000mAh", "support@robotx.com"):
        assert mot in reconstitue


def test_la_coupure_evite_de_casser_les_mots():
    """On coupe aux frontières naturelles : phrase, puis mot."""
    morceaux = d.decouper(TEXTE, taille=150, chevauchement=0)
    for morceau in morceaux[:-1]:
        assert not morceau.endswith(("Robot", "batteri", "techniq"))


def test_le_chevauchement_repete_du_contenu():
    """C'est son but : une phrase coupée en deux reste lisible quelque part."""
    sans = d.decouper(TEXTE, taille=100, chevauchement=0)
    avec = d.decouper(TEXTE, taille=100, chevauchement=50)
    assert len(avec) > len(sans)


def test_un_chevauchement_trop_grand_est_refuse():
    """Sans ce garde-fou, la boucle n'avancerait plus."""
    with pytest.raises(ValueError):
        d.decouper(TEXTE, taille=100, chevauchement=100)


def test_le_decoupage_termine_toujours():
    """Un texte sans espace ni ponctuation : le pire cas pour un découpeur."""
    morceaux = d.decouper("a" * 1000, taille=100, chevauchement=10)
    assert len(morceaux) > 5
    assert sum(len(m) for m in morceaux) >= 1000


@pytest.mark.parametrize("taille", [50, 100, 300, 1000])
def test_toutes_les_tailles_fonctionnent(taille):
    morceaux = d.decouper(TEXTE, taille=taille, chevauchement=taille // 10)
    assert morceaux
    assert all(m.strip() for m in morceaux)


# --------------------------------------------------------------- question 2
def test_plus_les_morceaux_sont_petits_plus_ils_sont_nombreux():
    petits = d.statistiques(d.decouper(TEXTE, taille=80, chevauchement=10))
    gros = d.statistiques(d.decouper(TEXTE, taille=300, chevauchement=30))
    assert petits["nombre"] > gros["nombre"]
    assert petits["taille_moyenne"] < gros["taille_moyenne"]


def test_les_statistiques_dun_texte_vide_ne_plantent_pas():
    assert d.statistiques([])["nombre"] == 0


# --------------------------------------------------------------- question 3
def test_deux_phrases_de_sens_proche_nont_aucun_mot_commun():
    """La démonstration de l'exercice : la recherche par mots-clés échoue là
    où un embedding réussit."""
    a = "Quelle est la duree de vie de la batterie ?"
    b = "Combien de temps tient l'accumulateur avant recharge ?"
    assert d.recouvrement_mots(a, b) == 0


def test_deux_phrases_identiques_partagent_tous_leurs_mots():
    phrase = "La batterie du robot est lithium-ion."
    assert d.recouvrement_mots(phrase, phrase) == len(d.mots_cles(phrase))


def test_les_mots_vides_sont_ignores():
    assert d.mots_cles("le la les de du et ou") == set()


def test_les_mots_cles_sont_normalises():
    assert d.mots_cles("Batterie, BATTERIE ; batterie.") == {"batterie"}

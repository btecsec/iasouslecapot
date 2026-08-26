"""Tests du corrigé du chapitre 6."""

import copy

import exercice_commandes as ex


def test_les_lignes_sont_numerotees_a_partir_de_1():
    lignes = ex.lignes_numerotees(ex.COMMANDES)
    assert len(lignes) == 4
    assert lignes[0] == "1. Alice a commandé un clavier (45 €)"
    assert lignes[-1].startswith("4. ")


def test_les_clients_distincts_dedoublonnent_alice():
    # Alice passe deux commandes : elle ne doit apparaître qu'une fois.
    assert ex.clients_distincts(ex.COMMANDES) == {"Alice", "Bob", "Chris"}


def test_le_total_par_client_additionne_les_commandes():
    assert ex.total_par_client(ex.COMMANDES) == {
        "Alice": 225,   # 45 + 180
        "Bob": 25,
        "Chris": 25,
    }


def test_le_total_fonctionne_sur_une_liste_vide():
    # C'est tout l'intérêt de .get(client, 0) : aucun cas particulier à écrire.
    assert ex.total_par_client([]) == {}


def test_la_commande_la_plus_chere_est_l_ecran_d_alice():
    chere = ex.commande_la_plus_chere(ex.COMMANDES)
    assert chere["produit"] == "écran"
    assert chere["client"] == "Alice"
    assert chere["prix"] == 180


def test_les_produits_d_un_client():
    assert ex.produits_de(ex.COMMANDES, "Alice") == {"clavier", "écran"}
    assert ex.produits_de(ex.COMMANDES, "Bob") == {"souris"}
    assert ex.produits_de(ex.COMMANDES, "Inconnu") == set()


def test_a_commande_repond_oui_et_non():
    assert ex.a_commande(ex.COMMANDES, "Alice", "écran") is True
    assert ex.a_commande(ex.COMMANDES, "Alice", "souris") is False


def test_l_augmentation_ne_touche_pas_la_liste_d_origine():
    """Le test qui démontre le piège : sans copie des dictionnaires, cette
    assertion échoue, car les deux listes partageraient les mêmes objets."""
    original = copy.deepcopy(ex.COMMANDES)
    augmentees = ex.prix_augmentes(ex.COMMANDES, 10)

    assert augmentees[0]["prix"] == 49.5      # 45 * 1.10
    assert augmentees[1]["prix"] == 27.5      # 25 * 1.10
    assert ex.COMMANDES == original           # l'originale est intacte
    assert augmentees[0] is not ex.COMMANDES[0]

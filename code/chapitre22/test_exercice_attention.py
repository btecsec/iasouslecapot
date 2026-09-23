"""Tests du corrigé du chapitre 22."""

import pytest

np = pytest.importorskip("numpy")

import exercice_attention as ex  # noqa: E402


# --------------------------------------------------------------- question 1
def test_les_trois_produits_scalaires_de_dort():
    scores = ex.produits_scalaires("dort")
    assert scores["Le"] == pytest.approx(0.20)
    assert scores["chat"] == pytest.approx(0.52)
    assert scores["dort"] == pytest.approx(0.80)


@pytest.mark.parametrize("mot", ["chat", "dort"])
def test_un_mot_se_reconnait_lui_meme(mot):
    scores = ex.produits_scalaires(mot)
    assert scores[mot] == max(scores.values())


def test_le_determinant_est_lexception_qui_instruit():
    """« Le » = [0.1, 0.2] est un vecteur très court : son produit scalaire
    avec lui-même (0.05) est plus petit que celui avec « dort » (0.20).

    Le produit scalaire mélange direction ET longueur : ce n'est pas une
    mesure de similarité pure. C'est précisément pourquoi un vrai Transformer
    projette les embeddings en Q, K et V appris, au lieu de les utiliser tels
    quels comme on le fait ici pour la démonstration.
    """
    scores = ex.produits_scalaires("Le")
    assert scores["Le"] == pytest.approx(0.05)
    assert scores["dort"] == pytest.approx(0.20)
    assert scores["Le"] < scores["dort"]


def test_les_produits_scalaires_de_chat_redonnent_le_chapitre():
    """Contrôle de non-régression sur l'exemple du livre."""
    scores = ex.produits_scalaires("chat")
    assert scores["Le"] == pytest.approx(0.13)
    assert scores["chat"] == pytest.approx(0.58)
    assert scores["dort"] == pytest.approx(0.52)


# --------------------------------------------------------------- question 2
def test_la_mise_a_lechelle_donne_les_chiffres_du_corrige():
    """Avec √2 exact (1.41421…), et non l'arrondi 1.41 de l'énoncé."""
    echelle = ex.mise_a_lechelle(ex.produits_scalaires("dort"))
    assert echelle["Le"] == pytest.approx(0.141, abs=1e-3)
    assert echelle["chat"] == pytest.approx(0.368, abs=1e-3)
    assert echelle["dort"] == pytest.approx(0.566, abs=1e-3)


def test_larrondi_de_la_racine_change_le_troisieme_chiffre():
    """1.41 au lieu de 1.41421… : l'écart se voit dès la 3e décimale.
    Sans conséquence sur les poids finaux, mais bon à savoir avant de
    s'inquiéter d'un écart avec le livre."""
    exact = ex.mise_a_lechelle(ex.produits_scalaires("dort"))
    arrondi = ex.mise_a_lechelle(ex.produits_scalaires("dort"), racine_dk=1.41)
    assert arrondi["dort"] == pytest.approx(0.567, abs=1e-3)
    assert arrondi["dort"] > exact["dort"]


def test_la_mise_a_lechelle_ne_change_pas_lordre():
    bruts = ex.produits_scalaires("dort")
    echelle = ex.mise_a_lechelle(bruts)
    assert sorted(bruts, key=bruts.get) == sorted(echelle, key=echelle.get)


# --------------------------------------------------------------- question 3
def test_les_poids_dattention_de_dort():
    poids = ex.poids_attention("dort")
    assert poids["Le"] * 100 == pytest.approx(26.4, abs=0.15)
    assert poids["chat"] * 100 == pytest.approx(33.2, abs=0.15)
    assert poids["dort"] * 100 == pytest.approx(40.4, abs=0.15)


def test_les_poids_somment_a_cent_pour_cent():
    for mot in ex.MOTS:
        assert sum(ex.poids_attention(mot).values()) == pytest.approx(1.0)


def test_les_poids_de_chat_redonnent_le_chapitre():
    poids = ex.poids_attention("chat")
    assert poids["Le"] * 100 == pytest.approx(27.1, abs=0.15)
    assert poids["chat"] * 100 == pytest.approx(37.2, abs=0.15)
    assert poids["dort"] * 100 == pytest.approx(35.7, abs=0.15)


def test_tous_les_poids_sont_positifs():
    assert all(p > 0 for p in ex.poids_attention("dort").values())


# --------------------------------------------------------------- question 4
def test_le_vecteur_enrichi_de_dort():
    assert ex.attention_a_la_main("dort") == pytest.approx([0.42, 0.48], abs=5e-3)


def test_le_vecteur_enrichi_de_chat():
    assert ex.attention_a_la_main("chat") == pytest.approx([0.43, 0.45], abs=5e-3)


def test_la_sortie_nest_plus_lembedding_de_depart():
    """Tout l'intérêt de l'attention : le vecteur a absorbé du contexte."""
    assert not np.allclose(ex.attention_a_la_main("dort"), ex.EMBEDDINGS["dort"])


def test_la_main_et_les_matrices_donnent_le_meme_resultat():
    """Le test le plus important du fichier : la formule ne cache rien."""
    sorties = ex.attention_matricielle(ex.X, ex.X, ex.X)
    for i, mot in enumerate(ex.MOTS):
        assert sorties[i] == pytest.approx(ex.attention_a_la_main(mot), abs=1e-9)


# --------------------------------------------------------------- question 5
def test_hors_lui_meme_dort_regarde_surtout_chat():
    poids = ex.poids_attention("dort")
    autres = {m: p for m, p in poids.items() if m != "dort"}
    assert max(autres, key=autres.get) == "chat"


def test_le_determinant_recoit_le_moins_dattention():
    poids = ex.poids_attention("dort")
    assert min(poids, key=poids.get) == "Le"


def test_lattention_nest_pas_symetrique():
    """chat -> dort (35,7 %) n'est pas dort -> chat (33,2 %)."""
    chat_vers_dort = ex.poids_attention("chat")["dort"]
    dort_vers_chat = ex.poids_attention("dort")["chat"]
    assert chat_vers_dort != pytest.approx(dort_vers_chat, abs=1e-3)


# ------------------------------------------------------------- la softmax
def test_softmax_somme_a_un():
    assert ex.softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


def test_softmax_ne_deborde_pas():
    resultat = ex.softmax(np.array([1000.0, 999.0]))
    assert np.all(np.isfinite(resultat))


def test_sans_mise_a_lechelle_le_softmax_sature():
    """Pourquoi on divise par √dk : sans elle, les poids deviennent 0 ou 1,
    le gradient s'annule et le modèle n'apprend plus."""
    grands_scores = np.array([50.0, 10.0, 5.0])
    poids = ex.softmax(grands_scores)
    assert poids[0] == pytest.approx(1.0)
    assert poids[1] == pytest.approx(0.0, abs=1e-15)

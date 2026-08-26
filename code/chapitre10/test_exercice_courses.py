"""Tests du corrigé du chapitre 10."""

import pytest

pd = pytest.importorskip("pandas", reason="pandas est requis : pip install -r requirements.txt")

import exercice_courses as ex  # noqa: E402  (import après importorskip, volontaire)


@pytest.fixture
def panier():
    return ex.creer_panier()


def test_le_panier_a_trois_lignes_et_deux_colonnes(panier):
    assert panier.shape == (3, 2)
    assert list(panier.columns) == ["produit", "prix"]


def test_les_prix_sont_numeriques(panier):
    assert pd.api.types.is_numeric_dtype(panier["prix"])


def test_une_colonne_est_une_series(panier):
    assert isinstance(ex.colonne_prix(panier), pd.Series)


def test_deux_crochets_donnent_un_dataframe(panier):
    # Le piège classique : df["prix"] vs df[["prix"]].
    assert isinstance(panier[["prix"]], pd.DataFrame)


def test_prix_moyen():
    # (1.2 + 0.9 + 2.5) / 3 = 1.5333...
    assert ex.prix_moyen(ex.creer_panier()) == pytest.approx(1.5333333, rel=1e-6)


def test_filtrage_garde_pain_et_oeufs(panier):
    chers = ex.produits_chers(panier)
    assert list(chers["produit"]) == ["pain", "œufs"]


def test_filtrage_ne_modifie_pas_le_dataframe_origine(panier):
    ex.produits_chers(panier)
    assert len(panier) == 3


def test_seuil_tres_haut_renvoie_un_tableau_vide(panier):
    assert ex.produits_chers(panier, seuil=99).empty


def test_seuil_tres_bas_renvoie_tout(panier):
    assert len(ex.produits_chers(panier, seuil=0)) == 3


def test_double_condition(panier):
    entre = ex.produits_entre(panier, 1.0, 2.0)
    assert list(entre["produit"]) == ["pain"]


def test_tva_ajoute_une_colonne_sans_toucher_a_loriginal(panier):
    avec_tva = ex.ajouter_tva(panier)
    assert "prix_ttc" not in panier.columns
    assert avec_tva.loc[0, "prix_ttc"] == pytest.approx(1.27, abs=0.01)


@pytest.mark.parametrize("produit", ["pain", "lait", "œufs"])
def test_chaque_produit_est_present(panier, produit):
    assert produit in set(panier["produit"])

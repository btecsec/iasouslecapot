"""Tests du corrigé du chapitre 32."""

import pytest

pd = pytest.importorskip("pandas", reason="pip install -r requirements.txt")

import donnees  # noqa: E402
import exercice_nettoyage as ex  # noqa: E402


@pytest.fixture
def df():
    return donnees.manchots_synthetiques()


# --------------------------------------------------------------- question 1
def test_les_trous_sont_comptes(df):
    manquants = ex.compter_manquants(df)
    assert manquants["body_mass_g"] == 2
    assert manquants["sex"] == 11


# --------------------------------------------------------------- question 2
def test_apres_remplissage_plus_aucun_trou_numerique(df):
    rempli = ex.remplir_medianes(df)
    for colonne in donnees.COLONNES_NUMERIQUES:
        assert rempli[colonne].isnull().sum() == 0


def test_le_remplissage_ne_touche_pas_les_colonnes_texte(df):
    rempli = ex.remplir_medianes(df)
    assert rempli["sex"].isnull().sum() == 11


def test_le_remplissage_ne_modifie_pas_loriginal(df):
    ex.remplir_medianes(df)
    assert df["body_mass_g"].isnull().sum() == 2


def test_la_mediane_est_bien_la_valeur_injectee(df):
    mediane = df["body_mass_g"].median()
    ligne_trouee = df["body_mass_g"].isnull().idxmax()
    assert ex.remplir_medianes(df).loc[ligne_trouee, "body_mass_g"] == mediane


def test_la_mediane_resiste_a_une_valeur_aberrante():
    """L'argument du corrigé, vérifié : la moyenne bouge, pas la médiane."""
    normal = pd.Series([3000.0, 3500.0, 4000.0, 4500.0, 5000.0])
    avec_aberration = pd.Series([3000.0, 3500.0, 4000.0, 4500.0, 500000.0])
    assert normal.median() == avec_aberration.median()
    assert normal.mean() != avec_aberration.mean()


def test_les_categories_manquantes_sont_remplies(df):
    rempli = ex.remplir_categories(df)
    assert rempli["sex"].isnull().sum() == 0
    assert set(rempli["sex"].unique()) <= {"Male", "Female"}


# --------------------------------------------------------------- question 3
def test_les_doublons_sont_supprimes(df):
    avec_doublons = pd.concat([df, df.head(5)], ignore_index=True)
    assert len(ex.supprimer_doublons(avec_doublons)) == len(ex.supprimer_doublons(df))


def test_sans_doublon_rien_ne_change(df):
    assert len(ex.supprimer_doublons(df)) == len(df)


# --------------------------------------------------------------- question 4
def test_lencodage_cree_une_colonne_par_categorie(df):
    encode = ex.encoder(ex.remplir_categories(df))
    assert "island_Biscoe" in encode.columns
    assert "sex_Male" in encode.columns


def test_lencodage_ne_touche_pas_la_cible(df):
    encode = ex.encoder(ex.remplir_categories(df))
    assert "species" in encode.columns
    assert "species_Adelie" not in encode.columns


def test_les_colonnes_encodees_ne_valent_que_zero_ou_un(df):
    encode = ex.encoder(ex.remplir_categories(df))
    for colonne in ("island_Biscoe", "sex_Male"):
        assert set(encode[colonne].unique()) <= {0, 1}


def test_le_one_hot_evite_dinventer_un_ordre(df):
    """Le point du quiz : avec island=1,2,3 le modèle croirait Torgersen > Dream."""
    encode = ex.encoder(ex.remplir_categories(df))
    colonnes_iles = [c for c in encode.columns if c.startswith("island_")]
    assert len(colonnes_iles) == 3
    # Une seule île active par ligne, donc la somme vaut toujours 1.
    assert (encode[colonnes_iles].sum(axis=1) == 1).all()


# --------------------------------------------------------------- question 5
def test_apres_normalisation_moyenne_nulle_et_ecart_type_un(df):
    pytest.importorskip("sklearn")
    propre = ex.encoder(ex.remplir_categories(ex.remplir_medianes(df)))
    normalise, _ = ex.normaliser(propre)
    for colonne in donnees.COLONNES_NUMERIQUES:
        assert normalise[colonne].mean() == pytest.approx(0, abs=1e-9)
        assert normalise[colonne].std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_le_scaler_memorise_les_moyennes_dorigine(df):
    """Sans lui, impossible de traiter une nouvelle donnée de la même façon."""
    pytest.importorskip("sklearn")
    propre = ex.encoder(ex.remplir_categories(ex.remplir_medianes(df)))
    _, scaler = ex.normaliser(propre)
    assert scaler.mean_[3] == pytest.approx(propre["body_mass_g"].mean())


def test_la_normalisation_ne_change_pas_lordre_des_valeurs(df):
    """Normaliser déplace l'échelle, jamais le classement."""
    pytest.importorskip("sklearn")
    propre = ex.encoder(ex.remplir_categories(ex.remplir_medianes(df)))
    normalise, _ = ex.normaliser(propre)
    assert (
        propre["body_mass_g"].rank().tolist() == normalise["body_mass_g"].rank().tolist()
    )


# ------------------------------------------------------------ la chaîne
def test_la_chaine_complete_ne_laisse_aucun_trou(df):
    pytest.importorskip("sklearn")
    final, _ = ex.nettoyer_tout(df)
    assert final.isnull().sum().sum() == 0


def test_la_chaine_complete_ne_laisse_que_des_nombres_et_la_cible(df):
    pytest.importorskip("sklearn")
    final, _ = ex.nettoyer_tout(df)
    non_numeriques = [
        c for c in final.columns if not pd.api.types.is_numeric_dtype(final[c])
    ]
    assert non_numeriques == ["species"]


def test_la_chaine_conserve_toutes_les_lignes(df):
    pytest.importorskip("sklearn")
    final, _ = ex.nettoyer_tout(df)
    assert len(final) == len(df)

"""Tests du corrigé du chapitre 31.

Les tests tournent sur le jeu **synthétique** : ils sont donc reproductibles,
instantanés, et ne dépendent d'aucun téléchargement. Un seul test touche au
vrai dataset, et il s'ignore proprement s'il n'est pas disponible.
"""

import pytest

pytest.importorskip("pandas", reason="pip install -r requirements.txt")

import donnees  # noqa: E402
import exercice_exploration as ex  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return donnees.manchots_synthetiques()


@pytest.fixture(scope="module")
def df_reel():
    """Le vrai dataset, ou un skip si le téléchargement est impossible."""
    try:
        import seaborn as sns

        return sns.load_dataset("penguins")
    except Exception:
        pytest.skip("dataset penguins indisponible (pas de réseau ou pas de seaborn)")


# --------------------------------------------------------------- question 1
def test_dimensions(df):
    assert ex.dimensions(df) == (344, 7)


def test_les_sept_colonnes_attendues(df):
    assert set(df.columns) == {
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    }


# --------------------------------------------------------------- question 2
def test_les_colonnes_a_trous_sont_reperees(df):
    manquantes = ex.valeurs_manquantes(df)
    assert manquantes["bill_length_mm"] == 2
    assert manquantes["sex"] == 11


def test_les_colonnes_sans_trou_ne_sont_pas_listees(df):
    assert "species" not in ex.valeurs_manquantes(df)
    assert "island" not in ex.valeurs_manquantes(df)


def test_un_dataset_sans_trou_ne_signale_rien():
    propre = donnees.manchots_synthetiques(avec_trous=False)
    assert ex.valeurs_manquantes(propre) == {}


# --------------------------------------------------------------- question 3
def test_les_trois_especes(df):
    assert set(ex.especes(df)) == {"Adelie", "Chinstrap", "Gentoo"}


def test_les_effectifs(df):
    effectifs = ex.especes(df)
    assert effectifs["Adelie"] == 152
    assert effectifs["Gentoo"] == 124
    assert effectifs["Chinstrap"] == 68
    assert sum(effectifs.values()) == 344


def test_le_dataset_est_legerement_desequilibre(df):
    """152 / 68 ≈ 2,2 : acceptable, mais l'accuracy seule sera trompeuse."""
    assert ex.est_equilibre(df, tolerance=3.0) is True
    assert ex.est_equilibre(df, tolerance=2.0) is False


# --------------------------------------------------------------- question 4
def test_lhistogramme_compte_tous_les_manchots_mesures(df):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = ex.distribution_masse(df, bins=10)
    total = sum(barre.get_height() for barre in figure.axes[0].patches)
    assert total == df["body_mass_g"].notna().sum()
    plt.close(figure)


# --------------------------------------------------------------- question 5
def test_le_diagnostic_signale_les_valeurs_manquantes(df):
    diagnostic = " | ".join(ex.problemes_a_corriger(df))
    assert "valeurs manquantes" in diagnostic


def test_le_diagnostic_signale_les_colonnes_a_encoder(df):
    diagnostic = " | ".join(ex.problemes_a_corriger(df))
    assert "encoder" in diagnostic


def test_le_diagnostic_signale_les_echelles_incomparables(df):
    """La masse (en grammes, ~5000) contre le bec (en mm, ~40)."""
    diagnostic = " | ".join(ex.problemes_a_corriger(df))
    assert "normalisation" in diagnostic


def test_un_dataset_deja_propre_pose_moins_de_problemes(df):
    propre = donnees.manchots_synthetiques(avec_trous=False)
    assert len(ex.problemes_a_corriger(propre)) < len(ex.problemes_a_corriger(df))


# ------------------------------------------------------- le vrai dataset
def test_le_jeu_synthetique_a_la_meme_forme_que_le_vrai(df, df_reel):
    """Si ce test casse, le jeu de secours a divergé du vrai dataset."""
    assert ex.dimensions(df) == ex.dimensions(df_reel)
    assert set(df.columns) == set(df_reel.columns)
    assert ex.especes(df) == ex.especes(df_reel)
    assert ex.valeurs_manquantes(df) == ex.valeurs_manquantes(df_reel)

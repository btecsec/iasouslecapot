"""Tests du corrigé du chapitre 11.

Tester un graphique, ce n'est pas comparer des pixels : c'est vérifier qu'on a
bien tracé ce qu'on croyait tracer (les bonnes valeurs, les bons libellés).
Le backend « Agg » dessine en mémoire, sans écran : indispensable en CI.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="pip install -r requirements.txt")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import exercice_visualisation as ex  # noqa: E402


@pytest.fixture(autouse=True)
def fermer_les_figures():
    """Après chaque test, on referme tout : sinon matplotlib avertit à partir
    de 20 figures ouvertes."""
    yield
    plt.close("all")


def test_la_courbe_trace_les_cinq_points():
    figure = ex.courbe_frequentation()
    (ligne,) = figure.axes[0].get_lines()
    assert list(ligne.get_ydata()) == ex.VISITEURS


def test_la_courbe_est_legendee():
    axes = ex.courbe_frequentation().axes[0]
    assert axes.get_title() == "Fréquentation de la semaine"
    assert axes.get_xlabel() == "Jour"
    assert axes.get_ylabel() == "Nombre de visiteurs"


def test_les_barres_ont_les_bonnes_hauteurs():
    axes = ex.barres_frequentation().axes[0]
    hauteurs = [barre.get_height() for barre in axes.patches]
    assert hauteurs == ex.VISITEURS


def test_il_y_a_une_barre_par_jour():
    axes = ex.barres_frequentation().axes[0]
    assert len(axes.patches) == len(ex.JOURS)


@pytest.mark.parametrize("bins", [5, 10, 20])
def test_lhistogramme_respecte_le_nombre_de_tranches(bins):
    axes = ex.histogramme_ages(bins=bins).axes[0]
    assert len(axes.patches) == bins


def test_lhistogramme_compte_tous_les_ages():
    axes = ex.histogramme_ages(bins=5).axes[0]
    total = sum(barre.get_height() for barre in axes.patches)
    assert total == len(ex.AGES)


def test_lecture_du_pic():
    assert ex.jour_le_plus_frequente() == "Jeu"


def test_lecture_du_creux():
    assert ex.jour_le_moins_frequente() == "Mer"


def test_donnees_personnalisees():
    jours = ["A", "B"]
    visiteurs = [10, 50]
    assert ex.jour_le_plus_frequente(jours, visiteurs) == "B"
    axes = ex.barres_frequentation(jours, visiteurs).axes[0]
    assert [b.get_height() for b in axes.patches] == [10, 50]


def test_la_figure_sait_secrire_sur_le_disque(tmp_path):
    fichier = tmp_path / "courbe.png"
    ex.courbe_frequentation().savefig(fichier)
    assert fichier.exists() and fichier.stat().st_size > 0

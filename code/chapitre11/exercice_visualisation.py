"""Chapitre 11 — corrigé de l'exercice : la fréquentation de la semaine.

Chaque fonction **renvoie** sa figure au lieu d'appeler plt.show() elle-même.
Deux avantages : on peut enchaîner les graphiques, et on peut les tester
(voir test_exercice_visualisation.py) sans ouvrir la moindre fenêtre.

Usage :
    python exercice_visualisation.py         # affiche les fenêtres
    python exercice_visualisation.py --png   # écrit les fichiers .png
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven"]
VISITEURS = [200, 240, 180, 300, 260]
AGES = [19, 22, 25, 25, 28, 31, 33, 34, 34, 36, 39, 41, 45, 48, 52, 55, 61, 67]


# --------------------------------------------------------------- question 2
def courbe_frequentation(jours=JOURS, visiteurs=VISITEURS) -> Figure:
    """Une courbe : la bonne forme quand l'axe des X est ordonné (le temps)."""
    figure, axes = plt.subplots()
    axes.plot(jours, visiteurs, marker="o")
    axes.set_title("Fréquentation de la semaine")
    axes.set_xlabel("Jour")
    axes.set_ylabel("Nombre de visiteurs")
    axes.grid(True, alpha=0.3)
    return figure


# --------------------------------------------------------------- question 3
def barres_frequentation(jours=JOURS, visiteurs=VISITEURS) -> Figure:
    """Les mêmes données en barres : on compare des catégories."""
    figure, axes = plt.subplots()
    axes.bar(jours, visiteurs)
    axes.set_title("Fréquentation de la semaine (barres)")
    axes.set_xlabel("Jour")
    axes.set_ylabel("Nombre de visiteurs")
    return figure


# ------------------------------------------------------------------- bonus
def histogramme_ages(ages=AGES, bins: int = 5) -> Figure:
    """Un histogramme : la répartition d'une seule variable numérique.

    `bins` n'est pas un détail : trop peu de tranches lisse tout, trop de
    tranches transforme le dessin en peigne. Essayez 5 puis 20.
    """
    figure, axes = plt.subplots()
    axes.hist(ages, bins=bins, edgecolor="white")
    axes.set_title(f"Répartition des âges ({bins} tranches)")
    axes.set_xlabel("Âge")
    axes.set_ylabel("Effectif")
    return figure


def jour_le_plus_frequente(jours=JOURS, visiteurs=VISITEURS) -> str:
    """La lecture du graphique, en code : le pic de la semaine."""
    return jours[visiteurs.index(max(visiteurs))]


def jour_le_moins_frequente(jours=JOURS, visiteurs=VISITEURS) -> str:
    """Et le creux."""
    return jours[visiteurs.index(min(visiteurs))]


def main(argv: list[str]) -> int:
    en_png = "--png" in argv

    figures = {
        "courbe.png": courbe_frequentation(),
        "barres.png": barres_frequentation(),
        "histogramme.png": histogramme_ages(),
    }

    print(f"Pic de la semaine   : {jour_le_plus_frequente()}")
    print(f"Creux de la semaine : {jour_le_moins_frequente()}")

    if en_png:
        for nom, figure in figures.items():
            figure.savefig(nom, dpi=120, bbox_inches="tight")
            print(f"écrit : {nom}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

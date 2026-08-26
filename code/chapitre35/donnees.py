"""Le chargement des manchots, utilisé par tous les chapitres de la partie IV.

`charger_manchots()` essaie d'abord le vrai dataset (Seaborn le télécharge la
première fois), et bascule sinon sur un jeu **synthétique** de même forme.
Ainsi, ni un avion ni un pare-feu d'entreprise ne peut bloquer l'exercice — et
les tests tournent hors ligne, ce qui est la règle du chapitre 9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLONNES_NUMERIQUES = [
    "bill_length_mm",  # longueur du bec
    "bill_depth_mm",  # épaisseur du bec
    "flipper_length_mm",  # longueur des nageoires
    "body_mass_g",  # masse corporelle
]
COLONNES_CATEGORIELLES = ["island", "sex"]
CIBLE = "species"

# Moyennes réelles par espèce (arrondies), pour que le jeu synthétique
# ressemble vraiment aux manchots de Palmer.
PROFILS = {
    #            bec long, bec épais, nageoire, masse
    "Adelie": (38.8, 18.3, 190.0, 3700.0),
    "Chinstrap": (48.8, 18.4, 195.8, 3733.0),
    "Gentoo": (47.5, 15.0, 217.2, 5076.0),
}
EFFECTIFS = {"Adelie": 152, "Chinstrap": 68, "Gentoo": 124}


def manchots_synthetiques(graine: int = 42, avec_trous: bool = True) -> pd.DataFrame:
    """Un faux dataset de manchots, reproductible et hors ligne.

    Mêmes colonnes, mêmes espèces, mêmes proportions et quelques valeurs
    manquantes : de quoi rejouer tous les exercices de la partie IV.
    """
    generateur = np.random.default_rng(graine)
    lignes = []

    for espece, (bec_l, bec_e, nageoire, masse) in PROFILS.items():
        for _ in range(EFFECTIFS[espece]):
            lignes.append(
                {
                    "species": espece,
                    "island": generateur.choice(["Biscoe", "Dream", "Torgersen"]),
                    "bill_length_mm": round(float(generateur.normal(bec_l, 2.5)), 1),
                    "bill_depth_mm": round(float(generateur.normal(bec_e, 1.0)), 1),
                    "flipper_length_mm": round(float(generateur.normal(nageoire, 6.0)), 1),
                    "body_mass_g": round(float(generateur.normal(masse, 400.0)), 1),
                    "sex": generateur.choice(["Male", "Female"]),
                }
            )

    df = pd.DataFrame(lignes).sample(frac=1, random_state=graine).reset_index(drop=True)

    if avec_trous:
        # Les mêmes trous que dans le vrai dataset : 2 par colonne numérique,
        # 11 sur le sexe. Sans eux, le chapitre 32 n'aurait rien à nettoyer.
        for colonne in COLONNES_NUMERIQUES:
            df.loc[generateur.choice(df.index, 2, replace=False), colonne] = np.nan
        df.loc[generateur.choice(df.index, 11, replace=False), "sex"] = np.nan

    return df


def charger_manchots(hors_ligne: bool = False) -> pd.DataFrame:
    """Le vrai dataset si possible, le jeu synthétique sinon."""
    if not hors_ligne:
        try:
            import seaborn as sns

            return sns.load_dataset("penguins")
        except Exception:  # pas de seaborn, pas de réseau, cache vide…
            pass
    return manchots_synthetiques()

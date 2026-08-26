"""Chapitre 49 — l'API à faire passer dans la chaîne CI/CD.

Même principe qu'au chapitre 45, en autonome : le modèle est entraîné au
démarrage si le fichier n'existe pas, pour que la chaîne CI puisse tourner sur
une machine vierge sans rien télécharger d'autre que les dépendances.

Usage :
    uvicorn api:app --reload
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

MODELE = Path(__file__).parent / "modele.joblib"

app = FastAPI(title="API manchots (CI/CD)")


class Manchot(BaseModel):
    """Le contrat d'entrée. `gt=0` fait le tri avant que le modèle ne voie
    quoi que ce soit : une masse négative n'a pas de sens."""

    longueur_bec: float = Field(gt=0)
    profondeur_bec: float = Field(gt=0)
    longueur_nageoire: float = Field(gt=0)
    masse: float = Field(gt=0)


def entrainer_si_besoin() -> RandomForestClassifier:
    """Charge le modèle, ou l'entraîne (2 secondes) et le sauvegarde.

    SÉCURITÉ : joblib exécute du code en désérialisant. On ne charge ici qu'un
    fichier produit par ce script (chapitre 42).
    """
    if MODELE.exists():
        return joblib.load(MODELE)

    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    modele = RandomForestClassifier(n_estimators=100, random_state=42)
    modele.fit(df[COLONNES_NUMERIQUES], df[CIBLE])
    joblib.dump(modele, MODELE)
    return modele


modele = entrainer_si_besoin()  # chargé une fois, au démarrage


@app.get("/sante")
def sante():
    """Sonde de vie : `ok` seulement si le modèle est réellement chargé."""
    return {"statut": "ok" if modele is not None else "degrade"}


@app.post("/predire")
def predire(manchot: Manchot):
    donnees = pd.DataFrame(
        [
            {
                "bill_length_mm": manchot.longueur_bec,
                "bill_depth_mm": manchot.profondeur_bec,
                "flipper_length_mm": manchot.longueur_nageoire,
                "body_mass_g": manchot.masse,
            }
        ]
    )[COLONNES_NUMERIQUES]  # l'ordre des colonnes est imposé, pas subi
    return {"espece": str(modele.predict(donnees)[0])}

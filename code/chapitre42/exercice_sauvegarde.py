"""Chapitre 42 — corrigé de l'exercice : sauvegarder et recharger.

La règle du chapitre tient en une phrase : **on sauvegarde tout ce qui a été
appris**, pas seulement le modèle. Un scaler oublié, et les prédictions sont
fausses sans le moindre message d'erreur.

Usage :
    python exercice_sauvegarde.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42
DOSSIER = Path(__file__).parent / "modeles"


def preparer(graine: int = GRAINE):
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X, y = df[COLONNES_NUMERIQUES], df[CIBLE]
    return train_test_split(X, y, test_size=0.2, random_state=graine, stratify=y)


# --------------------------------------------------------------- question 1
def sauver_scikit(modele, chemin: Path) -> Path:
    """joblib : le format standard pour Scikit-learn (plus efficace que pickle
    sur les gros tableaux NumPy)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin)
    return chemin


def charger_scikit(chemin: Path):
    """Recharge un modèle.

    SÉCURITÉ — à savoir avant de mettre ce code en production : joblib (comme
    pickle) **exécute du code** en désérialisant. Ne chargez jamais un fichier
    .joblib venant d'une source que vous ne contrôlez pas : télécharger le
    « modèle entraîné » d'un inconnu revient à exécuter son script. Pour un
    modèle tiers, préférez un format qui ne transporte que des nombres (ONNX,
    safetensors) — c'est le même raisonnement que le `weights_only=True` de
    `torch.load`.
    """
    return joblib.load(chemin)


# --------------------------------------------------------------- question 4
def sauver_avec_scaler(modele, scaler, colonnes, chemin: Path) -> Path:
    """La bonne pratique : un seul fichier, qui contient tout le nécessaire.

    On y met aussi la **liste des colonnes** : c'est le troisième oubli
    classique, celui qui fait qu'une API reçoit les mesures dans le désordre
    et prédit n'importe quoi sans jamais lever d'erreur.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modele": modele, "scaler": scaler, "colonnes": list(colonnes)}, chemin)
    return chemin


def predire_avec_paquet(paquet: dict, X) -> np.ndarray:
    """Rejoue exactement la préparation de l'entraînement, dans le même ordre."""
    X = X[paquet["colonnes"]]
    return paquet["modele"].predict(paquet["scaler"].transform(X))


# --------------------------------------------------------------- question 5
def taille_ko(chemin: Path) -> float:
    return round(chemin.stat().st_size / 1024, 1)


def main() -> None:
    X_train, X_test, y_train, y_test = preparer()

    # --- Scikit-learn, la version fragile : modèle et scaler séparés
    scaler = StandardScaler().fit(X_train)
    modele = RandomForestClassifier(n_estimators=100, random_state=GRAINE)
    modele.fit(scaler.transform(X_train), y_train)

    chemin_modele = sauver_scikit(modele, DOSSIER / "foret.joblib")
    recharge = charger_scikit(chemin_modele)
    identiques = np.array_equal(
        modele.predict(scaler.transform(X_test)), recharge.predict(scaler.transform(X_test))
    )
    print(f"1. modele scikit recharge : predictions identiques -> {identiques}")

    # --- le piège : recharger le modèle sans le scaler
    sans_scaler = recharge.predict(X_test.to_numpy())  # données brutes !
    avec_scaler = recharge.predict(scaler.transform(X_test))
    differentes = int((sans_scaler != avec_scaler).sum())
    print(
        f"   sans le scaler : {differentes} predictions sur {len(X_test)} changent"
        " — et aucune erreur ne s'affiche"
    )

    # --- Scikit-learn, la version solide : tout dans un seul fichier
    chemin_paquet = sauver_avec_scaler(
        modele, scaler, X_train.columns, DOSSIER / "manchots_complet.joblib"
    )
    paquet = charger_scikit(chemin_paquet)
    print(
        "4. paquet complet recharge : predictions identiques -> "
        f"{np.array_equal(predire_avec_paquet(paquet, X_test), avec_scaler)}"
    )

    # --- l'alternative la plus simple : un pipeline
    pipeline = make_pipeline(
        StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=GRAINE)
    ).fit(X_train, y_train)
    chemin_pipeline = sauver_scikit(pipeline, DOSSIER / "pipeline.joblib")
    print(
        "   variante pipeline : le scaler est DANS le modele, "
        "impossible de l'oublier"
    )

    print("\n5. tailles des fichiers :")
    for chemin in (chemin_modele, chemin_paquet, chemin_pipeline):
        print(f"     {chemin.name:<26} {taille_ko(chemin):>8} Ko")
    print(
        "   Le paquet complet ne pese presque rien de plus : le scaler ne"
        " contient\n   que quelques moyennes et ecarts-types."
    )


if __name__ == "__main__":
    main()

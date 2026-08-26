"""Chapitre 4 — suis-je, oui ou non, dans un environnement virtuel ?

Usage :
    python verifier_venv.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def dans_un_venv(prefix: str | None = None, base_prefix: str | None = None) -> bool:
    """True si l'interpréteur courant tourne dans un environnement virtuel.

    Le principe : `sys.prefix` pointe sur l'environnement actif, alors que
    `sys.base_prefix` pointe toujours sur l'installation Python d'origine.
    Hors venv, les deux sont identiques ; dans un venv, ils diffèrent.

    Les paramètres servent uniquement aux tests, pour simuler les deux cas.
    """
    prefix = sys.prefix if prefix is None else prefix
    base_prefix = sys.base_prefix if base_prefix is None else base_prefix
    return prefix != base_prefix


def nom_environnement(prefix: str | None = None) -> str:
    """Le nom du dossier de l'environnement, par exemple 'venv'."""
    return Path(sys.prefix if prefix is None else prefix).name


def chemin_pip() -> Path:
    """Là où `pip install` déposera les paquets."""
    return Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")


def main() -> int:
    if dans_un_venv():
        print(f"[ok]   Environnement virtuel actif : {nom_environnement()}")
        print(f"       Interpréteur : {sys.executable}")
        print(f"       pip installera dans : {chemin_pip()}")
        return 0

    print("[KO]   Aucun environnement virtuel actif.")
    print("       Vous êtes sur le Python global — vos installations vont")
    print("       polluer toute la machine.")
    print("       Créez-en un : python -m venv venv, puis activez-le.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

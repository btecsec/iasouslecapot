"""Chapitre 3 — vérifier qu'un poste est prêt pour la suite du livre.

Usage :
    python verifier_installation.py

Le script ne modifie rien : il regarde et il raconte.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys

VERSION_MINIMALE = (3, 10)


def version_python_ok(version: tuple[int, ...] = None) -> bool:
    """Renvoie True si la version de Python est >= 3.10.

    On passe `version` explicitement dans les tests ; sinon on lit celle de
    l'interpréteur qui exécute ce fichier.
    """
    version = version or sys.version_info[:2]
    return tuple(version[:2]) >= VERSION_MINIMALE


def python_dans_le_path() -> bool:
    """True si une commande `python` (ou `py`) est trouvable dans le PATH."""
    return shutil.which("python") is not None or shutil.which("py") is not None


def pip_disponible() -> bool:
    """True si le module pip est importable.

    On teste le *module* et non la commande : `python -m pip` fonctionne même
    quand le raccourci `pip` manque, et c'est la forme à privilégier.
    """
    return importlib.util.find_spec("pip") is not None


def encodage_sortie() -> str:
    """L'encodage du terminal, en minuscules ('utf-8' attendu)."""
    return (getattr(sys.stdout, "encoding", None) or "inconnu").lower()


def diagnostic() -> list[tuple[bool, str]]:
    """Construit la liste des vérifications, sous forme (réussi, message)."""
    majeur, mineur = sys.version_info[:2]
    attendu = ".".join(str(n) for n in VERSION_MINIMALE)
    return [
        (
            version_python_ok(),
            f"Version de Python : {majeur}.{mineur} (>= {attendu} requis)",
        ),
        (python_dans_le_path(), "Python est bien dans le PATH"),
        (pip_disponible(), "pip disponible"),
        (
            encodage_sortie().replace("-", "") == "utf8",
            f"Encodage de sortie : {encodage_sortie()}",
        ),
    ]


def main() -> int:
    """Affiche le diagnostic. Renvoie 0 si tout va bien, 1 sinon."""
    resultats = diagnostic()
    for reussi, message in resultats:
        print(f"[{'ok' if reussi else 'KO'}]   {message}")

    if all(reussi for reussi, _ in resultats):
        print("Tout est prêt. Passez au chapitre 4.")
        return 0

    print("\nRegardez la section « les trois pannes classiques » du README.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

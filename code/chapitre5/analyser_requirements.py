"""Chapitre 5 — lire un requirements.txt et le commenter.

Usage :
    python analyser_requirements.py requirements.txt
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# nom[extra] opérateur version — on accepte les opérateurs courants de PEP 508
MOTIF = re.compile(
    r"^\s*(?P<nom>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]+\])?"
    r"\s*(?P<operateur>==|>=|<=|~=|!=|>|<)?"
    r"\s*(?P<version>[A-Za-z0-9][A-Za-z0-9.*+!-]*)?\s*$"
)


@dataclass(frozen=True)
class Dependance:
    """Une ligne utile de requirements.txt."""

    nom: str
    operateur: str | None
    version: str | None

    @property
    def figee(self) -> bool:
        """True si la version est verrouillée au chiffre près (`==`)."""
        return self.operateur == "=="

    @property
    def sans_version(self) -> bool:
        """True si aucune version n'est indiquée — la dépendance dangereuse."""
        return self.version is None


def nettoyer(lignes: list[str]) -> list[str]:
    """Retire les commentaires, les lignes vides et les options (`-r`, `--`)."""
    utiles = []
    for ligne in lignes:
        ligne = ligne.split("#", 1)[0].strip()
        if not ligne or ligne.startswith("-"):
            continue
        utiles.append(ligne)
    return utiles


def analyser_ligne(ligne: str) -> Dependance | None:
    """Transforme une ligne en Dependance, ou None si elle est illisible."""
    correspondance = MOTIF.match(ligne)
    if correspondance is None:
        return None
    return Dependance(
        nom=correspondance["nom"],
        operateur=correspondance["operateur"],
        version=correspondance["version"],
    )


def analyser(contenu: str) -> tuple[list[Dependance], list[str]]:
    """Analyse un fichier entier.

    Renvoie (dépendances comprises, lignes incomprises).
    """
    dependances: list[Dependance] = []
    illisibles: list[str] = []
    for ligne in nettoyer(contenu.splitlines()):
        dependance = analyser_ligne(ligne)
        if dependance is None:
            illisibles.append(ligne)
        else:
            dependances.append(dependance)
    return dependances, illisibles


def rapport(dependances: list[Dependance], illisibles: list[str]) -> str:
    """Le texte affiché à l'utilisateur."""
    figees = [d for d in dependances if d.figee]
    libres = [d for d in dependances if not d.figee and not d.sans_version]
    nues = [d for d in dependances if d.sans_version]

    lignes = [f"{len(dependances)} dépendances lues."]
    lignes.append(f"  figées (==)        : {len(figees)}")
    lignes.append(f"  contraintes souples: {len(libres)}")
    lignes.append(f"  sans version       : {len(nues)}")

    if nues:
        lignes.append("")
        lignes.append("Attention — sans version, l'installation de demain")
        lignes.append("ne donnera pas le même environnement qu'aujourd'hui :")
        lignes += [f"  - {d.nom}" for d in nues]

    if illisibles:
        lignes.append("")
        lignes.append("Lignes non comprises :")
        lignes += [f"  ? {ligne}" for ligne in illisibles]

    return "\n".join(lignes)


def main(argv: list[str]) -> int:
    chemin = Path(argv[1] if len(argv) > 1 else "requirements.txt")
    if not chemin.exists():
        print(f"Fichier introuvable : {chemin}")
        return 1
    dependances, illisibles = analyser(chemin.read_text(encoding="utf-8"))
    print(rapport(dependances, illisibles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

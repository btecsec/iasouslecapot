"""Chapitre 27 — corrigé de l'exercice : les bases de PyTorch.

Formes, découpes, assemblage et dérivée automatique.

Usage :
    python exercice_bases_torch.py
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------- question 1
def suite() -> torch.Tensor:
    """Les 24 valeurs de 0 à 23, forme (24,)."""
    return torch.arange(24)


# --------------------------------------------------------------- question 2
def en_cube(tenseur: torch.Tensor) -> torch.Tensor:
    """Forme (2, 3, 4) : 2 × 3 × 4 = 24, le compte est bon."""
    return tenseur.reshape(2, 3, 4)


def en_tableau(tenseur: torch.Tensor) -> torch.Tensor:
    """Forme (6, 4). Le -1 laisse PyTorch calculer la seconde dimension."""
    return tenseur.reshape(6, -1)


# --------------------------------------------------------------- question 3
def premiere_colonne(tenseur: torch.Tensor) -> torch.Tensor:
    """Toutes les rangées, colonne 0 : le « : » veut dire « tout l'axe »."""
    return tenseur[:, 0]


def deuxieme_rangee(tenseur: torch.Tensor) -> torch.Tensor:
    """Les indices commencent à 0 : la deuxième rangée porte l'indice 1."""
    return tenseur[1]


# --------------------------------------------------------------- question 4
def formes_assemblage(a: torch.Tensor, b: torch.Tensor) -> dict[str, tuple[int, ...]]:
    """Les trois façons d'assembler deux tenseurs, et la forme obtenue.

    stack ajoute une dimension ; cat allonge une dimension existante.
    """
    return {
        "stack": tuple(torch.stack([a, b]).shape),
        "cat_dim0": tuple(torch.cat([a, b], dim=0).shape),
        "cat_dim1": tuple(torch.cat([a, b], dim=1).shape),
    }


def lot_dexemples(donnees: torch.Tensor, debuts: list[int],
                  taille_contexte: int) -> torch.Tensor:
    """Le geste du livre : fabriquer un lot (taille_lot, taille_contexte).

    C'est exactement la ligne dont se sert un modèle de langage pour piocher
    ses exemples d'entraînement dans un long texte encodé.
    """
    return torch.stack([donnees[i:i + taille_contexte] for i in debuts])


# --------------------------------------------------------------- question 5
def derivee_du_cube(valeur: float) -> float:
    """d(x³)/dx = 3x², soit 12 en x = 2. PyTorch le retrouve seul."""
    x = torch.tensor(valeur, requires_grad=True)
    y = x ** 3
    y.backward()
    return x.grad.item()


def main() -> None:
    t = suite()
    print(f"1. suite            : forme {tuple(t.shape)}")
    print(f"2. en cube          : forme {tuple(en_cube(t).shape)}")

    tableau = en_tableau(t)
    print(f"   en tableau       : forme {tuple(tableau.shape)}")
    print(f"3. première colonne : {premiere_colonne(tableau).tolist()}")
    print(f"   deuxième rangée  : {deuxieme_rangee(tableau).tolist()}")

    a, b = torch.randn(4, 5), torch.randn(4, 5)
    for nom, forme in formes_assemblage(a, b).items():
        print(f"4. {nom:9s}        : forme {forme}")

    lot = lot_dexemples(torch.arange(100), debuts=[0, 10, 20], taille_contexte=8)
    print(f"   lot d'exemples   : forme {tuple(lot.shape)}")

    print(f"5. dérivée de x³ en 2 : {derivee_du_cube(2.0)}")


if __name__ == "__main__":
    main()

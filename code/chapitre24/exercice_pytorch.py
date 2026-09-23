"""Chapitre 24 — corrigé de l'exercice : premiers pas avec PyTorch.

Usage :
    python exercice_pytorch.py
"""

from __future__ import annotations

import torch


def version() -> str:
    """La version installée, par exemple '2.4.1+cpu'."""
    return torch.__version__


def creer_tenseur() -> torch.Tensor:
    """Le tenseur de l'énoncé : [5, 10, 15]."""
    return torch.tensor([5, 10, 15])


def forme(tenseur: torch.Tensor) -> tuple[int, ...]:
    """La forme, sous forme de tuple ordinaire : (3,) et non torch.Size([3])."""
    return tuple(tenseur.shape)


def doubler(tenseur: torch.Tensor) -> torch.Tensor:
    """Multiplication vectorisée : aucune boucle Python n'est écrite."""
    return tenseur * 2


def gpu_disponible() -> bool:
    """True si un GPU NVIDIA utilisable est présent. False n'est pas une panne."""
    return torch.cuda.is_available()


def appareil() -> str:
    """Le motif standard : le même code tourne sur GPU comme sur processeur."""
    return "cuda" if gpu_disponible() else "cpu"


def main() -> None:
    print(f"Version de PyTorch : {version()}")

    t = creer_tenseur()
    print(f"Tenseur            : {t}")
    print(f"Forme              : {t.shape}  (soit {forme(t)})")
    print(f"Type des éléments  : {t.dtype}")
    print(f"Multiplié par 2    : {doubler(t)}")

    print(f"GPU disponible     : {gpu_disponible()}")
    print(f"Calculs sur        : {appareil()}")


if __name__ == "__main__":
    main()

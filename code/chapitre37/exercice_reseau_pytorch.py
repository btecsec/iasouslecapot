"""Chapitre 37 — corrigé de l'exercice : le même réseau, en PyTorch.

Le but n'est pas d'obtenir un meilleur score que Keras (ce sera le même), mais
de voir **où passe chaque étape** quand on écrit la boucle soi-même.

Usage :
    python exercice_reseau_pytorch.py
"""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from donnees import CIBLE, COLONNES_NUMERIQUES, charger_manchots

GRAINE = 42


# --------------------------------------------------------------- question 1
def preparer(graine: int = GRAINE):
    """Les données, converties en tenseurs.

    Le détail qui coûte une heure à qui l'ignore : les labels doivent être en
    `torch.long`, jamais en float. CrossEntropyLoss attend des indices de
    classe, pas des nombres réels.
    """
    df = charger_manchots().dropna(subset=COLONNES_NUMERIQUES + [CIBLE])
    X = df[COLONNES_NUMERIQUES].to_numpy(dtype="float32")
    encodeur = LabelEncoder()
    y = encodeur.fit_transform(df[CIBLE])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=graine, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        encodeur,
    )


# --------------------------------------------------------------- question 2
class ReseauManchots(nn.Module):
    """Deux couches cachées ReLU, une sortie à 3 neurones.

    Remarquez l'absence de softmax en sortie : `CrossEntropyLoss` l'applique
    elle-même (en version numériquement stable). L'ajouter ici serait une
    double application, et l'apprentissage en souffrirait.
    """

    def __init__(self, n_entrees: int = 4, n_classes: int = 3):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(n_entrees, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.couches(x)


# --------------------------------------------------------------- question 3
def outils(modele, lr: float = 0.01):
    """La perte et l'optimiseur — les deux choix du chapitre 38."""
    return nn.CrossEntropyLoss(), torch.optim.Adam(modele.parameters(), lr=lr)


# --------------------------------------------------------------- question 4
def entrainer(modele, X_train, y_train, epochs: int = 50, lr: float = 0.01):
    """La boucle d'entraînement, écrite à la main.

    Quatre lignes, toujours dans cet ordre — c'est le cœur du deep learning :
      1. zero_grad : effacer les gradients du tour précédent
      2. forward   : prédire
      3. backward  : calculer les gradients
      4. step      : corriger les poids
    """
    critere, optimiseur = outils(modele, lr)
    historique = []

    for _ in range(epochs):
        optimiseur.zero_grad()  # 1
        sorties = modele(X_train)  # 2
        perte = critere(sorties, y_train)
        perte.backward()  # 3
        optimiseur.step()  # 4
        historique.append(float(perte.item()))

    return historique


# --------------------------------------------------------------- question 5
def evaluer(modele, X_test, y_test) -> float:
    """`no_grad` : on ne calcule pas de gradients pour une simple évaluation.

    Le résultat serait identique sans, mais on gaspillerait mémoire et temps —
    et sur un gros modèle, la différence n'est pas anecdotique.
    """
    modele.eval()
    with torch.no_grad():
        predictions = modele(X_test).argmax(dim=1)
    modele.train()
    return float((predictions == y_test).float().mean().item())


def main() -> None:
    torch.manual_seed(GRAINE)
    X_train, X_test, y_train, y_test, encodeur = preparer()
    print(f"1. tenseurs : X {tuple(X_train.shape)}, y {y_train.dtype}")
    print(f"   classes : {dict(enumerate(encodeur.classes_))}")

    modele = ReseauManchots()
    total = sum(p.numel() for p in modele.parameters())
    print(f"2. {total} parametres (les memes 243 qu'en Keras)")

    critere, optimiseur = outils(modele)
    print(f"3. perte {type(critere).__name__}, optimiseur {type(optimiseur).__name__}")

    historique = entrainer(modele, X_train, y_train)
    print(f"4. perte : {historique[0]:.4f} -> {historique[-1]:.4f}")

    print(f"5. accuracy sur le test : {evaluer(modele, X_test, y_test):.4f}")

    print(
        "\n6. Chaque etape de Keras se retrouve ici :\n"
        "   Sequential -> nn.Sequential | compile -> outils()\n"
        "   fit        -> la boucle for | evaluate -> evaluer()"
    )


if __name__ == "__main__":
    main()

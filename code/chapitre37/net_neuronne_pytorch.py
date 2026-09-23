import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Ajout du chemin pour importer prepare_data depuis le chapitre 34
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chapitre34')))
from prepare_data import prepare_penguins_data

# 1. Données et Tenseurs
X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()

# Mapping des espèces vers des entiers (nécessaire pour CrossEntropyLoss)
species_to_int = {species: i for i, species in enumerate(y_train.unique())}
y_train_int = y_train.map(species_to_int)
y_test_int = y_test.map(species_to_int)

# Conversion en tenseurs (cast en float pour éviter numpy.object_)
X_train_t = torch.tensor(X_train.values.astype(float), dtype=torch.float32)
y_train_t = torch.tensor(y_train_int.values, dtype=torch.long)
X_test_t = torch.tensor(X_test.values.astype(float), dtype=torch.float32)
y_test_t = torch.tensor(y_test_int.values, dtype=torch.long)

# 2. Classe réseau avec deux couches cachées ReLU
class ReseauManchots(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(), # Couche 1
            nn.Linear(16, 8), nn.ReLU(),          # Couche 2
            nn.Linear(8, 3)                       # Sortie (3 espèces)
        )
    def forward(self, x):
        return self.couches(x)

modele = ReseauManchots(X_train_t.shape[1])

# 3. Perte et Optimiseur
critere = nn.CrossEntropyLoss()
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.01)

# 4. Boucle d'entraînement (50 époques)
print("Entraînement en cours...")
losses = []
for epoch in range(50):
    optimiseur.zero_grad()              # REMISE À ZÉRO (Crucial en PyTorch)
    sorties = modele(X_train_t)         # PASSAGE AVANT
    perte = critere(sorties, y_train_t) # CALCUL ERREUR
    perte.backward()                    # RÉTROPROPAGATION
    optimiseur.step()                   # MISE À JOUR POIDS

    losses.append(perte.item())         # Stockage perte
    if epoch % 10 == 0:
        print(f"Époque {epoch}, perte : {perte.item():.4f}")

# 5. Évaluation
with torch.no_grad():
    predictions = modele(X_test_t).argmax(dim=1)
    exactitude = (predictions == y_test_t).float().mean()
    print(f"\nExactitude sur le test : {exactitude:.2%}")

# 6. Courbe de perte
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Courbe de perte (Loss Curve)")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.grid(True)
plt.savefig("courbe_perte.png")
print("\nCourbe de perte sauvegardée sous 'courbe_perte.png'")
plt.show()

# --- Comparaison avec Keras ---
# 1. Tenseurs : Keras gère NumPy nativement, PyTorch exige des tenseurs explicites.
# 2. Architecture : nn.Sequential est l'équivalent de Sequential(Dense...).
# 3. Compilation : Pas de .compile(). On définit critere et optimiseur séparément.
# 4. Loop : Keras .fit() cache la boucle. PyTorch demande d'écrire zero_grad -> backward -> step.
# 5. Évaluation : torch.no_grad() est essentiel pour désactiver le gradient et gagner en vitesse.


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import data preparation from chapter 5
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chapitre29')))
from prepare_data import prepare_penguins_data

def setup_data():
    """Prépare les données et les convertit en tenseurs PyTorch."""
    X_train, X_test, y_train, y_test, scaler = prepare_penguins_data()

    # Mapping espèces -> entiers
    species_to_int = {species: i for i, species in enumerate(y_train.unique())}
    y_train_int = y_train.map(species_to_int)
    y_test_int = y_test.map(species_to_int)

    X_train_t = torch.tensor(X_train.values.astype(float), dtype=torch.float32)
    y_train_t = torch.tensor(y_train_int.values, dtype=torch.long)
    X_test_t = torch.tensor(X_test.values.astype(float), dtype=torch.float32)
    y_test_t = torch.tensor(y_test_int.values, dtype=torch.long)

    return X_train_t, X_test_t, y_train_t, y_test_t

class ReseauManchots(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3)
        )
    def forward(self, x):
        return self.couches(x)

def train_model(X_train, y_train, lr=0.01, batch_size=32, epochs=50):
    """Entraîne le modèle avec les paramètres donnés et retourne l'historique de la perte."""
    model = ReseauManchots(X_train.shape[1])
    criterion = nn.CrossEntropyLoss() # Fonction adaptée pour la classification multi-classe
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Création du DataLoader pour gérer le batch_size
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))

    return losses

# --- EXERCICES PRATIQUES ---

X_train_t, X_test_t, y_train_t, y_test_t = setup_data()

plt.figure(figsize=(15, 10))

# 1. Taux d'apprentissage (LR)
# Piste : lr=1.0 fera exploser la perte. lr=0.001 est stable.
plt.subplot(3, 1, 1)
loss_lr_low = train_model(X_train_t, y_train_t, lr=0.001, epochs=50)
loss_lr_high = train_model(X_train_t, y_train_t, lr=1.0, epochs=50)
plt.plot(loss_lr_low, label="LR = 0.001")
plt.plot(loss_lr_high, label="LR = 1.0")
plt.title("Effet du Taux d'Apprentissage")
plt.ylabel("Perte")
plt.legend()
plt.grid(True)

# 2. Taille des batchs (Batch Size)
# Piste : batch petit (8) = plus bruyant mais peut converger différemment.
# batch grand (64) = plus stable, calculs plus rapides par époque.
plt.subplot(3, 1, 2)
loss_bs_small = train_model(X_train_t, y_train_t, batch_size=8, epochs=50)
loss_bs_large = train_model(X_train_t, y_train_t, batch_size=64, epochs=50)
plt.plot(loss_bs_small, label="Batch Size = 8")
plt.plot(loss_bs_large, label="Batch Size = 64")
plt.title("Effet de la Taille du Batch")
plt.ylabel("Perte")
plt.legend()
plt.grid(True)

# 3. Nombre d'époques
# Piste : Doubler les époques permet de voir si on stagne (plateau) ou si on continue de descendre.
plt.subplot(3, 1, 3)
loss_epochs_50 = train_model(X_train_t, y_train_t, epochs=50)
loss_epochs_100 = train_model(X_train_t, y_train_t, epochs=100)
plt.plot(loss_epochs_50, label="50 Époques")
plt.plot(loss_epochs_100, label="100 Époques")
plt.title("Effet du Nombre d'Époques")
plt.xlabel("Époque")
plt.ylabel("Perte")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("comparaison_parametrage.png")
print("\nGraphiques sauvegardés sous 'comparaison_parametrage.png'")
plt.show()

"""
SYNTHÈSE DE L'EXERCICE :

1. Taux d'apprentissage :
   - lr = 0.001 : La courbe descend doucement et régulièrement.
   - lr = 1.0 : La perte explose ou oscille violemment (instabilité numérique).
   - Symptôme : "Explosion du gradient" -> Courbe qui monte brusquement ou NaN.

2. Batch size :
   - Taille 8 : Courbe plus "dentelée" (bruyante) car chaque mise à jour est basée sur peu de données.
   - Taille 64 : Courbe plus lisse et stable. L'entraînement est souvent plus rapide (mieux vectorisé).

3. Époques :
   - Avec 100 époques, on observe souvent un plateau : la perte ne descend plus,
     le modèle a atteint sa capacité maximale ou commence à overfitter.

4. Fonction de perte :
   - CrossEntropyLoss est la fonction adaptée pour la classification multi-classe.
   - Elle combine Softmax (pour transformer les sorties en probabilités) et l'entropie croisée.
"""

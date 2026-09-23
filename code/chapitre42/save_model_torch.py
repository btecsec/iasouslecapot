# ==============================================================================
# EXERCICE PRATIQUE : Sauvegarde et chargement PyTorch (Version Autonome)
# Objectif : Utiliser TorchScript pour sauvegarder l'architecture ET les poids
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import zipfile
import os
import shutil
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Définition de l'architecture (nécessaire pour l'entraînement)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(20, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def main():
    # NOTE : Le chargement de modèles provenant de sources non fiables peut
    # exécuter du code arbitraire. Utilisez uniquement vos propres fichiers.

    # 1. Préparation des données
    print("--- Étape 1 : Préparation des données ---")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test_scaled)

    # 2. Entraînement
    print("Entraînement du modèle PyTorch...")
    model = SimpleNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        original_pred = (model(X_test_t) > 0.5).int().numpy()

    # 3. Sauvegarde Autonome (TorchScript)
    print("\n--- Étape 2 : Sauvegarde ---")

    # A. On transforme le modèle en "ScriptModule"
    # Cela compile l'architecture et les poids dans un format indépendant du code Python
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "model_jit.pt")

    # B. Sauvegarde du scaler
    joblib.dump(scaler, "scaler_torch.joblib")

    # Archivage
    with zipfile.ZipFile("torch_autonomous_pack.zip", "w") as zf:
        zf.write("model_jit.pt")
        zf.write("scaler_torch.joblib")

    print("Pack autonome créé : torch_autonomous_pack.zip (TorchScript + Scaler)")

    # Nettoyage
    os.remove("model_jit.pt")
    os.remove("scaler_torch.joblib")

    # 4. Chargement et Vérification
    print("\n--- Étape 3 : Chargement et Vérification ---")

    with zipfile.ZipFile("torch_autonomous_pack.zip", "r") as zf:
        zf.extractall("temp_torch_reload")

    # 1. Recharger le scaler
    reloaded_scaler = joblib.load("temp_torch_reload/scaler_torch.joblib")

    # 2. Charger le modèle JIT
    # REMARQUE : On n'a plus besoin de faire "model = SimpleNet()" !
    # torch.jit.load reconstruit l'architecture tout seul depuis le fichier.
    reloaded_model = torch.jit.load("temp_torch_reload/model_jit.pt")
    reloaded_model.eval()

    # Test
    X_test_reloaded_scaled = reloaded_scaler.transform(X_test)
    X_test_reloaded_t = torch.FloatTensor(X_test_reloaded_scaled)
    with torch.no_grad():
        reloaded_pred = (reloaded_model(X_test_reloaded_t) > 0.5).int().numpy()

    np.testing.assert_array_equal(original_pred, reloaded_pred)
    print("✅ Succès : Modèle rechargé sans connaissance de la classe SimpleNet !")

    shutil.rmtree("temp_torch_reload")

if __name__ == "__main__":
    main()

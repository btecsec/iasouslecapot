from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from generate_model import generate_model

app = FastAPI()

# Modèle de données pour un manchot
class Manchot(BaseModel):
    longueur_bec: float
    profondeur_bec: float
    longueur_nageoire: float
    masse: float

# SÉCURITÉ : joblib.load utilise pickle. Ne charger que des fichiers de confiance.
# Un fichier malveillant peut exécuter du code arbitraire sur votre machine.
try:
    modele = joblib.load("modele_manchots.joblib")
except FileNotFoundError:
    print("Modèle non trouvé. Génération en cours...")
    generate_model()
    modele = joblib.load("modele_manchots.joblib")

@app.get("/")
def accueil():
    """Route d'accueil pour vérifier que l'API fonctionne."""
    return {"message": "Mon API de manchots fonctionne"}

@app.get("/sante")
def sante():
    """Route de santé pour vérifier si le service est vivant."""
    return {"statut": "ok"}

@app.post("/predire")
def predire(manchot: Manchot):
    """Route pour prédire l'espèce d'un manchot à partir de ses mesures."""
    if modele is None:
        return {"erreur": "Modèle non chargé sur le serveur"}

    # Transformer l'entrée Pydantic en DataFrame pandas
    donnees = pd.DataFrame([manchot.dict()])

    # Prédire l'espèce
    espece = modele.predict(donnees)[0]

    return {"espece": espece}

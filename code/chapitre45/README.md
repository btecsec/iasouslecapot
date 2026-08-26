# Chapitre 45 — Exposer son modèle avec une API

## L'énoncé

1. Créez `mon_api.py` avec la route d'accueil, lancez-le avec uvicorn.
2. Ajoutez le modèle Pydantic `Manchot` et la route `/predire`.
3. Ouvrez `/docs` et envoyez les mesures d'un manchot.
4. Envoyez du texte à la place d'un nombre. Comment l'API réagit-elle ?
5. **Défi** : ajoutez une route `/sante` qui renvoie `{"statut": "ok"}`.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `mon_api.py` | L'API : accueil, `/sante`, `/predire`. |
| `generate_model.py`, `prepare_data.py` | Entraînent le modèle au premier lancement s'il manque. |
| `modele_manchots.joblib` | Le modèle pré-entraîné. |
| `test_mon_api.py` | Les tests : routes, prédictions, validation, documentation. |

```bash
pip install -r requirements.txt
uvicorn mon_api:app --reload        # puis http://127.0.0.1:8000/docs
pytest -q
```

## Corrigé

### 1-2. L'API en trois blocs

```python
app = FastAPI()

class Manchot(BaseModel):              # 1. le contrat d'entrée
    longueur_bec: float
    profondeur_bec: float
    longueur_nageoire: float
    masse: float

modele = joblib.load("modele_manchots.joblib")   # 2. chargé UNE fois

@app.post("/predire")                  # 3. la route
def predire(manchot: Manchot):
    donnees = pd.DataFrame([manchot.dict()])
    return {"espece": modele.predict(donnees)[0]}
```

**Le chargement du modèle est hors de la fonction, et c'est essentiel.**
Placé à l'intérieur, il relirait le fichier à chaque requête : quelques
millisecondes de calcul pour plusieurs centaines de millisecondes de lecture
disque. Sur un modèle de langage de 2 Go, l'API deviendrait inutilisable.

### 3. Tester depuis `/docs`

FastAPI génère la documentation interactive à partir de vos annotations de
types — vous n'écrivez rien de plus. Envoyez :

```json
{"longueur_bec": 39.1, "profondeur_bec": 18.7,
 "longueur_nageoire": 181.0, "masse": 3750.0}
```

```json
{"espece": "Adelie"}
```

Bec court, nageoires courtes, masse moyenne : profil Adelie. Essayez ensuite
`217 mm` de nageoire et `5100 g` : vous obtiendrez `Gentoo`. Les deux cas sont
verrouillés par des tests.

### 4. Envoyer du texte à la place d'un nombre

```json
{"masse": "beaucoup", ...}
```

```text
422 Unprocessable Entity
{"detail": [{"loc": ["body", "masse"],
             "msg": "Input should be a valid number", ...}]}
```

Trois choses à remarquer :

- **Le code 422**, pas 500. Ce n'est pas un plantage du serveur, c'est un
  **refus poli** : la requête est mal formée.
- **L'erreur arrive avant votre code.** Pydantic valide *avant* d'entrer dans
  la fonction `predire`. Votre modèle ne voit jamais de données aberrantes —
  c'est une couche de protection gratuite.
- **Le message désigne le champ fautif** (`"loc": ["body", "masse"]`), ce qui
  rend le déboguage côté client immédiat.

Nuance utile : `"3750"` (un nombre écrit en texte) est **accepté** et converti.
Pydantic refuse l'ambigu, pas le convertible.

### 5. Défi : la route `/sante`

```python
@app.get("/sante")
def sante():
    return {"statut": "ok"}
```

Trois lignes, et pourtant indispensable en production. Cette route est le
*health check* : c'est elle que Docker, Kubernetes ou Cloud Run appellent toutes
les quelques secondes pour savoir si le conteneur est vivant. Sans réponse, la
plateforme le redémarre.

Bonne pratique pour la suite : faites-la répondre `ok` **seulement si le
modèle est chargé**. Un service qui répond « je vais bien » alors qu'il ne peut
rien prédire est pire qu'un service muet.

### Tester une API sans lancer de serveur

C'est le point que l'énoncé ne demande pas et qui change tout pour la suite
(chapitre 49) :

```python
from fastapi.testclient import TestClient
client = TestClient(app)

def test_sante():
    assert client.get("/sante").json() == {"statut": "ok"}
```

`TestClient` appelle l'application **en mémoire** : pas de port, pas de
processus, quelques millisecondes. C'est ce qui rend ces tests exécutables dans
une chaîne CI/CD.

### Une note sur les versions

Si vous voyez cet avertissement :

```text
PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump`
```

C'est que vous êtes en Pydantic v2. Le code fonctionne toujours ; pour le
moderniser, remplacez `manchot.dict()` par `manchot.model_dump()`.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Une API permet au monde extérieur d'envoyer des données et de recevoir des prédictions. |
| 2 | **b** | `uvicorn` est le serveur ASGI qui exécute l'application FastAPI. |
| 3 | **a** | Pour éviter de recharger le modèle inutilement à chaque requête. |
| 4 | **b** | Le modèle Pydantic décrit et vérifie les données d'entrée — et documente l'API au passage. |

## Ce qu'il faut retenir

Une API transforme un fichier `.joblib` en service utilisable par n'importe
qui. Chargez le modèle une fois, validez les entrées avec Pydantic, ajoutez une
route de santé — et testez le tout avec `TestClient`, sans jamais lancer de
serveur.

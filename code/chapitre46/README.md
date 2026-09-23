# Chapitre 46 — Emballer son application avec Docker

## L'énoncé

1. Vérifiez que vous avez un `requirements.txt` (fastapi, uvicorn,
   scikit-learn, joblib, pandas).
2. Écrivez le `Dockerfile` dans le dossier de votre API.
3. Construisez l'image. Combien de temps prend la construction ?
4. Lancez le conteneur et testez `/docs` dans le navigateur.
5. **Défi** : ajoutez un `.dockerignore` excluant les données brutes et les
   `__pycache__`. L'image est-elle plus légère ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `Dockerfile` | La recette de l'image. |
| `.dockerignore` | Ce qui ne doit jamais entrer dedans. |
| `mon_api.py`, `generate_model.py`, `prepare_data.py`, `modele_manchots.joblib` | L'application à emballer. |
| `test_dockerfile.py` | Les tests : ils relisent le Dockerfile, sans lancer Docker. |
| `pyproject.toml` | Isole le test lent (`-m docker`) du reste. |

```bash
pytest -q              # rapide : relit le Dockerfile (aucun Docker requis)
pytest -q -m docker    # lent : construit vraiment l'image

docker build -t api-manchots .
docker run -p 8000:8000 api-manchots      # puis http://localhost:8000/docs
```

## Corrigé

### 2. Le Dockerfile, ligne par ligne

```dockerfile
FROM python:3.11-slim          # 1. le point de départ

WORKDIR /app                   # 2. le dossier de travail dans l'image

COPY requirements.txt .        # 3. d'abord la liste des dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY . .                       # 4. ensuite seulement le code

CMD ["uvicorn", "mon_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Quatre décisions, et chacune évite un problème réel :

**`python:3.11-slim` plutôt que `python:latest`.** Une version épinglée
garantit que l'image de la semaine prochaine sera identique à celle
d'aujourd'hui — c'est tout l'intérêt de Docker. Et `-slim` divise la taille par
environ cinq (≈ 130 Mo au lieu de ≈ 1 Go).

**Les dépendances copiées *avant* le code.** C'est l'optimisation la plus
rentable du fichier. Docker met chaque instruction en cache et réutilise tout
ce qui n'a pas changé. Avec cet ordre :

| Vous modifiez | Docker refait |
|---|---|
| une ligne de `mon_api.py` | seulement la copie du code — **2 secondes** |
| `requirements.txt` | l'installation complète — 2 à 5 minutes |

Si `COPY . .` venait en premier, la moindre virgule changée relancerait
l'installation entière.

**`--no-cache-dir`.** Le cache de pip ne resservira jamais dans une image, et
il pèse des dizaines de mégaoctets.

**`--host 0.0.0.0`.** Le piège numéro un du chapitre. Avec `127.0.0.1`
(le défaut d'uvicorn), le serveur n'écoute que l'intérieur du conteneur :
`docker run -p 8000:8000` semble fonctionner, mais rien ne répond dans le
navigateur, **et aucune erreur ne s'affiche**. `0.0.0.0` signifie « écoute sur
toutes les interfaces », donc aussi sur celle que Docker publie.

### 3. Le temps de construction

- **Première fois** : 2 à 5 minutes. Docker télécharge l'image de base et
  installe scikit-learn, pandas, numpy, fastapi.
- **Ensuite, après un changement de code** : 2 à 5 secondes, grâce au cache
  des couches.

C'est exactement ce que l'ordre des instructions permet.

### 4. Lancer et tester

```bash
docker run -p 8000:8000 api-manchots
```

`-p 8000:8000` signifie **port de votre machine : port du conteneur**. Le
conteneur est une petite machine isolée ; sans cette publication, son port 8000
n'existe que pour lui.

Vérifiez ensuite `http://localhost:8000/docs`, puis `/sante`.

### 5. Défi : le `.dockerignore`

```text
__pycache__
data/
.git
.env
venv
```

Chaque ligne évite un problème distinct :

| Exclusion | Pourquoi |
|---|---|
| `__pycache__` | des `.pyc` compilés pour *votre* Python, inutiles et parfois nuisibles |
| `data/` | les données brutes n'ont rien à faire dans une image de service : des mégaoctets pour rien |
| `.git` | tout l'historique du dépôt — souvent plus lourd que le code, et il peut contenir d'anciens secrets |
| `.env` | **le pire** : vos clés d'API, copiées dans une image que vous allez pousser sur un registre |
| `venv` | des centaines de Mo de binaires compilés pour votre système, inutilisables dans l'image |

**L'image est-elle plus légère ?** Oui, souvent de façon spectaculaire :
un `venv` et un `.git` représentent facilement 500 Mo. Mais le gain le plus
important n'est pas la taille : c'est le `.env` qui ne partira pas en
production.

### Pourquoi tester un Dockerfile sans Docker

Construire une image prend des minutes et exige un démon Docker : ce n'est pas
un test unitaire (chapitre 9). Or les erreurs les plus fréquentes se lisent
dans le fichier :

- l'ordre des `COPY` (cache cassé) ;
- `127.0.0.1` au lieu de `0.0.0.0` (service muet) ;
- `:latest` au lieu d'une version (images non reproductibles) ;
- un secret écrit en dur.

`test_dockerfile.py` vérifie ces quatre points en 80 millisecondes, et garde
le vrai build derrière un marqueur `docker`, exécuté seulement à la demande.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Docker résout le « ça marche chez moi » : l'image embarque l'OS, Python, les bibliothèques et le code. |
| 2 | **b** | L'image est le plan figé ; le conteneur est une image en cours d'exécution. Une image, plusieurs conteneurs. |
| 3 | **b** | Le Dockerfile décrit, étape par étape, comment construire l'image. |
| 4 | **b** | Jamais dans l'image ni dans le code copié dedans : une image se télécharge et s'inspecte couche par couche. Les secrets se passent au lancement. |

## Ce qu'il faut retenir

Un Dockerfile est une recette lisible et versionnée de votre environnement
complet. Copiez les dépendances avant le code, écoutez sur `0.0.0.0`, épinglez
la version de base, et n'y mettez jamais de secret.

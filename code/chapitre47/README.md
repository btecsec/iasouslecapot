# Chapitre 47 — Déployer son modèle dans le cloud

Exercice de préparation : **pas de code à écrire ici** (le déploiement réel se
fait chez un fournisseur). Le corrigé du livre détaille la voie Google Cloud
Run ; ce dossier le complète par la méthode générale et les pièges de facture.

## Corrigé de l'exercice

### 1-2. Choisir un fournisseur, et repérer la marche à suivre

Les trois grands, et ce qu'il faut chercher dans leur documentation :

| Fournisseur | Service « conteneur sans serveur » | Registre d'images |
|---|---|---|
| Google Cloud | **Cloud Run** | Artifact Registry |
| AWS | App Runner (ou ECS Fargate) | ECR |
| Azure | Container Apps | ACR |

Il existe aussi des plateformes plus simples (Railway, Render, Fly.io,
Scaleway Serverless) qui suppriment 80 % des étapes ci-dessous. Pour un premier
déploiement, elles sont un excellent choix.

Le schéma est **toujours le même**, quel que soit le nom des commandes :

```text
image Docker locale → poussée dans un registre → service qui la fait tourner → URL publique
```

### 3. Les variables d'environnement à prévoir

Pour l'API de manchots des chapitres 45-45 :

| Variable | Rôle | Secrète ? |
|---|---|---|
| `PORT` | le port d'écoute — **imposé par la plateforme**, à lire, jamais à fixer en dur | non |
| `MODEL_PATH` | où trouver le `.joblib` | non |
| `LOG_LEVEL` | verbosité | non |
| `API_KEY` | authentifier les appelants | **oui** |
| `SENTRY_DSN` / clé de monitoring | remontée d'erreurs | **oui** |

Le piège n° 1 des premiers déploiements : coder `--port 8000` en dur dans le
`CMD`. Cloud Run (comme la plupart) injecte le port à respecter dans `$PORT` et
considère le service mort s'il n'écoute pas dessus.

```dockerfile
CMD exec uvicorn mon_api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Les secrets ne se mettent **jamais** dans l'image, ni dans le code, ni dans le
dépôt Git : ils se passent au lancement, ou via un gestionnaire de secrets
(Secret Manager, AWS Secrets Manager, Key Vault). Une image Docker se
télécharge et s'inspecte couche par couche — une clé copiée dedans est une clé
publiée.

### 4. Le plan de déploiement en cinq lignes

```text
1. Construire l'image et la tester en local (docker run -p 8000:8000 …).
2. Créer un dépôt d'images chez le fournisseur, s'y authentifier.
3. Étiqueter l'image avec le chemin du registre et une VERSION (jamais « latest »),
   puis la pousser.
4. Créer le service à partir de cette image : région, mémoire, variables
   d'environnement, HTTPS activé, mise à l'échelle min=0.
5. Tester l'URL publique (/sante puis /predire), brancher les journaux,
   puis supprimer les ressources si c'était un essai.
```

### 5. Le défi, avec Google Cloud Run

Les commandes du corrigé du livre, dans l'ordre :

```bash
gcloud init && gcloud auth login

gcloud artifacts repositories create my-app-repo \
  --repository-format=docker --location=us-central1

docker build -t us-central1-docker.pkg.dev/PROJECT_ID/my-app-repo/my-app:1.0.0 .
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/PROJECT_ID/my-app-repo/my-app:1.0.0

gcloud run deploy my-app-service \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/my-app-repo/my-app:1.0.0 \
  --platform=managed --region=us-central1 --allow-unauthenticated

gcloud run services describe my-app-service \
  --region=us-central1 --format="value(status.url)"
```

**Deux remarques que le lecteur pressé saute, et qu'il regrette :**

- `--allow-unauthenticated` rend votre API **publique sur Internet**. Pour un
  essai c'est pratique ; pour autre chose, ajoutez au minimum une clé d'API et
  une limitation de débit. Une API de modèle publique, c'est du calcul gratuit
  offert à qui la trouve.
- `--min-instances=0` (le défaut) fait tomber le service à zéro quand personne
  ne l'appelle : vous ne payez rien, mais le premier appel après une pause
  subit un **démarrage à froid** de plusieurs secondes — le temps de charger
  le modèle. C'est le compromis à connaître.

### Le nettoyage : la partie que personne ne lit

```bash
gcloud run services delete my-app-service --region=us-central1
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/PROJECT_ID/my-app-repo/my-app:1.0.0 --delete-tags
gcloud artifacts repositories delete my-app-repo --location=us-central1
```

Le service serverless coûte zéro au repos, mais **le stockage des images est
facturé en permanence**. Quelques images de 2 Go oubliées dans un registre,
et la facture arrive tous les mois pour rien.

**Le réflexe à prendre dès le premier essai** : créer un **budget avec
alerte** (5 €) sur le projet, avant même de déployer. Tous les fournisseurs
le proposent, et c'est gratuit.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le cloud, ce sont des ordinateurs loués chez un fournisseur, dans des centres de données. |
| 2 | **a** | Pour que le modèle soit disponible en permanence et accessible depuis partout. |
| 3 | **b** | Les secrets vont dans les variables d'environnement du service, jamais dans le code ni dans l'image. |
| 4 | **b** | HTTPS chiffre les échanges — sans lui, les données envoyées à votre modèle circulent en clair. |

## Ce qu'il faut retenir

Déployer, c'est pousser une image dans un registre et demander à un service de
la faire tourner. Le reste — secrets hors de l'image, HTTPS, versions
explicites, alerte de budget, nettoyage — n'est pas de la décoration : c'est ce
qui distingue une démo d'un service.

# Chapitre 44 — Du notebook à la production, qu'est-ce que le MLOps ?

Exercice de réflexion : **pas de code**.

## Corrigé de l'exercice

### 1. Trois obstacles pour qu'une autre personne utilise votre modèle

| Obstacle | Symptôme concret | Réponse du livre |
|---|---|---|
| **L'environnement** | « `ModuleNotFoundError: sklearn` » chez le collègue, ou pire : scikit-learn 1.2 chez lui, 1.5 chez vous, et le `.joblib` refuse de s'ouvrir. | `requirements.txt` (ch. 5) puis Docker (ch. 42) |
| **L'accès** | Le modèle vit dans un fichier `.joblib` sur votre disque. Personne d'autre ne peut l'appeler. | Une API (ch. 41), puis un hébergement (ch. 43) |
| **Le prétraitement** | Le collègue envoie des masses en kilos, vous aviez entraîné en grammes. Aucune erreur, des prédictions fausses. | Sauvegarder le scaler et l'ordre des colonnes (ch. 38), les enfermer dans un pipeline |

Un quatrième, souvent oublié : **la reproductibilité**. Sans `random_state`
fixé et sans version du dataset notée, vous ne saurez même pas reconstruire
votre propre modèle dans six mois.

### 2. La boucle du cycle de vie

```text
        ┌──────────────────────────────────────────────┐
        ↓                                              │
   développer  →  empaqueter  →  déployer  →  surveiller
   (ch. 26-39)     (ch. 42)      (ch. 43)      (ch. 44)
        ↑                                              │
        └────────── ré-entraîner (ch. 45) ─────────────┘
```

**Le chapitre 43 occupe entièrement la première case** — et rien d'autre. Tout
ce que vous avez fait en partie IV, aussi soigné soit-il, ne représente que le
premier quart du travail. C'est le message central du chapitre 44, et il
surprend souvent.

### 3. Deux raisons qu'un modèle se dégrade sans que son code ait changé

1. **La dérive des données** (*data drift*) — la forme des entrées change.
   Un capteur est remplacé et mesure 2 mm plus court ; une application mobile
   attire une nouvelle tranche d'âge ; un formulaire ajoute un champ. Le
   modèle reçoit des données qui ne ressemblent plus à celles de
   l'entraînement.
2. **La dérive du concept** (*concept drift*) — le lien entre entrée et
   réponse change. Les fraudeurs adoptent une nouvelle méthode ; une
   réglementation change les comportements d'achat ; une pandémie modifie les
   déplacements. Les entrées se ressemblent, mais la bonne réponse n'est plus
   la même.

Un troisième cas, très fréquent en pratique : le **pipeline en amont** casse.
Un champ passe de `"3200"` à `"3200 g"` et votre conversion silencieuse produit
des `NaN`. Le modèle n'a rien fait de mal ; ses entrées sont devenues du bruit.

### 4. Défi : un modèle de recommandation de films, six mois plus tard

Les trois dégradations se cumulent :

- **Nouveaux films** : le catalogue s'est enrichi de centaines de titres que
  le modèle n'a jamais vus. Il ne les recommandera jamais — c'est le problème
  du *démarrage à froid*.
- **Nouveaux utilisateurs** : inscrits après l'entraînement, ils n'existent
  pas dans les données. Même problème.
- **Goûts qui changent** : saisons, actualité, effets de mode. Le lien
  « profil → film apprécié » se déplace : dérive du concept.
- **La boucle de rétroaction**, la plus insidieuse : le modèle recommande des
  films, les gens regardent ce qui est recommandé, et le prochain
  entraînement apprend sur ces données-là. Le modèle **crée** les données qui
  le confirmeront. Il se rétrécit sur lui-même, et personne ne voit rien dans
  les métriques — le taux de clic reste bon.

D'où la conclusion pratique : un système de recommandation se réentraîne
souvent (quotidiennement, parfois en continu), et se surveille sur des
métriques **de diversité**, pas seulement de clic.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le MLOps est l'ensemble des pratiques pour déployer et maintenir des modèles en production. |
| 2 | **b** | Un notebook n'est accessible à personne d'autre et ne vit que sur votre machine. |
| 3 | **a** | Parce qu'un modèle dépend des données, qui changent dans le temps — un logiciel classique, lui, fait la même chose indéfiniment. |
| 4 | **b** | C'est une boucle continue : déployer, surveiller, ré-entraîner. |

## Ce qu'il faut retenir

Un logiciel classique se dégrade quand on y touche ; un modèle se dégrade
**quand on n'y touche pas**, parce que le monde bouge. Le MLOps est la
discipline qui organise cette maintenance permanente.

# Chapitre 54 — Les bases de données vectorielles

## L'énoncé

1. Installez Chroma et créez une collection **persistante**, en distance
   cosinus.
2. Indexez les six morceaux du manuel avec leurs métadonnées (`rubrique`,
   `page`), puis affichez le contenu de la base. Combien de nombres par
   vecteur ?
3. Posez trois questions **par le sens**, sans réutiliser les mots du
   document. Relevez la distance du meilleur résultat.
4. Rejouez la question sur la batterie en filtrant `rubrique="batterie"`.
   Qu'est-ce que le filtre change ?
5. **Défi** : trouvez le seuil de distance qui laisse passer vos trois
   questions et rejette « quelle est la capitale de l'Australie ? ».

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `base_vectorielle.py` | Ouvrir, indexer, inspecter, chercher, filtrer, refuser. |
| `test_base_vectorielle.py` | 10 tests en 1,3 s — sans réseau ni modèle téléchargé. |

```bash
pip install -r requirements.txt
python base_vectorielle.py
pytest -q
```

Le premier lancement télécharge le modèle d'embeddings (environ 450 Mo), puis
il est réutilisé depuis le cache. Le dossier `ma_base/` créé à côté du script
**est** la base de données : supprimez-le et tout est reconstruit.

## Corrigé

### 1. Persistance et distance : deux réglages, pas un

```python
client = chromadb.PersistentClient(path="ma_base")
collection = client.get_or_create_collection(
    name="manuel_robotx",
    embedding_function=charger_embeddings(),
    metadata={"hnsw:space": "cosine"},
)
```

Deux pièges dans ces quatre lignes :

- `Client()` tout court garde tout **en mémoire** : à l'arrêt du script, la
  base disparaît et il faut tout réencoder. `PersistentClient` écrit dans un
  dossier.
- Sans `hnsw:space`, Chroma indexe en distance **euclidienne**. Sur des
  embeddings de texte on veut le **cosinus**, qui compare la direction des
  vecteurs sans se laisser influencer par leur longueur — un texte long ne
  doit pas paraître « plus loin » simplement parce qu'il est long.

### 2. Ce que contient la base

```text
3. collection "manuel_robotx"
   6 vecteurs, dimension 384, distance cosine

   id         rubrique   page  document (extrait)
   ---------  ---------  ----  --------------------------
   manuel-01  securite      4  Pour reinitialiser le...
   manuel-02  batterie      7  Le RobotX1000...
   manuel-03  batterie      8  La recharge complete de...

   manuel-01 vu de l'interieur (384 nombres) :
   [ -0.3023  +0.1851  -0.0966  ... ]
```

**384 nombres par morceau.** Chaque ligne stocke donc trois choses côte à
côte : le texte d'origine (pour le rendre au LLM), le vecteur (pour le
retrouver), les métadonnées (pour filtrer). C'est exactement ce trio qui
distingue une base vectorielle d'un simple index de mots-clés.

### 3. La recherche par le sens

| Question | Meilleur résultat | Distance |
|---|---|---|
| « j'ai oublie mon code d'acces » | `manuel-01` (mot de passe) | 0,6676 |
| « combien de temps tient la batterie ? » | `manuel-03` (recharge) | 0,2501 |
| « comment nettoyer l'appareil ? » | `manuel-04` (entretien) | 0,4820 |
| « quelle est la capitale de l'Australie ? » | `manuel-05` (au hasard) | 0,9456 |

La première ligne est la démonstration du chapitre : la question et le
document ne partagent **aucun mot** — « code d'accès » contre « mot de
passe » — et la base retrouve quand même le bon passage. Un `grep` aurait
rendu zéro résultat.

> **Le modèle doit être multilingue.** Avec le très populaire
> `all-MiniLM-L6-v2`, entraîné sur de l'anglais, la même question renvoie le
> paragraphe sur la **garantie** à 0,6118, devant celui du mot de passe. Le
> code est identique, les résultats sont faux. C'est l'erreur la plus coûteuse
> de ce chapitre.

### 4. Ce que change le filtre

```python
collection.query(query_texts=[question], n_results=2,
                 where={"rubrique": "batterie"})
```

Le filtre s'applique **avant** la comparaison des vecteurs : Chroma ne
regarde que les deux morceaux de la rubrique `batterie`. Sur six documents,
cela ne change pas le classement ; sur un million, cela change tout — c'est
la différence entre parcourir toute la base et n'en parcourir qu'un
millième. C'est aussi la seule façon propre de cloisonner les données de
deux clients dans une même collection.

### 5. Le seuil

Les distances mesurées se séparent en deux paquets nets :

```text
questions pertinentes : 0,2501   0,4820   0,6676
question hors sujet   : 0,9456
```

Tout seuil entre 0,70 et 0,90 fait l'affaire ; le corrigé prend **0,85**,
à mi-chemin, pour laisser une marge des deux côtés. Ce nombre n'est pas
transposable : il dépend du modèle, de la langue et de la taille des
morceaux. Sur vos documents, refaites la mesure — une dizaine de questions
suffisent à voir les deux paquets.

Le garde-fou est ensuite trivial, et c'est la meilleure protection contre
l'hallucination :

```python
if not resultats or resultats[0].distance > SEUIL_DISTANCE:
    return []          # on n'appelle même pas le LLM
```

## Ce que les tests vérifient (et ce qu'ils ne vérifient pas)

Les 10 tests utilisent une **fonction d'embedding factice** : un sac de mots
normalisé, calculé en trois lignes. Ils tournent hors ligne, en 1,3 seconde,
et vérifient la tuyauterie — indexation, `upsert` sans doublon, métadonnées,
filtre, seuil, persistance après fermeture.

Aucun test ne vérifie que « code d'accès » retrouve « mot de passe » : cela
dépend des poids d'un modèle téléchargé, qui changent d'une version à
l'autre. Un test qui échoue pour cette raison n'apprend rien. C'est la règle
du chapitre 9, appliquée à la lettre.

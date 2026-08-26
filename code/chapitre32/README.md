# Chapitre 32 — Nettoyer et préparer les données

## L'énoncé

Reprenez « penguins » (chapitre 31) et nettoyez-le :

1. Comptez les valeurs manquantes par colonne.
2. Remplissez les colonnes numériques manquantes par leur médiane.
3. Supprimez les éventuels doublons.
4. Transformez `species` (et `island`) en variables numériques avec
   `get_dummies`.
5. Normalisez les colonnes numériques avec `StandardScaler`.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `clean_data.py` | Le script du chapitre. |
| `donnees.py` | Le chargement des manchots (avec repli hors ligne). |
| `exercice_nettoyage.py` | Le corrigé : une fonction par étape, plus la chaîne complète. |
| `test_exercice_nettoyage.py` | Les tests, y compris les pièges du nettoyage. |

```bash
pip install -r requirements.txt
python exercice_nettoyage.py
pytest -q
```

## Corrigé, étape par étape

### 1. Compter les manquants

```python
df.isnull().sum()
```

4 colonnes numériques à 2 trous, plus `sex` à 11 trous. (Analyse détaillée au
chapitre 31.)

### 2. Remplir par la médiane

```python
df["body_mass_g"] = df["body_mass_g"].fillna(df["body_mass_g"].median())
```

**Pourquoi la médiane et non la moyenne ?** Parce qu'une seule valeur
aberrante suffit à déplacer la moyenne, jamais la médiane. Une masse saisie à
45 000 g au lieu de 4 500 g (une frappe en trop) tirerait la moyenne vers le
haut, et vous rempliriez les trous avec une valeur fausse. Le test
`test_la_mediane_resiste_a_une_valeur_aberrante` le démontre en trois lignes.

**Le piège que l'énoncé contient** : `df.fillna(df["age"].median())` — la
forme du quiz — remplit **toutes** les colonnes avec la médiane d'**une**
colonne. Remplissez colonne par colonne.

**Et pour `sex` ?** La médiane n'existe pas sur du texte. Trois options :
le mode (valeur la plus fréquente), une catégorie « inconnu », ou supprimer les
lignes. Ici le sexe n'est pas indispensable pour prédire l'espèce : le
supprimer serait aussi défendable.

### 3. Supprimer les doublons

```python
df = df.drop_duplicates()
```

Sur ce dataset : **zéro doublon**. La ligne reste utile — elle documente le
fait que vous avez vérifié. Un doublon n'est pas seulement inutile : il donne
un poids double à un exemple pendant l'entraînement, sans raison.

### 4. Encoder les catégories

```python
df = pd.get_dummies(df, columns=["island", "sex"], dtype=int)
```

On passe de 7 à 10 colonnes :

```text
species | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g
        | island_Biscoe | island_Dream | island_Torgersen
        | sex_Female | sex_Male
```

**Pourquoi pas simplement `Biscoe=1, Dream=2, Torgersen=3` ?** Parce que le
modèle prendrait ces nombres au sérieux : il en déduirait que Torgersen est
« trois fois » Biscoe, et que Dream est « entre les deux ». Ces relations
n'existent pas. Le one-hot supprime tout ordre artificiel.

**Attention à la cible.** L'énoncé dit « transformez `species` », mais dans une
chaîne d'entraînement on ne l'encode pas en one-hot : c'est la réponse à
prédire, et Scikit-learn l'accepte telle quelle (il encode lui-même en
interne). Le corrigé garde donc `species` intacte — c'est le comportement testé
par `test_lencodage_ne_touche_pas_la_cible`.

### 5. Normaliser

```python
scaler = StandardScaler()
df[colonnes] = scaler.fit_transform(df[colonnes])
```

Après quoi chaque colonne a une moyenne de 0 et un écart-type de 1. Le scaler
retient les valeurs d'origine :

```text
moyennes memorisees : [43.9  17.2  200.9  4200.9]
```

**Ces quatre nombres sont aussi précieux que le modèle lui-même.** Le jour où
votre API reçoit un manchot (chapitre 45), elle devra lui appliquer
*exactement* la même transformation. Un scaler oublié, et toutes les
prédictions sont fausses sans qu'aucune erreur ne s'affiche — c'est la
question 4 du quiz du chapitre 42.

### L'ordre des opérations, et l'erreur à ne pas commettre

```text
remplir → dédoublonner → encoder → normaliser
```

Mais surtout, **la normalisation vient après le découpage train/test**
(chapitre 33). Calculer la moyenne sur l'ensemble des données ferait entrer
une information du test dans l'entraînement : c'est une fuite de données. La
solution propre est le `Pipeline` du chapitre 35.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Elle remplit les trous par la médiane — attention, appliquée à `df` entier, elle utilise la médiane d'une seule colonne partout. |
| 2 | **b** | Pour éviter d'inventer un ordre entre des catégories qui n'en ont pas. |
| 3 | **b** | Pour mettre les variables sur une échelle comparable. |
| 4 | **b** | La qualité du modèle ne dépasse jamais celle des données : aucun algorithme ne rattrape des données fausses. |

## Ce qu'il faut retenir

Nettoyer, c'est prendre des décisions traçables : quelle valeur pour un trou,
quel encodage pour une catégorie, quelle échelle pour un nombre. Gardez le
`scaler` et l'encodeur avec le modèle — ils font partie du modèle.

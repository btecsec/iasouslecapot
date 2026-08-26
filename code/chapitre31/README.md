# Chapitre 31 — Récupérer et explorer un jeu de données

## L'énoncé

Chargez le dataset « penguins » de Seaborn, puis répondez **par du code** :

1. Combien de lignes et de colonnes ?
2. Quelles colonnes ont des valeurs manquantes, et combien ?
3. Combien d'espèces, et sont-elles équilibrées ?
4. Tracez la distribution de la masse (`body_mass_g`).
5. Notez au moins un problème à corriger avant l'entraînement.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `explore_data.py` | Le script du chapitre : la visite guidée, en affichage direct. |
| `donnees.py` | Le chargement des manchots, avec repli hors ligne. Réutilisé par toute la partie IV. |
| `exercice_exploration.py` | Le corrigé : une fonction testable par question. |
| `test_exercice_exploration.py` | Les tests, dont un qui compare le jeu de secours au vrai dataset. |

```bash
pip install -r requirements.txt
python explore_data.py
python exercice_exploration.py
pytest -q
```

## Corrigé, question par question

### 1. Dimensions

```python
df.shape        # (344, 7)
```

**344 manchots, 7 colonnes** : 4 mesures numériques, 3 colonnes textuelles
(espèce, île, sexe). C'est un tout petit dataset — et c'est une bonne
nouvelle : chaque expérience prendra une seconde, ce qui autorise à essayer
beaucoup de choses.

### 2. Valeurs manquantes

```python
df.isnull().sum()
```

| Colonne | Trous |
|---|---|
| `bill_length_mm` | 2 |
| `bill_depth_mm` | 2 |
| `flipper_length_mm` | 2 |
| `body_mass_g` | 2 |
| `sex` | 11 |

Regardez bien : ce sont **les deux mêmes lignes** qui manquent partout sur les
quatre mesures. Deux manchots n'ont tout simplement pas été mesurés. Onze
autres n'ont pas de sexe renseigné — probablement indéterminable visuellement.

Cette lecture change la décision du chapitre 32 : supprimer 2 lignes sur 344
est indolore ; supprimer les 11 (soit 3 %) l'est moins, surtout si le sexe ne
sert pas au modèle.

### 3. Équilibre des espèces

```python
df["species"].value_counts()
```

```text
Adelie       152      (44 %)
Gentoo       124      (36 %)
Chinstrap     68      (20 %)
```

**Léger déséquilibre** : la classe majoritaire vaut 2,2 fois la minoritaire.
Ce n'est pas grave (on ne parle pas de fraude bancaire à 0,1 %), mais cela a
une conséquence immédiate : un modèle stupide qui répondrait toujours
« Adelie » obtiendrait déjà **44 % d'accuracy**. C'est votre vrai plancher, et
c'est pourquoi le chapitre 40 insistera sur le rappel par classe.

Conséquence pour le chapitre 33 : il faudra découper avec `stratify=y` pour
conserver ces proportions dans le train et le test.

### 4. Distribution de la masse

```python
sns.histplot(df["body_mass_g"], bins=20)
```

La forme est **bimodale** : une bosse vers 3 700 g et une autre vers 5 000 g.
Ce n'est pas un défaut des données, c'est de l'information : les Gentoo sont
nettement plus lourds que les deux autres espèces. Autrement dit, **la masse
seule sépare déjà partiellement les espèces** — le futur modèle s'appuiera
dessus, et un score élevé au chapitre 35 n'aura rien de miraculeux.

### 5. Problèmes à corriger avant l'entraînement

1. **Valeurs manquantes** (2 + 11) — à supprimer ou à imputer (ch. 28).
2. **Colonnes textuelles** (`species`, `island`, `sex`) — un modèle ne mange
   que des nombres : one-hot encoding (ch. 28).
3. **Échelles incomparables** — la masse est en milliers de grammes, le bec en
   dizaines de millimètres. Sans normalisation, la masse écrase tout dans les
   calculs de distance (ch. 21 et 28).
4. **Fuite potentielle** : `species` est la cible, elle ne doit jamais se
   retrouver dans X (ch. 29).

## Sur le jeu de secours

`donnees.py` télécharge le vrai dataset ; s'il n'y parvient pas, il fabrique un
jeu **synthétique** de même forme (344 lignes, mêmes colonnes, mêmes effectifs
par espèce, mêmes trous). Un test compare les deux et échoue si le second
dérive du premier. C'est le motif standard pour un exercice de livre : personne
ne doit être bloqué par un pare-feu.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `df.head()` affiche les 5 premières lignes. |
| 2 | **b** | `df.isnull().sum()` compte les valeurs manquantes par colonne. |
| 3 | **b** | EDA = *Exploratory Data Analysis*, l'analyse exploratoire des données. |
| 4 | **b** | Comprendre les données et diagnostiquer les problèmes — pas les corriger : c'est le chapitre suivant. |

## Ce qu'il faut retenir

Explorer, ce n'est pas afficher des tableaux : c'est produire une **liste de
décisions à prendre**. À la fin d'une bonne exploration, vous savez déjà ce que
fera votre script de nettoyage.

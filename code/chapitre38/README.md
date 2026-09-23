# Chapitre 38 — Comprendre l'entraînement

## L'énoncé

1. Entraînez avec un taux d'apprentissage de 0.001, puis de 1.0. Que devient la
   courbe de perte ?
2. Changez le `batch_size` (8, puis 64). Plus stable, ou plus rapide ?
3. Doublez le nombre d'époques. La perte baisse-t-elle encore ?
4. Pour votre tâche (classification), quelle fonction de perte est adaptée ?
5. Reliez chaque symptôme du tableau à une courbe observée.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `train_parametrage.py` | Le script du chapitre. |
| `exercice_entrainement.py` | Le corrigé : les trois expériences, plus un lecteur de courbes automatique. |
| `test_exercice_entrainement.py` | Les tests, dont la preuve chiffrée de l'explosion à `lr=1.0`. |

```bash
pip install -r requirements.txt
python exercice_entrainement.py --png
pytest -q
```

Le corrigé utilise le `MLPClassifier` de Scikit-learn : un vrai réseau, mais
qui expose directement les trois boutons du chapitre et conserve sa courbe dans
`loss_curve_`. Chaque expérience prend deux secondes au lieu de deux minutes.

## Corrigé

### 1. Le taux d'apprentissage

```text
lr=0.001   perte finale     0.0151   descente saine
lr=0.01    perte finale     0.0024   descente bruitee
lr=1.0     perte finale     0.4018   explosion puis recuperation
```

**Ce qui se passe à `lr=1.0`** est plus intéressant que le simple « ça ne
marche pas ». Regardez le **début** de la courbe : la perte démarre à **9,7**,
alors qu'un modèle non entraîné sur 3 classes doit démarrer à
`-log(1/3) ≈ 1,0986`.

Autrement dit, les tout premiers pas — trop grands — ont projeté les poids
très loin, et le modèle a passé le reste de l'entraînement à réparer les
dégâts. Il finit à 0,40 : **160 fois pire** que `lr=0.01`, alors qu'il a vu
exactement les mêmes données.

C'est le repère à retenir : **votre perte de départ doit valoir environ
`log(nombre de classes)`**. Si elle est beaucoup plus haute, votre taux est
trop grand ou vos données sont mal normalisées. (Sur un modèle de langage à
2000 jetons de vocabulaire, le même contrôle donne `log(2000) ≈ 7,6`.)

**Et `lr=0.001` ?** Il descend proprement mais finit à 0,0151, soit 6 fois plus
haut que `lr=0.01` au même nombre d'époques : trop prudent, il n'a pas fini
d'apprendre. Le bon taux se trouve toujours entre « trop lent » et « ça
explose », et se cherche par essais — d'où le chapitre 41.

### 2. La taille de batch

```text
batch=8    perte finale 0.0048   irregularite 0.0627
batch=32   perte finale 0.0030   irregularite 0.0510
batch=64   perte finale 0.0044   irregularite 0.0396
```

**Réponse : plus stable, pas plus rapide.** L'irrégularité (l'écart-type des
variations d'une époque à l'autre) baisse de 37 % en passant de 8 à 64. La
raison est simple : un gros lot moyenne davantage d'exemples, donc son
gradient est une meilleure estimation de la « vraie » direction.

Les pertes finales, elles, se valent. Le batch n'est pas un bouton de qualité,
c'est un arbitrage :

| Petit batch (8) | Gros batch (64+) |
|---|---|
| descente bruitée | descente lisse |
| peu de mémoire | beaucoup de mémoire (le mur des GPU) |
| le bruit aide parfois à s'échapper d'un mauvais creux | plus efficace sur GPU (calcul parallèle) |

### 3. Doubler les époques

```text
50 epoques  : 0.0061
100 epoques : 0.0030
gain        : 0.0032
```

La perte **baisse encore**, mais de miettes : on est sur le **plateau**. Deux
fois plus de calcul pour 0,003 de perte en moins.

Et surtout : cette perte-là est celle de l'**entraînement**. Continuer à la
faire baisser après le plateau, c'est exactement ce qui produit du
surapprentissage — le sujet du chapitre 39. La bonne pratique est l'**arrêt
anticipé** (*early stopping*) : on surveille la perte de **validation** et on
s'arrête quand elle cesse de baisser.

### 4. Quelle fonction de perte ?

**L'entropie croisée** (*cross-entropy*, ou *log-loss*).

| Tâche | Perte | Pourquoi |
|---|---|---|
| Classification multi-classes | entropie croisée | punit la confiance mal placée : `-log(0.1) = 2,3` |
| Classification binaire | entropie croisée binaire | même principe, deux classes |
| Régression | MSE ou MAE | l'écart au carré, ou l'écart absolu |

L'accuracy ne peut **pas** servir de fonction de perte : elle ne change que par
sauts (une prédiction bascule ou non), donc son gradient est nul presque
partout. Il n'y a rien à descendre. L'entropie croisée, elle, varie de façon
continue avec la confiance du modèle — c'est ce qui la rend dérivable, donc
optimisable.

### 5. Le tableau symptôme → diagnostic

La fonction `profil_courbe()` du corrigé encode ce tableau, et les tests le
vérifient sur des courbes fabriquées :

| Ce que vous voyez | Nom | Cause probable | Remède |
|---|---|---|---|
| La perte monte ou reste plate | **divergence** | taux beaucoup trop grand | diviser le taux par 10 |
| Départ à 9,7 au lieu de 1,1 | **explosion puis récupération** | taux trop grand, ou données non normalisées | baisser le taux, vérifier le scaler |
| Elle baisse à peine | **stagnation** | taux trop petit, ou modèle trop simple | augmenter le taux ou la capacité |
| Elle descend en dents de scie | **descente bruitée** | mini-batchs — **c'est normal** | augmenter le batch si ça gêne |
| Elle descend puis s'aplatit | **descente saine** | rien à signaler | surveiller la validation |

Notez la nuance de l'avant-dernière ligne : une courbe irrégulière n'est pas un
problème en soi. Avec des mini-batchs, environ une époque sur deux remonte
légèrement. Ce qui compte est la **tendance**, pas chaque point.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | La perte mesure de combien le modèle se trompe. |
| 2 | **b** | Une époque est un passage complet sur tout le dataset. |
| 3 | **b** | L'optimiseur décide comment ajuster les poids à partir des gradients. |
| 4 | **b** | Avec un taux trop grand, la perte explose ou oscille sans baisser. |

## Ce qu'il faut retenir

Trois boutons : le taux d'apprentissage (le plus important, de loin), la taille
de batch (stabilité contre mémoire), le nombre d'époques (à arrêter quand la
validation cesse de progresser). Et un repère qui vaut tous les diagnostics :
votre perte de départ doit valoir `log(nombre de classes)`.

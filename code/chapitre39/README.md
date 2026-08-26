# Chapitre 39 — Le combat contre le surapprentissage

## L'énoncé

1. Entraînez volontairement un réseau trop gros sur peu de données et beaucoup
   d'époques. Observez l'écart entraînement / validation.
2. Diagnostiquez : sous-apprentissage ou surapprentissage ?
3. Ajoutez `Dropout(0.3)` et réentraînez. L'écart se réduit-il ?
4. Réduisez le nombre de neurones et comparez.
5. Décrivez, en une phrase, l'étudiant qui correspond à chaque cas.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `surapprentissage.py` | Le script du chapitre. |
| `exercice_surapprentissage.py` | Le corrigé : quatre configurations comparées. |
| `test_exercice_surapprentissage.py` | Les tests (≈ 45 s : ils entraînent vraiment trois réseaux). |

```bash
pip install -r requirements.txt
python exercice_surapprentissage.py --png
pytest -q
```

## Le protocole : comment forcer le surapprentissage

Les manchots sont un jeu **trop propre** pour surapprendre : quel que soit le
réseau, validation et entraînement atteignent 100 % ensemble. Il faut donc
créer les conditions du problème, et elles sont au nombre de trois :

1. **40 exemples d'entraînement** seulement ;
2. **10 étiquettes fausses sur ces 40** — un manchot Adelie déclaré Gentoo.
   Ce n'est pas une triche : tout dataset réel contient des erreurs
   d'annotation. C'est même la définition du bruit qu'un modèle ne devrait
   *pas* apprendre ;
3. **200 époques**, largement au-delà du nécessaire.

La validation, elle, garde 100 exemples **corrects**. Un modèle qui apprend
les 10 erreurs par cœur sera donc forcément puni sur la validation : c'est
exactement le phénomène qu'on veut rendre visible.

## Corrigé

### 1-2. Le gros réseau seul

```text
1. gros reseau (256)     67843 params | train 1.000 | val 0.760 | ecart +0.240
                         sommet de validation 0.920 a l'epoque 5
```

**Diagnostic : surapprentissage**, sans ambiguïté.

- `train 1.000` : le réseau classe parfaitement ses 40 exemples — **y compris
  les 10 faux**. Il n'a pas appris à reconnaître des manchots, il a appris
  cette liste-là.
- `val 0.760` : sur des exemples corrects jamais vus, il chute de 24 points.

Le rappel du contraire, pour bien fixer les deux notions :

| | Entraînement | Validation | Diagnostic |
|---|---|---|---|
| Trop simple ou trop court | 0,35 | 0,34 | **sous-apprentissage** |
| Trop gros, trop longtemps | 1,00 | 0,76 | **surapprentissage** |
| Le bon réglage | 0,85 | 0,89 | **équilibre** |

Le sous-apprentissage est mauvais **partout** ; le surapprentissage est
excellent chez lui et mauvais dehors. Un seul chiffre ne permet jamais de
trancher : il en faut deux.

### Le chiffre le plus instructif du tableau

```text
sommet de validation 0.920 a l'epoque 5
```

Le modèle atteignait **92 % dès la 5ᵉ époque**, puis il s'est dégradé jusqu'à
76 % en continuant de « progresser » sur l'entraînement. Les 195 époques
suivantes ont donc activement **détruit** 16 points de performance réelle.

C'est l'argument massue en faveur de l'**arrêt anticipé** :

```python
arret = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True
)
modele.fit(..., callbacks=[arret])
```

`restore_best_weights=True` est la clé : sans lui, on garde les poids finaux,
c'est-à-dire les mauvais.

### 3. Avec du dropout

```text
3. gros + Dropout(0.3)   67843 params | train 0.950 | val 0.790 | ecart +0.160
3bis. gros + Dropout(0.5) 67843 params | train 0.925 | val 0.820 | ecart +0.105
```

**Oui, l'écart se réduit** : de 0,240 à 0,160 avec 0,3, puis à 0,105 avec 0,5.

Notez le mécanisme : l'accuracy d'**entraînement baisse** (1,000 → 0,925) et
celle de validation **monte** (0,760 → 0,820). C'est le principe même du
dropout — on handicape volontairement le réseau pendant l'apprentissage. En
éteignant au hasard une partie des neurones à chaque passage, on l'empêche de
compter sur une combinaison précise de neurones pour mémoriser un exemple.

Deux détails à retenir :

- **Le dropout n'ajoute aucun paramètre** (67 843 dans les deux cas) : ce
  n'est pas une couche qui apprend, c'est un interrupteur aléatoire.
- **Il n'agit qu'à l'entraînement.** En prédiction, tous les neurones sont
  actifs — c'est Keras qui gère la bascule, et en PyTorch c'est `modele.eval()`
  qui s'en charge (chapitre 37).

### 4. Avec un petit réseau

```text
4. petit reseau (8)        139 params | train 0.850 | val 0.890 | ecart -0.040
```

**Le meilleur des quatre**, avec 490 fois moins de paramètres.

Regardez `train 0.850` : le petit réseau n'atteint *pas* 100 % sur son jeu
d'entraînement — et c'est précisément pour cela qu'il est bon. Avec 139
paramètres, il n'a pas la capacité de mémoriser 10 exceptions arbitraires ; il
est donc contraint d'apprendre la règle générale. Son écart est même
légèrement **négatif** : il fait mieux dehors que chez lui.

**La conclusion pratique**, et elle est contre-intuitive : quand un modèle
surapprend, le premier réflexe n'est pas d'ajouter de la régularisation, c'est
de se demander s'il n'est pas simplement **trop gros pour la quantité de
données disponible**.

### Le classement des remèdes

| Remède | Effet ici | Coût |
|---|---|---|
| **Plus de données** | le plus efficace, toujours | souvent impossible |
| **Réduire le modèle** | écart 0,240 → −0,040 | rien, et le modèle est 490× plus léger |
| **Dropout 0,5** | écart 0,240 → 0,105 | un hyperparamètre à régler |
| **Arrêt anticipé** | aurait gardé 0,920 au lieu de 0,760 | trois lignes |
| Augmentation de données | non testé ici | dépend du domaine |

### 5. Les trois étudiants

| Cas | L'étudiant |
|---|---|
| **Surapprentissage** | Celui qui a appris les annales par cœur, corrigés compris — y compris leurs coquilles. Il récite parfaitement, et sèche dès qu'on change l'énoncé. |
| **Sous-apprentissage** | Celui qui n'a pas ouvert le cours. Il échoue aux annales comme à l'examen. |
| **Équilibre** | Celui qui a compris la méthode. Il fait quelques erreurs d'inattention sur les annales, et réussit sur un sujet inédit. |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le surapprentissage, c'est mémoriser l'entraînement et mal généraliser. |
| 2 | **b** | Excellent à l'entraînement, mauvais en validation. |
| 3 | **b** | Le dropout éteint au hasard une partie des neurones pendant l'entraînement. |
| 4 | **b** | Ajouter plus de données variées reste le remède le plus efficace — tous les autres ne font que limiter les dégâts. |

## Ce qu'il faut retenir

Deux chiffres, jamais un seul : entraînement **et** validation. Quand l'écart
se creuse, le modèle mémorise. Et avant d'empiler les régularisations,
demandez-vous s'il n'est pas simplement trop gros pour vos données.

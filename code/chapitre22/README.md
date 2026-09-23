# Chapitre 22 — Dans le moteur du Transformer

## L'énoncé

Reprenez « Le chat dort » et ses trois embeddings, et calculez l'attention du
mot **« dort »** (et non plus « chat »).

1. Les trois produits scalaires `dort·Le`, `dort·chat`, `dort·dort`.
2. Divisez chacun par √2 ≈ 1.41.
3. Vérifiez que le softmax donne environ 26,4 %, 33,1 % et 40,4 %.
4. Multipliez chaque embedding par son poids, additionnez, comparez à
   `[0.42, 0.48]`.
5. Réflexion : sur quel mot « dort » porte-t-il le plus d'attention, en dehors
   de lui-même ? Est-ce cohérent ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_attention.py` | L'attention refaite à la main, puis en matrices — les deux doivent coïncider. |
| `test_exercice_attention.py` | Les tests : tous les chiffres du corrigé, plus les propriétés de la softmax. |
| `requirements.txt` | numpy, pytest. |

```bash
python exercice_attention.py
pytest -q
```

## Corrigé, chiffre par chiffre

Rappel des embeddings :

```text
Le   = [0.1, 0.2]      chat = [0.7, 0.3]      dort = [0.4, 0.8]
```

### 1. Les produits scalaires (Q = « dort »)

```text
dort · Le   = (0.4 × 0.1) + (0.8 × 0.2) = 0.04 + 0.16 = 0.20
dort · chat = (0.4 × 0.7) + (0.8 × 0.3) = 0.28 + 0.24 = 0.52
dort · dort = (0.4 × 0.4) + (0.8 × 0.8) = 0.16 + 0.64 = 0.80
```

### 2. Mise à l'échelle par √dk = √2 ≈ 1.41

```text
Le   : 0.20 / 1.41421… = 0.141
chat : 0.52 / 1.41421… = 0.368
dort : 0.80 / 1.41421… = 0.566
```

> Les chiffres ci-dessus utilisent √2 **exact**. Si vous divisez par 1,41 tout
> rond, vous obtiendrez 0,567 au lieu de 0,566 sur la dernière ligne : c'est
> l'arrondi, pas une erreur de votre part. Le test
> `test_larrondi_de_la_racine_change_le_troisieme_chiffre` le vérifie.

**L'exception instructive.** Si vous refaites le calcul pour « Le », vous
verrez que `Le·Le = 0.05` est **plus petit** que `Le·dort = 0.20` : un mot n'est
pas toujours son propre plus proche voisin ! La raison : le produit scalaire
mélange direction *et* longueur, et `[0.1, 0.2]` est un vecteur très court.
C'est exactement pourquoi un vrai Transformer n'utilise pas les embeddings tels
quels, mais trois **projections apprises** Q, K et V — ici, on s'en passe pour
que l'arithmétique tienne sur une feuille.

### 3. Softmax

```text
Le   : 26.4 %
chat : 33.2 %      (le livre arrondit à 33,1 % ; l'écart vient de l'arrondi
dort : 40.4 %       de √2 à 1.41 au lieu de 1.41421…)
```

Somme : 100 %, comme toujours avec une softmax.

### 4. Le mélange final

```text
0.264 × [0.1, 0.2] = [0.0264, 0.0528]
0.332 × [0.7, 0.3] = [0.2322, 0.0995]
0.404 × [0.4, 0.8] = [0.1615, 0.3231]
------------------------------------------
somme              = [0.4201, 0.4756]  ≈  [0.42, 0.48]
```

Conforme à l'énoncé.

### 5. Sur quel mot « dort » porte-t-il le plus d'attention ?

Sur **lui-même** (40,4 %), puis sur **« chat » (33,2 %)**, et enfin sur « Le »
(26,4 %).

Oui, c'est cohérent — et c'est même le plus intéressant de l'exercice. « Le »
est un déterminant : il n'apporte presque aucune information sémantique, et
l'attention le relègue en dernier **sans qu'on le lui ait dit**. « chat », en
revanche, est le sujet du verbe : le lien grammatical le plus fort de la phrase
est aussi celui que l'attention pondère le plus. Le modèle n'a aucune notion de
grammaire ; la géométrie des embeddings suffit à faire émerger la relation.

### La comparaison qui éclaire tout

| Vue par | Attention à « Le » | à « chat » | à « dort » | Sortie |
|---|---|---|---|---|
| « chat » (chapitre) | 27,1 % | 37,2 % | 35,7 % | `[0.43, 0.45]` |
| « dort » (exercice) | 26,4 % | 33,2 % | 40,4 % | `[0.42, 0.48]` |

Chaque mot obtient **sa propre** distribution d'attention et **son propre**
vecteur enrichi. C'est pour cela qu'on parle d'embeddings *contextuels* : le
même mot dans une autre phrase ressortirait avec un vecteur différent.

Et un détail qui n'en est pas un : « chat » regarde « dort » à 35,7 %, tandis
que « dort » regarde « chat » à 33,2 %. **L'attention n'est pas symétrique**,
car chaque mot est comparé aux autres depuis sa propre position.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le produit scalaire Q·K mesure la similarité de deux mots dans l'espace des embeddings. |
| 2 | **b** | La division par √dk empêche les scores de devenir trop grands, ce qui saturerait le softmax (gradients quasi nuls, plus d'apprentissage). |
| 3 | **b** | Le softmax produit les poids d'attention, dont la somme fait 1. |
| 4 | **b** | Transposer K permet d'obtenir tous les produits scalaires d'un coup par une seule multiplication matricielle — c'est ce qui rend le calcul parallélisable sur GPU. |
| 5 | **b** | Il manque l'encodage de position, le multi-têtes, les connexions résiduelles, la normalisation et le feed-forward : un bloc Transformer complet les empile tous. |

## Ce qu'il faut retenir

L'attention est un calcul de moyenne pondérée, où les poids sont des
similarités. Rien de plus. Ce que vous venez de faire à la main sur trois mots,
un GPU le fait sur des milliers de jetons et des dizaines de couches — mais
c'est exactement la même arithmétique.

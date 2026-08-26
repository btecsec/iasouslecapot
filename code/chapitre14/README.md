# Chapitre 14 — Comment une machine « apprend » ?

Exercice de pensée : **pas de code** (le code arrive au chapitre 38, où vous
ferez varier ces réglages pour de vrai).

## Corrigé de l'exercice

### 1. « Plus chaud, plus froid » : qui joue le rôle du signal ?

**Le gradient.** Le signal « plus chaud » ne vous donne pas la position de
l'objet : il vous dit seulement dans quelle **direction** vous rapprocher.
C'est exactement ce qu'est un gradient — une direction et une intensité, pas
une solution.

Deux conséquences que l'analogie rend évidentes :

- il faut **bouger** pour avoir un nouveau signal (d'où l'itération) ;
- si le signal est mesuré au mauvais endroit, on part dans le décor (d'où
  l'importance de la fonction de perte, qui définit ce que « chaud » veut dire).

### 2. L'ami qui fait des pas de géant et rate la porte

Hyperparamètre mal réglé : **le taux d'apprentissage** (*learning rate*), et
il est **trop grand**.

Le symptôme est caractéristique : la personne ne s'arrête jamais *à* la porte,
elle oscille de part et d'autre, parfois de plus en plus loin. Sur une courbe
de perte, cela donne une ligne qui zigzague, stagne haut, ou part vers le ciel
(vous le verrez en vrai au chapitre 38 avec `--lr 1.0`).

L'erreur inverse existe aussi : des pas de fourmi. L'apprentissage finit par
marcher, mais après un temps déraisonnable — et il peut s'enliser dans le
premier creux venu.

### 3. « Apprendre, c'est faire diminuer la perte »

Une reformulation possible :

> Le modèle commence par se tromper beaucoup. On mesure de combien il se
> trompe avec un seul nombre — la perte. Puis, encore et encore, on modifie
> légèrement ses réglages internes dans le sens qui fait baisser ce nombre.
> Quand il ne baisse plus, l'apprentissage est terminé.

Le point important est le mot **un seul nombre** : tout l'apprentissage
automatique consiste à transformer un objectif humain flou (« bien reconnaître
les chats ») en une quantité mesurable qu'un algorithme peut faire décroître.
Si vous vous trompez de mesure, le modèle optimisera consciencieusement… la
mauvaise chose.

### 4. Pourquoi partir de paramètres au hasard n'est pas un problème ?

Parce que la descente de gradient **corrige** le point de départ. Peu importe
où vous vous trouvez sur la piste de ski : suivre la pente vous fait descendre.
Le hasard initial ne fait que choisir votre point de départ, pas votre
destination.

Deux nuances de professionnel :

- Le hasard est **nécessaire** : si tous les poids valaient 0, tous les
  neurones d'une couche calculeraient la même chose et recevraient la même
  correction — ils resteraient identiques pour toujours (*symétrie*).
- Le point de départ n'est pas totalement neutre : sur un modèle profond, deux
  graines aléatoires donnent deux résultats légèrement différents. C'est
  pourquoi on fixe `random_state` / `torch.manual_seed` pour être reproductible
  (chapitre 43).

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Les paramètres (poids) sont les boutons de réglage internes ajustés par l'apprentissage. |
| 2 | **b** | La perte mesure de combien le modèle se trompe, en un seul nombre. |
| 3 | **b** | La descente de gradient ajuste les paramètres par petits pas pour réduire la perte. |
| 4 | **c** | Avec un taux trop grand, le modèle enjambe la solution : la perte oscille ou explose. |

## Ce qu'il faut retenir

Apprendre = mesurer l'erreur, calculer la pente, faire un petit pas, recommencer.
Tout le reste du livre — Scikit-learn, Keras, PyTorch, le fine-tuning d'un LLM —
n'est qu'une variation sur ces quatre gestes.

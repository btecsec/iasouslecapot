# Solution de l'exercice — Chapitre 8 : Astuces Python pour coder plus vite et mieux

> Le code complet et exécutable est dans `exercice_astuces.py`.
> Les durées ci-dessous ont été mesurées sur une machine de bureau ordinaire :
> ce sont les **écarts** entre les versions qui comptent, pas les valeurs absolues.

Rappel de l'énoncé : un million d'identifiants de commandes, dix mille identifiants
« déjà traités », et il faut produire la liste des commandes restant à traiter.

---

## 1. La version naïve, et pourquoi elle ne finit jamais

```python
def restantes_naif(commandes: list[int], deja: list[int]) -> list[int]:
    resultat = []
    for identifiant in commandes:
        if identifiant not in deja:      # deja est une LISTE
            resultat.append(identifiant)
    return resultat
```

Le piège est dans le `not in`. Sur une liste, Python doit relire les dix mille
éléments **à chaque tour de boucle**. Pour un million de commandes, cela fait
1 000 000 × 10 000 = **dix milliards** de comparaisons.

C'est pour cela que le corrigé mesure cette version sur un échantillon de 5 000
commandes, puis extrapole — savoir estimer un temps d'exécution plutôt que le
subir fait partie du métier :

```text
5,000 commandes : 1.062s
Extrapolation à 1,000,000 : 212s (~4 min)
```

Quatre minutes. Le café était mérité.

## 2. Un seul mot change : `set`

```python
def restantes_avec_set(commandes: list[int], deja: list[int]) -> list[int]:
    deja_set = set(deja)                 # la conversion coûte un seul passage
    resultat = []
    for identifiant in commandes:
        if identifiant not in deja_set:
            resultat.append(identifiant)
    return resultat
```

```text
1,000,000 commandes : 0.109s
Gain estimé : ~1943x
Commandes restantes : 990,000
```

**Presque deux mille fois plus rapide.** Et remarquez : le gain mesuré ici
dépasse largement le facteur ~237 du chapitre, parce que la recherche est répétée
un million de fois. Plus la boucle est longue, plus le mauvais choix de structure
coûte cher.

**Le détail qui tue :** la conversion `set(deja)` doit être faite **avant** la
boucle. Écrite à l'intérieur, elle serait refaite à chaque tour et le programme
deviendrait plus lent que la version naïve.

## 3. En une compréhension, avec annotations

```python
def restantes_comprehension(
    commandes: list[int], deja: list[int]
) -> list[int]:
    deja_set = set(deja)
    return [i for i in commandes if i not in deja_set]
```

```text
1,000,000 commandes : 0.088s
Même résultat que l'étape 2 : True
```

Trois lignes au lieu de six, et encore un peu plus rapide : la compréhension
évite un appel à `.append()` par élément. Les annotations `list[int]`, elles, ne
changent rien à la vitesse — elles servent à votre éditeur et à vos collègues.

## 4. Le générateur : mémoire constante

```python
from collections.abc import Iterator

def restantes_generateur(
    commandes: list[int], deja: list[int]
) -> Iterator[int]:
    deja_set = set(deja)
    for identifiant in commandes:
        if identifiant not in deja_set:
            yield identifiant
```

```text
Liste      :    8,448,728 octets
Générateur :          224 octets
Rapport    : ~37,718x plus léger
```

Huit mégaoctets contre deux cent vingt-quatre octets. Et ces 224 octets ne
bougeront pas, même sur un flux de dix milliards de commandes.

**La contrepartie, à connaître :** un générateur ne se parcourt **qu'une seule
fois**. Après un premier `for`, il est vide. Si vous avez besoin du résultat
plusieurs fois, gardez la liste ; si vous ne faites que traverser les données une
fois (compter, filtrer, écrire dans un fichier), prenez le générateur.

Le type de retour s'écrit `Iterator[int]`, pas `list[int]` : la fonction ne rend
pas une liste, elle rend un objet qui produit des entiers à la demande.

---

## Bonus

**Ruff.** Depuis le dossier :

```bash
pip install ruff
ruff format .        # met tout le fichier au format standard
ruff check .         # signale imports inutiles, variables mortes, etc.
```

**tqdm sur l'étape 1.** Enveloppez la boucle lente pour voir, en direct, à quel
point elle est lente — c'est très parlant :

```python
from tqdm import tqdm

for identifiant in tqdm(commandes, desc="Version naïve"):
    if identifiant not in deja:
        resultat.append(identifiant)
```

La barre affiche un débit en « it/s » : comparez-le à celui de la version `set`,
l'écart saute aux yeux bien mieux qu'un chronomètre.

---

## Ce qu'il fallait voir

| Version | Durée sur 1 000 000 | Mémoire du résultat |
|---|---|---|
| Boucle + `list` | ~212 s (extrapolé) | 8,4 Mo |
| Boucle + `set` | 0,109 s | 8,4 Mo |
| Compréhension + `set` | 0,088 s | 8,4 Mo |
| Générateur + `set` | 0,088 s (à la demande) | **224 octets** |

Le gain de **temps** vient du choix de la structure de données (`set`).
Le gain de **mémoire** vient du choix du mode de production (`yield`).
Ce sont deux décisions indépendantes : dans un vrai projet, on prend les deux.

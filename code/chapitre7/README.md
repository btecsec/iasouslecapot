# Chapitre 7 — Les classes : donner un comportement à ses données

## L'énoncé

Modélisez les documents d'un futur moteur de recherche.

1. Écrivez une classe abstraite `Document` avec un `__init__` qui stocke un
   `titre` et un `contenu`, et deux méthodes abstraites : `resumer()` et
   `type_document()`.
2. Créez `Article(Document)` : `resumer()` renvoie les 50 premiers caractères
   suivis de `...`, et `type_document()` renvoie `"article"`.
3. Créez `PageWeb(Document)` avec un `__init__` qui accepte en plus une `url`.
   Appelez le constructeur du parent avec `super().__init__(titre, contenu)`.
4. Rangez trois documents mélangés dans une liste, parcourez-la avec
   `enumerate` et affichez `1. [article] Titre — résumé`.
5. Ajoutez un attribut de classe `compteur` incrémenté à chaque création, et
   une `@classmethod nombre_crees()` qui le renvoie.

*Bonus : essayez d'instancier `Document` directement et lisez l'erreur.*

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_documents.py` | Le corrigé exécutable : la classe abstraite, ses deux implémentations et la boucle polymorphe. |
| `test_exercice_documents.py` | Les tests unitaires, dont deux qui démontrent un point du cours. |
| `requirements.txt` | pytest seulement : le chapitre n'utilise que la bibliothèque standard. |

```bash
pip install -r requirements.txt
python exercice_documents.py
pytest -q
```

## Corrigé commenté

### 1. Le contrat

```python
from abc import ABC, abstractmethod

class Document(ABC):
    compteur = 0                      # attribut de CLASSE

    def __init__(self, titre, contenu):
        self.titre = titre            # attributs d'INSTANCE
        self.contenu = contenu
        Document.compteur += 1

    @abstractmethod
    def resumer(self): ...

    @abstractmethod
    def type_document(self): ...
```

### Le point qui bloque tout le monde : `Document.compteur` et non `self.compteur`

Écrire `self.compteur += 1` **semble** fonctionner, et c'est ce qui le rend
piégeux. Python lit d'abord `self.compteur`, ne le trouve pas sur l'instance,
le récupère donc sur la classe (0), ajoute 1, puis **écrit le résultat sur
l'instance**. Chaque objet se retrouve avec son propre `compteur` valant 1, et
l'attribut de classe reste à 0 pour toujours.

C'est la règle générale : lire un attribut de classe via `self` fonctionne,
mais **écrire** via `self` crée un attribut d'instance qui masque celui de la
classe. Pour modifier le compteur partagé, il faut le nommer explicitement.

### 2 et 3. Les implémentations, et `super()`

```python
class PageWeb(Document):
    def __init__(self, titre, contenu, url):
        super().__init__(titre, contenu)   # on ne recopie pas les lignes du parent
        self.url = url
```

`super().__init__(...)` appelle le constructeur de `Document`. Le recopier
fonctionnerait aujourd'hui, mais le jour où `Document` gagne un attribut,
`PageWeb` ne le recevrait pas — et le compteur ne serait plus incrémenté.

### 4. Le polymorphisme

```python
[
    f"{numero}. [{doc.type_document()}] {doc.titre} — {doc.resumer()}"
    for numero, doc in enumerate(documents, start=1)
]
```

Aucun `if isinstance(...)`, aucun test de type. Le test
`test_une_nouvelle_classe_s_integre_sans_toucher_a_la_boucle` le démontre : on
définit une classe `Video` dans le test lui-même, on la passe à la fonction, et
elle fonctionne sans qu'une ligne du corrigé ait changé.

### 5. `@classmethod`

```python
@classmethod
def nombre_crees(cls):
    return Document.compteur
```

Elle s'appelle sur la classe (`Document.nombre_crees()`), pas sur un objet :
la question « combien en a-t-on créé ? » ne concerne aucun document en
particulier.

### Bonus : l'erreur, en entier

```
TypeError: Can't instantiate abstract class Document without an
implementation for abstract methods 'resumer', 'type_document'
```

Le message nomme les méthodes manquantes. Et surtout, l'erreur survient **à la
construction** — pas au premier appel de `resumer()`, potentiellement des
semaines plus tard en production. C'est tout l'intérêt du contrat.

Le test `test_une_sous_classe_incomplete_est_refusee_aussi` vérifie le même
comportement sur une sous-classe à qui il manque une méthode.

### Réponses chiffrées

| Question | Résultat |
|---|---|
| Longueur d'un résumé tronqué | 53 caractères — 50 + `...` |
| Documents créés dans `main()` | `3` |
| Instancier `Document` | `TypeError` |
| Type d'une `PageWeb` | `"page web"` |

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `self` est l'instance sur laquelle la méthode est appelée — le mot par lequel l'objet se désigne lui-même. Ce n'est pas un mot-clé du langage, seulement une convention. |
| 2 | **b** | `__init__` est appelée automatiquement à la création de l'instance ; on ne l'appelle jamais soi-même. |
| 3 | **b** | L'attribut de classe est partagé par toutes les instances ; l'attribut d'instance, défini avec `self.`, est propre à chaque objet. |
| 4 | **b** | `cls` s'adapte à la classe sur laquelle la méthode est appelée : depuis une classe enfant, elle construit bien un objet de la classe enfant. |
| 5 | **c** | `@abstractmethod` oblige les classes enfants à fournir la méthode, et l'oubli est détecté dès l'instanciation. |

## Ce qu'il faut retenir

Une classe range au même endroit les données et les comportements qui vont
ensemble. L'héritage évite de réécrire, le polymorphisme permet d'écrire des
boucles qui ne changeront plus quand de nouveaux types apparaîtront, et la
classe abstraite transforme une convention tacite en contrat vérifié par
Python. C'est la structure de PyTorch, de scikit-learn et de LangChain — vous
en lirez le code beaucoup plus facilement à partir de maintenant.

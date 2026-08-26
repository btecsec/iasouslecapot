# Chapitre 51 — Piloter un LLM par le prompt

## L'énoncé

1. Prenez une question vague (« parle-moi des tests ») et réécrivez-la en un
   prompt précis (sujet, format, longueur, public). Comparez les réponses.
2. Ajoutez un rôle (« Tu es un examinateur exigeant ») et observez le ton.
3. Construisez un prompt *few-shot* avec deux exemples.
4. Si vous avez une clé d'API, reproduisez l'appel Python (clé dans
   l'environnement !).
5. **Défi** : trouvez un besoin que le prompt ne suffit pas à résoudre.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `prompts.py` | Le corrigé : prompts assemblés par du code, parsing des réponses. |
| `test_prompts.py` | 21 tests — sans clé d'API, sans réseau, sans un centime. |

```bash
pytest -q
python prompts.py
python prompts.py --appel-reel     # seulement si ANTHROPIC_API_KEY est définie
```

## Corrigé

### 1. Du vague au précis

```text
AVANT : parle-moi des tests
```

```text
APRÈS : Rédige une explication structurée en trois parties sur :
        les tests automatisés en Python.
        Longueur : 300 mots environ.
        Public visé : un développeur débutant qui n'a jamais écrit de test.
```

**Ce qui change dans la réponse** : le prompt vague produit un texte
passe-partout qui hésite entre les tests logiciels, les tests A/B et les tests
utilisateurs. Le prompt précis produit un texte utilisable immédiatement.

Les quatre ingrédients, dans l'ordre d'importance :

| Ingrédient | Question à laquelle il répond | Sans lui |
|---|---|---|
| **Sujet** | de quoi parle-t-on exactement ? | le modèle choisit à votre place |
| **Public** | qui lit ? | niveau moyen, donc bon pour personne |
| **Format** | à quoi ressemble le résultat ? | un texte à ré-éditer entièrement |
| **Longueur** | jusqu'où aller ? | trois paragraphes, toujours |

Le test à s'appliquer : **un humain compétent saurait-il répondre à votre
demande sans poser de question ?** Si non, le modèle non plus — il devinera,
et il devinera mal.

### 2. Ajouter un rôle

```text
Tu es un examinateur exigeant : signale ce qui manque.
Rédige une explication structurée...
```

Le ton bascule : moins de reformulations enthousiastes, plus de « attention
à », de lacunes signalées, de contre-exemples. Le rôle ne rend pas le modèle
plus compétent — il **oriente le registre** de sa continuation, et donc les
aspects du sujet qu'il fait remonter.

Le rôle qui marche est concret et vérifiable : « tu es un relecteur qui refuse
tout code sans test » vaut mieux que « tu es un expert mondial ».

### 3. Le few-shot

```text
Classe chaque avis en positif ou negatif.

Avis : Livraison rapide, produit conforme.
Classe : positif

Avis : Colis abime et service client injoignable.
Classe : negatif

Avis : Correct sans plus, mais cher pour ce que c'est.
Classe :
```

Trois détails qui font la différence entre un few-shot qui marche et un qui
dérive :

1. **Le prompt s'arrête pile là où le modèle doit écrire** (`Classe :`). Il n'a
   plus qu'une continuation naturelle possible.
2. **Le format des exemples est rigoureusement identique.** Une virgule qui
   change, et le modèle imite l'irrégularité.
3. **Les exemples couvrent les deux classes.** Deux exemples positifs, et le
   modèle répondra « positif » à tout.

Rappel important : le modèle **n'apprend rien**. Il imite un motif présent dans
le contexte. Fermez la conversation, tout disparaît — c'est là toute la
différence avec le fine-tuning du chapitre 52.

### 4. L'appel réel, et la clé d'API

```python
import os
from anthropic import Anthropic

client = Anthropic()          # lit ANTHROPIC_API_KEY dans l'environnement
reponse = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
print(reponse.content[0].text)
```

**La clé ne s'écrit jamais dans le fichier.** Pas même « juste pour tester » :
Git conserve l'historique, et une clé poussée une seule fois est une clé
compromise — les robots scannent GitHub en continu. Un test du corrigé relit
d'ailleurs le fichier source pour vérifier qu'aucun motif de clé n'y traîne.

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-..."
# macOS / Linux
export ANTHROPIC_API_KEY="sk-..."
```

### Tester une application à base de LLM sans dépenser un euro

C'est le point que l'énoncé ne demande pas, et qui rend le reste utilisable.
Un prompt est **du texte assemblé par du code** : il se teste comme n'importe
quelle chaîne de caractères.

Les 21 tests du dossier vérifient trois choses, et aucune n'appelle un modèle :

| Ce qui est testé | Pourquoi c'est là que ça casse |
|---|---|
| L'assemblage du prompt | un `{trou}` non rempli, un exemple oublié, un ordre inversé |
| **Le parsing de la réponse** | `« Positif »`, `« positif. »`, `« Classe : positif »` — trois formes pour une même réponse |
| L'appel lui-même (avec un *mock*) | on vérifie qu'on envoie le bon modèle et le bon message |

Le parsing mérite l'insistance. Le corrigé renvoie `None` quand la réponse est
ambiguë (« c'est entre positif et negatif ») **au lieu de deviner**. Un parsing
qui devine produit des erreurs silencieuses — et une erreur silencieuse dans
une chaîne de traitement de 500 avis, personne ne la voit passer.

### 5. Défi : ce que le prompt ne sait pas faire

| Besoin | Pourquoi le prompt échoue | La bonne réponse |
|---|---|---|
| Répondre sur vos documents internes | le modèle ne les a jamais vus | **RAG** (ch. 49) |
| Connaître l'actualité du jour | savoir figé à la date de coupure | **RAG** ou outil de recherche |
| Un ton parfaitement constant sur 100 000 appels | un prompt système long coûte des jetons à chaque appel, et le modèle dérive quand même | **fine-tuning** (ch. 48) |
| Un format strict garanti à 100 % | le prompt améliore, ne garantit pas | fine-tuning, ou une validation Pydantic qui **rejette** (ch. 41) |
| Un vocabulaire métier très spécialisé | absent du corpus d'entraînement | fine-tuning, ou un glossaire dans le contexte |

L'ordre économique reste le même : **prompt → RAG → fine-tuning**. Le prompt
est gratuit et instantané ; le RAG demande une infrastructure ; le fine-tuning
demande des données annotées et du calcul.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Un prompt est le texte que vous envoyez au LLM pour obtenir une réponse. |
| 2 | **b** | Le few-shot montre quelques exemples dans le prompt pour guider le modèle. |
| 3 | **b** | La clé d'API va dans une variable d'environnement, jamais en clair dans le code. |
| 4 | **b** | On commence toujours par un bon prompt : gratuit, immédiat, et souvent suffisant. |

## Ce qu'il faut retenir

Un prompt précis vaut un long réglage. Écrivez-le comme du code — assemblé par
des fonctions, versionné, testé — et testez surtout le **parsing** de la
réponse : c'est là que vivent les bugs, pas dans le modèle.

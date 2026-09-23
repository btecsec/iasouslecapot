# Chapitre 29 — LangChain et LangGraph

## L'énoncé

1. Dans la chaîne `prompt | modele | parser`, que se passerait-il si vous
   retiriez le parser ? Que devrait alors afficher `print` ?
2. Réécrivez le prompt pour obtenir un résumé en trois points d'un texte fourni
   en variable. Quels trous (`{...}`) définissez-vous ?
3. Retirez par la pensée le garde-fou `tentative >= 10`. Décrivez le pire
   scénario, en euros comme en temps.
4. LangChain ou LangGraph ? (a) traduire un fichier ; (b) un assistant qui
   cherche sur le web, lit, puis cherche encore ; (c) classer 500 avis.

## Corrigé

### 1. Sans le parser

La chaîne ne renverrait plus une chaîne de caractères mais un **objet message**
(`AIMessage`), qui contient le texte *et* ses métadonnées (nombre de jetons,
modèle utilisé, raison d'arrêt).

```python
resultat = (prompt | modele).invoke({"sujet": "les tests"})
print(resultat.content)      # ← il faut aller chercher .content
```

Le `StrOutputParser` ne fait rien de magique : il extrait `.content`. Son
intérêt est de rendre la chaîne **composable** — la sortie de l'étape est une
chaîne de caractères, donc directement réinjectable dans l'étape suivante.

### 2. Le prompt de résumé

```python
prompt = ChatPromptTemplate.from_template(
    "Résume le texte suivant en exactement trois points.\n"
    "Chaque point tient en une phrase, à destination de {public}.\n\n"
    "Texte : {texte}"
)
chaine = prompt | modele | parser
chaine.invoke({"texte": mon_texte, "public": "un débutant"})
```

Deux trous : `{texte}` (la donnée) et `{public}` (le cadrage). La règle de
conception : **un trou par chose qui varie**. Si le format ne varie jamais,
il reste écrit en dur dans le gabarit — ce n'est pas une variable, c'est une
consigne.

### 3. Sans le garde-fou : le pire scénario

Le cycle *chercher → évaluer → « pas assez bon » → chercher* n'a plus de sortie.

| | Estimation |
|---|---|
| Durée d'un tour | ~3 s (appel modèle + recherche) |
| Tours en une nuit (8 h) | ~9 600 |
| Jetons par tour | ~2 000 en entrée + 500 en sortie, et le contexte grossit |
| Facture | de quelques dizaines à plusieurs centaines d'euros — et elle croît car chaque tour rallonge le contexte |

Le vrai danger n'est pas la boucle infinie *théorique* : c'est qu'elle tourne
sans témoin, la nuit, sur une carte bancaire d'entreprise. D'où la règle :
**tout cycle dans un graphe porte un compteur, et le compteur a un maximum**.
C'est le seul garde-fou qui ne dépend pas du bon vouloir du modèle.

### 4. LangChain ou LangGraph ?

| Besoin | Choix | Pourquoi |
|---|---|---|
| (a) Traduire un fichier en anglais | **LangChain** | Un aller simple : entrée → modèle → sortie. Aucune décision, aucun retour en arrière. |
| (b) Assistant qui cherche, lit, puis cherche encore s'il n'a pas trouvé | **LangGraph** | Il y a une **boucle** conditionnelle : le résultat d'une étape décide de la suite. |
| (c) Classer 500 avis en positif/négatif | **LangChain** | 500 allers simples indépendants (`.batch()`), toujours la même chaîne. |

Le critère tient en une question : **le programme peut-il revenir en
arrière ?** Non → une chaîne suffit. Oui → il vous faut un graphe.

## Le code de ce dossier

Impossible de tester une vraie chaîne LangChain sans clé d'API — et un test
qui appelle un modèle payant n'est pas un test unitaire (chapitre 9). On
reconstruit donc les **deux mécanismes** en Python pur, ce qui les rend
vérifiables et, surtout, compréhensibles :

| Fichier | À quoi ça sert |
|---|---|
| `mini_chaine.py` | L'opérateur `\|` réimplémenté en 30 lignes, plus un graphe avec garde-fou. |
| `test_mini_chaine.py` | Les tests : composition, parser, boucle bornée, coût maîtrisé. |

```bash
pytest -q     # aucune dépendance, aucune clé d'API
```

`mini_chaine.py` ne remplace pas LangChain : il montre qu'un `Runnable` n'est
rien d'autre qu'un objet avec une méthode `invoke` et un `__or__`.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | LangChain fournit des briques prêtes (prompts, modèles, parsers, mémoire, outils) pour assembler une application à base de LLM. Il n'entraîne rien. |
| 2 | **b** | Le `\|` fait passer la sortie de chaque étape à la suivante, comme le tube du shell. |
| 3 | **b** | Dès que le programme doit boucler, décider ou revenir en arrière. |
| 4 | **c** | L'état transporte les informations d'un nœud à l'autre : c'est la mémoire de travail du graphe. |
| 5 | **b** | Pour éviter la boucle infinie — et la facture d'API qui va avec. |

## Ce qu'il faut retenir

Une chaîne va tout droit, un graphe peut revenir en arrière. Le jour où vous
écrivez un graphe, vous écrivez aussi son compteur de tours : c'est le prix
d'entrée de la boucle.

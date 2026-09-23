# Chapitre 21 — Les grands modèles de langage (LLM)

Exercice de réflexion : **pas de code** (le calcul de l'attention, lui, est
au chapitre 22).

## Corrigé de l'exercice

### 1. La mission fondamentale d'un LLM, en une phrase

> Prédire le jeton suivant le plus probable, étant donné tout ce qui précède.

C'est tout. Traduire, résumer, coder, argumenter : ce sont des **conséquences**
de cette unique capacité, pas des fonctions séparées. Un modèle à qui l'on
donne « Traduis en anglais : bonjour →  » continue le texte, et cette
continuation se trouve être une traduction.

Corollaire immédiat, et c'est le plus important : **rien dans cet objectif ne
mentionne la vérité.** Un modèle optimisé pour la plausibilité produit du
plausible. Le vrai n'est qu'un cas particulier fréquent du plausible.

### 2. Pourquoi transformer le texte en tokens puis en embeddings ?

Deux étapes, deux raisons distinctes :

| Étape | Ce qu'on fait | Pourquoi |
|---|---|---|
| **Tokenisation** | découper le texte en morceaux, puis en numéros (`"chat" → 4372`) | un réseau ne manipule que des nombres |
| **Embedding** | remplacer chaque numéro par un vecteur appris (`4372 → [0.12, -0.4, ...]`) | le numéro 4372 n'a aucun sens mathématique ; le vecteur, si |

Le point clé est le second. Avec des numéros bruts, le modèle croirait que le
jeton 4372 est « deux fois » le jeton 2186. L'embedding place au contraire les
mots dans un espace où **la proximité géométrique traduit la proximité de
sens** : « chat » et « chien » finissent voisins, parce qu'ils apparaissent
dans les mêmes contextes.

### 3. En quoi l'attention améliore-t-elle la compréhension du contexte ?

Prenez : « La **banque** était fermée, alors j'ai attendu sur la **berge**. »
Sans attention, le modèle traite chaque mot isolément et ne peut pas trancher
entre l'établissement financier et le bord du fleuve.

L'attention permet à chaque mot de **regarder tous les autres et de pondérer
leur importance** : « banque » consacre une part importante de son attention à
« berge », ce qui oriente sa représentation. Trois avantages concrets :

- la portée est **longue** : le mot 1 peut regarder le mot 500 directement, sans
  passer par les 499 intermédiaires (le défaut des anciens réseaux récurrents) ;
- les poids sont **dynamiques** : ils dépendent de la phrase, pas de règles ;
- le calcul est **parallélisable**, donc entraînable sur GPU — c'est ce qui a
  rendu les LLM économiquement possibles.

Le mécanisme exact, chiffres à l'appui, est au chapitre 22.

### 4. Une date historique fausse mais assurée

C'est une **hallucination**.

Le réflexe correct tient en trois gestes :

1. **Vérifier systématiquement les faits vérifiables** — dates, chiffres,
   noms, références, citations. L'assurance du ton n'est pas un indice de
   fiabilité : le modèle n'a aucun moyen de savoir qu'il ne sait pas.
2. **Demander les sources** — et les ouvrir. Un LLM peut inventer une
   référence entière, avec auteur et numéro de page plausibles.
3. **Changer d'outil pour les questions factuelles** : c'est exactement le rôle
   du RAG (chapitre 53), qui fait répondre le modèle *à partir de documents
   fournis*, pas de sa mémoire.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Prédire le mot (jeton) suivant : c'est l'unique tâche du pré-entraînement. |
| 2 | **b** | Un token est un morceau de texte — souvent un mot, parfois un fragment. |
| 3 | **b** | Le mécanisme d'attention, qui pèse les relations entre tous les mots d'un coup. |
| 4 | **b** | Une hallucination est une réponse plausible mais fausse, affirmée avec assurance. |

## Ce qu'il faut retenir

Un LLM est un moteur de plausibilité, pas une base de connaissances. Cette
phrase explique à la fois ses prouesses et ses erreurs — et elle justifie tout
ce que fait la partie VI du livre.

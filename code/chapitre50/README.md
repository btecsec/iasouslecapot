# Chapitre 50 — Comprendre les grands modèles de langage (LLM)

Exercice sans code : cinq questions pour ancrer les idées avant de passer à la
pratique (prompt, fine-tuning, RAG).

## Corrigé de l'exercice

### 1. Ce que fait un LLM, en une phrase

> « Il prédit le prochain morceau de texte le plus probable, compte tenu de
> tout ce qui précède. »

Tout le reste en découle. Un LLM ne « répond » pas : il **continue**. Quand
vous posez une question, vous fournissez un contexte dont la continuation la
plus probable ressemble à une réponse.

### 2. Trois tâches sans réentraînement

Un même modèle, sans qu'on touche à un seul poids :

1. **Résumer** un rapport de dix pages en cinq points.
2. **Traduire** un texte du français vers l'anglais.
3. **Reformater** — extraire d'un e-mail un JSON `{nom, date, montant}`.

On pourrait ajouter : classer, corriger, expliquer du code, générer des tests.
Cette polyvalence porte un nom : l'apprentissage **en contexte** (*in-context
learning*). Le modèle n'apprend rien de durable ; il adapte sa continuation aux
exemples et consignes présents dans le prompt. Fermez la conversation, tout
disparaît.

### 3. Repérer une hallucination possible

Les zones à risque sont toujours les mêmes :

| Zone à risque | Comment vérifier |
|---|---|
| Chiffres précis, dates, statistiques | source primaire, et ne pas se contenter du nom d'un rapport |
| Citations et références bibliographiques | **ouvrir** le lien ; un DOI inventé a l'air d'un vrai DOI |
| Noms de fonctions ou d'options d'une bibliothèque | la documentation officielle, ou l'autocomplétion de l'éditeur |
| Faits sur des personnes peu connues | recoupement sur deux sources indépendantes |

Le signal d'alerte le plus fiable : **l'absence d'hésitation sur un détail très
spécifique**. Un modèle sûr de lui n'est pas un modèle qui sait — il n'a aucun
moyen de mesurer sa propre ignorance.

### 4. Pourquoi un LLM ne connaît pas l'actualité du jour

Parce que son savoir a été figé au moment de l'entraînement. Trois causes
cumulées :

- **La date de coupure** : le corpus s'arrête à une date, souvent plusieurs
  mois avant la sortie du modèle.
- **Le coût** : un pré-entraînement complet coûte des millions d'euros et des
  semaines de calcul. On ne le refait pas chaque matin.
- **Le mode de stockage** : les connaissances sont réparties dans des
  milliards de poids, pas rangées dans une base. On ne peut pas « ajouter une
  ligne ».

D'où : un LLM sans outil ne peut pas connaître l'actualité, et **le
fine-tuning ne résout pas ce problème** (chapitre 52). La bonne réponse est le
RAG ou un accès à une recherche web (chapitre 53).

### 5. Défi : chaque limite et sa technique

| Limite | Technique | Chapitre |
|---|---|---|
| Réponses hors sujet ou mal formatées | un meilleur **prompt** (rôle, format, exemples) | 47 |
| Ton ou format inconstant sur des milliers d'appels | **fine-tuning** : le comportement est ancré dans les poids | 48 |
| Ne connaît pas vos documents internes | **RAG** : on lui fournit les extraits pertinents | 49 |
| Ne connaît pas l'actualité | **RAG** sur une source à jour, ou outil de recherche | 49-50 |
| Invente des sources | **RAG** + consigne « réponds uniquement à partir du contexte » | 50 |
| Réponse fausse malgré tout | **un humain dans la boucle** | 51 |

L'ordre a un sens économique : le prompt est gratuit et immédiat, le RAG coûte
une infrastructure, le fine-tuning coûte des données annotées et du calcul.
**On essaie toujours dans cet ordre.**

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | LLM = *Large Language Model*, grand modèle de langage. |
| 2 | **b** | Sa tâche d'entraînement est de prédire le mot suivant. |
| 3 | **b** | Une hallucination est un fait faux énoncé avec assurance. |
| 4 | **b** | L'attention permet de se concentrer sur les mots pertinents du contexte, aussi éloignés soient-ils. |

## Ce qu'il faut retenir

Un LLM est un moteur de continuation plausible, figé à sa date
d'entraînement. Ses trois limites — pas de connaissance récente, pas de
documents privés, aucune conscience de ce qu'il ignore — définissent
exactement le programme des chapitres 51 à 55.

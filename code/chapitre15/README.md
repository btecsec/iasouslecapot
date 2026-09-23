# Chapitre 15 — Supervisé, non supervisé, par renforcement

Exercice de classement : **pas de code**. Quel mode d'apprentissage pour
chaque situation ?

## Corrigé de l'exercice

| Situation | Mode | Justification |
|---|---|---|
| 50 000 avis clients déjà notés positif/négatif | **Supervisé** | Vous possédez la bonne réponse pour chaque exemple. C'est la définition même du supervisé : apprendre la relation avis → étiquette. |
| Un million de photos sans étiquette, chercher des groupes | **Non supervisé** | Aucune réponse attendue n'existe. L'algorithme regroupe par ressemblance (*clustering*) et c'est vous qui interpréterez ensuite les groupes. |
| Apprendre seul à garer une voiture dans un simulateur | **Renforcement** | Il n'y a pas de « bonne manœuvre » étiquetée, mais un environnement qui récompense (garé) et pénalise (raté, collision). L'agent apprend par essais et erreurs. |
| Patients avec diagnostic confirmé, prédire pour de nouveaux patients | **Supervisé** | Le diagnostic confirmé est l'étiquette. Classification (chapitre 16). |

### La question qui tranche

**Ai-je, pour chaque exemple, la bonne réponse écrite quelque part ?**

- Oui → supervisé.
- Non, et je veux découvrir une structure → non supervisé.
- Non, mais j'ai un environnement qui me dit après coup si c'était bien joué →
  renforcement.

### Le piège du cas 3

Beaucoup répondent « supervisé » pour la voiture, en pensant : « il suffit
d'enregistrer un humain qui se gare ». C'est une autre approche valable
(*imitation learning*, du supervisé), mais l'énoncé dit **« apprenne seul »** :
personne ne fournit d'exemples. Le signal est une récompense obtenue *après*
une séquence d'actions — et c'est ce décalage temporel qui fait toute la
difficulté du renforcement : quelle action, dix secondes plus tôt, mérite le
crédit du succès ?

### Le quatrième mode, celui des LLM

L'**auto-supervisé** ne figure pas dans les trois cases classiques, et c'est
pourtant celui qui a rendu les LLM possibles. Le principe : les étiquettes
sont extraites automatiquement de la donnée elle-même.

> Phrase du corpus : « Le chat dort sur le »
> Étiquette : « canapé » — le mot suivant, qu'il suffit de masquer.

Aucun humain n'annote quoi que ce soit : n'importe quel texte devient un jeu
de données étiqueté. C'est ce qui permet d'apprendre sur des milliers de
milliards de mots : entraîner un modèle de langage revient à décaler `y`
d'un jeton par rapport à `x`, puis à lui demander de prédire la suite.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le supervisé apprend d'exemples accompagnés de la bonne réponse. |
| 2 | **b** | Le non supervisé regroupe par ressemblance, sans étiquettes. |
| 3 | **b** | Le renforcement repose sur essais, erreurs et récompenses. |
| 4 | **b** | Leur pré-entraînement est auto-supervisé : l'étiquette (le mot suivant) vient du texte lui-même. |

## Ce qu'il faut retenir

Le mode d'apprentissage n'est pas un choix esthétique : il est **dicté par les
données dont vous disposez**. Avant de choisir un modèle, demandez-vous
toujours où sont vos étiquettes — et si elles n'existent pas, combien coûterait
leur fabrication.

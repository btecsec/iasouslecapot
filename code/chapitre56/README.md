# Chapitre 56 — Limites, éthique et suite du parcours

Exercice de bilan : **pas de code**. Le corrigé propose une trame ; vos
réponses, elles, doivent parler de *votre* domaine.

## Corrigé de l'exercice

### 1. Deux biais possibles, et comment les repérer

Prenons un modèle de tri de candidatures — le cas d'école, parce qu'il a
vraiment échoué en production chez plusieurs grandes entreprises.

| Biais | D'où il vient | Comment le repérer |
|---|---|---|
| **Biais historique** | Le modèle apprend sur dix ans de recrutements passés. Si l'entreprise a majoritairement recruté des hommes, il apprend que « homme » corrèle avec « recruté ». | Calculer le taux de sélection **par groupe**. Un écart marqué est un signal, pas une preuve — mais il impose l'enquête. |
| **Biais de représentation** | Les CV de personnes formées à l'étranger sont rares dans les données : le modèle les traite mal, faute d'exemples. | Regarder la performance **par sous-groupe**, pas la moyenne globale. Un modèle à 92 % peut être à 60 % sur 5 % de la population. |

Le piège classique : croire qu'il suffit de **retirer la variable sensible**.
Le code postal, le nom du lycée, la pratique d'un sport reconstituent l'origine
sociale ou le genre. La variable retirée revient par la fenêtre — on appelle
cela des *proxies*. La seule méthode fiable est de **mesurer les écarts de
résultat par groupe**, pas de fermer les yeux sur les entrées.

### 2. Trois réflexes responsables

1. **Documenter le domaine de validité.** Sur quelles données le modèle a-t-il
   appris, et pour quels cas est-il *hors* de son domaine ? Une phrase suffit :
   « entraîné sur des manchots adultes de l'archipel Palmer, mesurés entre 2007
   et 2009 ». Elle évite qu'on l'applique un jour à des juvéniles.
2. **Garder un humain sur les décisions importantes**, avec un vrai pouvoir de
   contredire. Un humain qui valide 400 décisions par jour ne valide rien —
   c'est un tampon, pas un contrôle.
3. **Prévoir le recours.** Une personne affectée par une décision automatisée
   doit pouvoir demander une explication et un réexamen. C'est une obligation
   légale en Europe (RGPD, article 22) autant qu'une exigence morale.

À quoi s'ajoutent deux réflexes techniques déjà vus : mesurer **par classe**
(chapitre 40), et surveiller la dérive (chapitre 48).

### 3-4. Le chemin parcouru et la feuille de route

Relisez ce que vous aviez écrit à l'exercice du **chapitre 1** — les trois
questions sur votre projet rêvé et vos freins. C'était l'objet de la consigne
« gardez vos réponses ».

Une feuille de route à trois mois qui fonctionne :

| Mois | Objectif | Livrable concret |
|---|---|---|
| 1 | **Un projet de bout en bout**, sur *vos* données | un dépôt Git avec données, notebook, tests, README |
| 2 | **La mise en production** | l'API en conteneur, déployée quelque part, avec sa surveillance |
| 3 | **Un approfondissement**, un seul | soit les LLM (RAG sur vos documents), soit les séries temporelles, soit la vision — pas les trois |

La règle qui compte : **un projet fini vaut mieux que trois commencés.** Un
dépôt public avec un README clair et des tests qui passent en dit plus long sur
vous que dix certificats.

### 5. Défi final : un projet de A à Z, chapitre par chapitre

| Étape | Ce que vous faites | Chapitres |
|---|---|---|
| Cadrer | tâche, données, métrique, risque éthique | 19, 23, 26 |
| Collecter et explorer | trouver les données, les regarder | 27 |
| Nettoyer | trous, doublons, encodage, échelles | 28 |
| Découper | train / validation / test, stratifié | 29 |
| Baseline | le modèle le plus simple, plus un `DummyClassifier` | 30 |
| Entraîner | pipeline, `fit`/`predict`/`score` | 31-33 |
| Diagnostiquer | courbes, surapprentissage | 34-35 |
| Évaluer | matrice de confusion, métriques adaptées au coût des erreurs | 36 |
| Ajuster | validation croisée, recherche d'hyperparamètres | 37 |
| Sauvegarder | modèle **et** scaler **et** colonnes | 38 |
| Exposer | API FastAPI, validation Pydantic | 41 |
| Emballer | Dockerfile, `.dockerignore` | 42 |
| Déployer | registre, service, HTTPS, secrets, budget | 43 |
| Surveiller | latence p95, dérive des données et du concept | 44 |
| Automatiser | CI/CD avec test de non-régression | 45 |
| Spécialiser (si LLM) | prompt → RAG → fine-tuning, dans cet ordre | 47-50 |

Si vous savez remplir cette colonne de droite sans regarder le sommaire, le
livre a atteint son but.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Un modèle apprend de données humaines, avec leurs biais — il les reproduit et peut les amplifier. |
| 2 | **b** | Les hallucinations : énoncer des faussetés avec assurance. |
| 3 | **a** | « Un humain dans la boucle » signifie qu'un humain garde le dernier mot sur les décisions importantes. |
| 4 | **b** | Avoir une méthode solide et le réflexe d'apprendre ce qui manque — les outils changeront, la méthode non. |

## Ce qu'il faut retenir

Vous savez développer, tester, livrer, surveiller et spécialiser un modèle.
Ce qui reste à cultiver n'est pas un outil de plus : c'est l'habitude de
demander, avant chaque projet, **qui subit une erreur du modèle** — et de
prévoir la réponse avant d'écrire la première ligne.

# Sous le capot de l'IA — le code du livre

Ce dépôt contient le code qui accompagne le livre **Sous le capot de l'IA**,
disponible sur Amazon : <https://www.amazon.fr/dp/B0HHFFW39S>

## Le projet

Le livre part de zéro et va jusqu'au métier d'AI Engineer : développer un
modèle, le tester, le mettre en production puis le surveiller. Il avance pas à
pas, avec des mots simples d'abord et les termes techniques ensuite, des
analogies pour les notions abstraites, et un exercice suivi d'un quiz à la fin
de chaque chapitre.

Ce dépôt en est le complément pratique. On y trouve, pour chacun des
**56 chapitres** :

- l'**énoncé** de l'exercice et son **corrigé complet**, commenté ;
- les réponses expliquées au **quiz** de validation ;
- le **code** du chapitre, prêt à lancer ;
- des **tests** `pytest` qui vérifient que le corrigé dit vrai.

## Le parcours en six parties

| Partie | Thème | Chapitres |
|---|---|---|
| I | Prérequis et environnement : Python, venv, pip, tests, NumPy, Pandas | 1-11 |
| II | Introduction à l'IA : vocabulaire, maths, réseaux, LLM, Transformer | 12-22 |
| III | Les frameworks : PyTorch, TensorFlow/Keras, LangChain | 23-29 |
| IV | Développer un modèle : explorer, nettoyer, entraîner, évaluer | 30-43 |
| V | Production (MLOps) : API, Docker, cloud, surveillance, CI/CD | 44-49 |
| VI | Fine-tuning (LoRA) et RAG | 50-56 |

## Démarrer

```bash
git clone https://github.com/btecsec/iasouslecapot.git
cd iasouslecapot/code/chapitre35
pip install -r requirements.txt
python exercice_scikit.py
pytest -q
```

Chaque chapitre a son propre `requirements.txt` : installez les dépendances
au fur et à mesure de votre lecture.

Le détail chapitre par chapitre, avec les dépendances de chaque famille de
chapitres, est dans [`code/README.md`](code/README.md).

## Licence

Le code de ce dépôt est distribué sous licence [MIT](LICENSE). Le texte du
livre, lui, n'est pas concerné : il reste protégé par le droit d'auteur.

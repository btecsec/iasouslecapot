# Chapitre 9 — Tester son code, et tester son IA

Un petit projet **entièrement testé**, qui sert à la fois de corrigé de
l'exercice et de squelette à recopier pour vos propres projets.

## Contenu

| Fichier | À quoi ça sert |
|---|---|
| `calculs.py` | Le code à tester : `moyenne` et `normaliser`. |
| `agent.py` | Le code déterministe qui entoure un LLM : interface, parsing, agent à outils. Rien n'appelle un vrai modèle. |
| `tests/conftest.py` | Les fixtures partagées : graines aléatoires figées, jeux de données. |
| `tests/test_calculs.py` | Cas normal, cas d'erreur, cas limite, `parametrize`. |
| `tests/test_donnees.py` | Tester les **données** : schéma, plages, doublons, types. |
| `tests/test_llm.py` | Tester autour d'un LLM avec des *mocks* : parsing robuste, agent, outils. |
| `tests/test_integration.py` | Les tests lents, marqués `integration` et exclus par défaut. |
| `pyproject.toml` | Configuration pytest (marqueurs, chemins) et Ruff. |
| `exemple_ci.yml` | Un pipeline GitHub Actions prêt à copier. |
| `solution_exercice.md` | Le corrigé commenté, avec les pièges rencontrés. |

## Démarrage

```bash
# Depuis ce dossier, dans un environnement virtuel activé (chapitre 4)
pip install -r requirements.txt

pytest -q -m "not integration"   # les tests rapides : gratuits, en < 1 s
pytest -q -m integration         # les tests lents (ignorés sans clé d'API)
pytest -q                        # tout
```

Sortie attendue :

```text
23 passed, 1 skipped, 2 deselected, 1 xfailed
```

- **1 skipped** : les tests de données sont ignorés si pandas n'est pas installé.
- **2 deselected** : les tests marqués `integration`, écartés par le filtre.
- **1 xfailed** : un test volontairement faux, qui sert à l'étape 3 de
  l'exercice. Retirez son décorateur `@pytest.mark.xfail` pour voir à quoi
  ressemble un vrai rapport d'échec.

## Couverture

```bash
pytest -m "not integration" --cov=. --cov-report=term-missing
```

La colonne `Missing` est une liste de suspects, pas une note. Ne visez pas
100 % : les seules lignes non couvertes ici sont celles qui appelleraient la
vraie API, et c'est exactement ce qu'on ne veut jamais exécuter en test.

## Le principe à emporter

Un modèle est non déterministe, lent et payant. Le code qui l'entoure — parsing
des réponses, choix des outils, validation des données, mise en forme des
prompts — est du **Python ordinaire**. C'est là que vivent la plupart des bugs,
et il se teste en quelques millisecondes pour zéro euro.

# Chapitre 5 — La boîte à outils Python indispensable

## L'énoncé

1. Dans un venv activé : `pip install requests rich`.
2. Vérifiez avec `pip list`.
3. Générez la recette : `pip freeze > requirements.txt`.
4. Ouvrez le fichier et observez les versions figées.
5. Bonus : désinstallez une bibliothèque, puis réinstallez tout avec
   `pip install -r requirements.txt`.

## Corrigé

```bash
# 1 — installation (le venv du chapitre 4 doit être actif : cherchez le (venv))
pip install requests rich

# 2 — vérification
pip list
```

```text
Package    Version
---------- -------
certifi    2024.7.4
charset-normalizer 3.3.2
idna       3.7
markdown-it-py 3.0.0
mdurl      0.1.2
pygments   2.18.0
requests   2.32.3
rich       13.7.1
urllib3    2.2.2
```

**La première surprise** : vous avez demandé 2 bibliothèques, il y en a 9.
Les 7 autres sont des *dépendances transitives* — ce dont `requests` et `rich`
ont eux-mêmes besoin. C'est normal, et c'est exactement pour cela qu'on fige la
liste complète.

```bash
# 3 — la recette
pip freeze > requirements.txt
```

```text
certifi==2024.7.4
charset-normalizer==3.3.2
idna==3.7
requests==2.32.3
rich==13.7.1
...
```

**La deuxième surprise** : `pip freeze` écrit `==` (version exacte) alors qu'on
écrit souvent `>=` à la main. Les deux ont un usage :

| Écriture | Sens | Quand l'utiliser |
|---|---|---|
| `requests==2.32.3` | exactement cette version | déploiement, Docker, CI — on veut la reproductibilité |
| `requests>=2.32` | cette version ou plus récente | bibliothèque partagée — on veut de la souplesse |

```bash
# 5 — bonus
pip uninstall rich -y
pip install -r requirements.txt   # tout revient, aux mêmes versions
```

### Le piège classique

`pip freeze` capture **tout** l'environnement, y compris ce que vous aviez
installé pour tester et oublié. D'où la règle : un venv par projet
(chapitre 4), sinon votre `requirements.txt` embarque la moitié d'Internet.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `analyser_requirements.py` | Lit un `requirements.txt` et dit ce qui est figé, ce qui est souple, ce qui est mal écrit. |
| `requirements.txt` | Le fichier d'exemple de l'exercice. |
| `test_analyser_requirements.py` | Les tests unitaires du parseur. |

```bash
python analyser_requirements.py requirements.txt
pytest -q
```

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `pip install pandas` télécharge le paquet depuis PyPI et l'installe dans l'environnement actif. |
| 2 | **b** | La syntaxe est `pip install pandas==2.1.0` (double égal, comme une comparaison). |
| 3 | **b** | `pip freeze` liste les paquets installés avec leur version ; le `>` redirige cette liste dans un fichier. |
| 4 | **c** | `pip install -r requirements.txt` — le `-r` signifie *requirements*. |

## Ce qu'il faut retenir

`requirements.txt` est la recette de votre projet. Sans elle, votre code n'est
reproductible ni par un collègue, ni par un serveur, ni par vous-même dans six
mois. Elle se versionne toujours ; le dossier `venv/`, jamais.

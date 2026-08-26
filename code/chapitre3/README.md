# Chapitre 3 — Installer Python et configurer son environnement de travail

Ce chapitre n'a pas d'exercice pratique séparé : l'installation *est*
l'exercice. Ce dossier contient de quoi vérifier que votre poste est prêt.

## Contenu

| Fichier | À quoi ça sert |
|---|---|
| `verifier_installation.py` | Contrôle Python, le PATH, pip et l'encodage. À lancer avant tout le reste du livre. |
| `test_verifier_installation.py` | Les tests unitaires du script (ils tournent partout, même sur une machine mal configurée). |

## Démarrage

```bash
python verifier_installation.py
pytest -q
```

Sortie attendue sur un poste correctement configuré :

```text
[ok]   Version de Python : 3.12.x (>= 3.10 requis)
[ok]   Python est bien dans le PATH
[ok]   pip disponible : 24.x
[ok]   Encodage de sortie : utf-8
Tout est prêt. Passez au chapitre 4.
```

## Corrigé : les trois pannes classiques

1. **`python n'est pas reconnu` (Windows).** La case *Add Python to PATH* n'a
   pas été cochée pendant l'installation. Le plus rapide : relancer
   l'installeur, choisir *Modify*, cocher la case. Sinon, utilisez `py` à la
   place de `python`.
2. **`pip n'est pas reconnu`.** Appelez-le comme un module :
   `python -m pip install ...`. Cette forme marche toujours, même quand le
   raccourci `pip` manque — c'est celle à prendre comme réflexe.
3. **Des `?` ou des `Ã©` à l'affichage.** Votre terminal n'est pas en UTF-8.
   Sous Windows : `chcp 65001`, ou lancez vos scripts depuis le terminal
   intégré de VS Code.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Sa lisibilité et son écosystème (NumPy, pandas, PyTorch…). Python n'est pas rapide en soi : ses bibliothèques scientifiques sont écrites en C et en Fortran. |
| 2 | **b** | *Add Python to PATH* : sans elle, le terminal ne trouve pas la commande `python`. |
| 3 | **b** | `print(...)` affiche à l'écran. Rien à voir avec une imprimante. |
| 4 | **c** | Un commentaire commence par `#`. Tout ce qui suit sur la ligne est ignoré par Python. |

## Ce qu'il faut retenir

Un poste de travail sain se vérifie en une commande. Prenez l'habitude de
lancer `verifier_installation.py` quand « ça ne marche pas » : neuf fois sur
dix, le problème est le PATH ou l'environnement virtuel, pas votre code.

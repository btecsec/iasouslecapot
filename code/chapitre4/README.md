# Chapitre 4 — Les environnements virtuels

## L'énoncé

1. Créez un dossier `test_env` et placez-vous dedans.
2. Créez un environnement virtuel : `python -m venv venv`.
3. Activez-le (repérez le `(venv)`).
4. Désactivez-le avec `deactivate`.

## Corrigé, commande par commande

```bash
mkdir test_env
cd test_env

python -m venv venv          # crée le dossier venv/ (~15 Mo)
```

Activation — **la commande change selon le terminal**, c'est le piège n° 1 :

| Terminal | Commande d'activation |
|---|---|
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |
| Windows cmd.exe | `venv\Scripts\activate.bat` |
| Git Bash / macOS / Linux | `source venv/bin/activate` |

```bash
deactivate                   # partout : la même commande
```

### Ce que vous devez observer

```text
C:\...\test_env>                  ← avant
(venv) C:\...\test_env>           ← après activation
C:\...\test_env>                  ← après deactivate
```

Le `(venv)` n'est pas décoratif : c'est le seul indicateur visuel qui vous dit
où `pip install` va déposer ses fichiers.

### Si PowerShell refuse d'activer

```text
Impossible de charger le fichier ... l'exécution de scripts est désactivée
```

C'est une protection de Windows, pas une erreur de votre part :

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Les deux règles à retenir

- **Un projet = un environnement.** On ne partage pas un venv entre projets.
- **`venv/` ne se versionne jamais.** On versionne `requirements.txt`
  (chapitre 5), qui permet de le reconstruire à l'identique. Ajoutez `venv/`
  à votre `.gitignore`.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `verifier_venv.py` | Dit si vous êtes dans un environnement virtuel, et lequel. |
| `test_verifier_venv.py` | Les tests unitaires du script. |

```bash
python verifier_venv.py
pytest -q
```

Le principe de détection tient en une ligne : Python expose `sys.prefix`
(l'environnement courant) et `sys.base_prefix` (l'installation d'origine).
Quand un venv est actif, **les deux diffèrent**. C'est le test le plus fiable,
bien plus que de chercher `(venv)` dans l'invite.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Les conflits de versions : le projet A veut pandas 1.5, le projet B veut pandas 2.2. Sans isolement, l'un des deux casse. |
| 2 | **b** | `python -m venv venv` : le premier `venv` est le module, le second le nom du dossier. |
| 3 | **b** | Le nom de l'environnement apparaît entre parenthèses au début de l'invite. |
| 4 | **c** | `venv` : livré avec Python, léger, suffisant pour tout ce livre. `conda` devient intéressant pour les dépendances non-Python lourdes. |

## Ce qu'il faut retenir

Créer → activer → installer → désactiver. Ce cycle de quatre gestes protège
votre machine de la « bouillie de bibliothèques » et rend vos projets
reproductibles.

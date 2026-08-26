# Chapitre 8 — Astuces Python pour coder plus vite et mieux

Le code du chapitre et le corrigé de son exercice.

## Contenu

| Fichier | À quoi ça sert |
|---|---|
| `demo_astuces.py` | Le bac à sable du chapitre : toutes les astuces exécutables d'un coup, section par section. Modifiez-le, relancez-le. |
| `exercice_astuces.py` | Le corrigé exécutable de l'exercice (filtrer un million de commandes en quatre versions). |
| `solution_exercice.md` | Le corrigé commenté, avec les mesures et les pièges à éviter. |
| `requirements.txt` | Les bibliothèques utilisées. |

## Démarrage

```bash
# Depuis ce dossier, dans un environnement virtuel activé (chapitre 4)
pip install -r requirements.txt

python demo_astuces.py       # la visite guidée du chapitre
python exercice_astuces.py   # le corrigé de l'exercice
```

`exercice_astuces.py` n'utilise que la bibliothèque standard : il tourne même
sans rien installer. `demo_astuces.py` saute proprement les sections NumPy et
pandas si ces bibliothèques sont absentes.

## À savoir sur les mesures

Les durées affichées viennent de **votre** machine. Ce ne sont pas les valeurs
absolues qui comptent, mais les **écarts** entre les versions : un `set` reste
des centaines de fois plus rapide qu'une `list` pour un test d'appartenance,
quel que soit le processeur.

La version naïve de l'exercice est mesurée sur un échantillon de 5 000 commandes
puis extrapolée : lancée sur le million complet, elle demanderait plusieurs
minutes. Estimer un temps d'exécution plutôt que le subir fait partie du métier.

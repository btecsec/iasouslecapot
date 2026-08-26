# Chapitre 24 — Installer et tester PyTorch

## L'énoncé

1. Installez PyTorch dans un environnement virtuel activé.
2. Affichez la version de PyTorch.
3. Créez `torch.tensor([5, 10, 15])`, affichez sa forme et son produit par 2.
4. Affichez si un GPU est disponible.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_pytorch.py` | Le corrigé exécutable, en six lignes utiles. |
| `test_exercice_pytorch.py` | Les tests unitaires (ignorés proprement si PyTorch n'est pas installé). |
| `requirements.txt` | torch, pytest. |

```bash
# Version CPU, largement suffisante pour tout le livre sauf gros entraînements
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest

python exercice_pytorch.py
pytest -q
```

## Corrigé commenté

```python
import torch

print(torch.__version__)                 # 2.x.y+cpu

t = torch.tensor([5, 10, 15])
print(t.shape)                           # torch.Size([3])
print(t * 2)                             # tensor([10, 20, 30])

print(torch.cuda.is_available())         # False sur une machine sans GPU NVIDIA
```

### Les cinq choses à remarquer

1. **Le module s'appelle `torch`, pas `pytorch`.** On installe `torch`, on
   importe `torch`. Le nom « PyTorch » n'existe que dans la documentation.
2. **`t.shape` renvoie `torch.Size([3])`**, pas `3`. Un tenseur à une dimension
   de 3 éléments. `torch.Size` se comporte comme un tuple : `t.shape[0] == 3`.
3. **`t * 2` ne fait pas de boucle.** L'opération est *vectorisée*, exactement
   comme avec NumPy (chapitre 10). C'est ce qui rend le calcul rapide, et
   transposable au GPU sans changer une ligne.
4. **`torch.cuda.is_available()` renvoyant `False` n'est pas une panne.** Sans
   GPU NVIDIA, ou avec la version CPU de torch, c'est le résultat normal. Tout
   ce livre tourne sur processeur.
5. **Le type est déduit** : `torch.tensor([5, 10, 15])` donne des entiers
   (`torch.int64`). Pour un réseau de neurones il faudra des flottants —
   d'où le `dtype=torch.float32` qui apparaîtra au chapitre 37.

### Le geste qui sert vraiment plus tard

```python
appareil = "cuda" if torch.cuda.is_available() else "cpu"
t = t.to(appareil)
```

Écrit comme ça, votre code tourne sur les deux machines sans modification.
C'est le motif standard : écrivez-le une fois, il vous servira dans tous vos
projets PyTorch.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `pip install torch` (le paquet `pytorch` sur PyPI n'est pas le bon). |
| 2 | **c** | `import torch`. |
| 3 | **b** | Un tenseur est un tableau de nombres, comme un array NumPy, avec deux pouvoirs en plus : il va sur GPU, et il mémorise ses gradients. |
| 4 | **b** | `torch.cuda.is_available()` dit si un GPU NVIDIA utilisable est présent. |

## Ce qu'il faut retenir

PyTorch, c'est NumPy plus deux superpouvoirs : le GPU et la dérivation
automatique. Si vous savez manipuler un array, vous savez déjà manipuler un
tenseur.

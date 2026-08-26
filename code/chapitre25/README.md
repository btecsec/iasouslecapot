# Chapitre 25 — Installer et tester TensorFlow/Keras

## L'énoncé

1. Installez TensorFlow dans un environnement virtuel activé.
2. Affichez sa version.
3. Créez `tf.constant([2, 4, 6])`, affichez sa forme et son produit par 3.
4. Comparez mentalement avec l'exercice PyTorch du chapitre 24.

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `exercice_keras.py` | Le corrigé exécutable, plus un mini-réseau `Sequential`. |
| `test_exercice_keras.py` | Les tests unitaires (ignorés si TensorFlow n'est pas installé). |
| `requirements.txt` | tensorflow, pytest. |

```bash
pip install tensorflow pytest
python exercice_keras.py
pytest -q
```

## Corrigé commenté

```python
import tensorflow as tf

print(tf.__version__)                 # 2.x.y

t = tf.constant([2, 4, 6])
print(t.shape)                        # (3,)
print(t * 3)                          # tf.Tensor([ 6 12 18], shape=(3,), dtype=int32)
```

### Question 4 : qu'est-ce qui change, qu'est-ce qui reste ?

| | PyTorch (ch. 11) | TensorFlow (ch. 12) |
|---|---|---|
| Import | `import torch` | `import tensorflow as tf` |
| Créer | `torch.tensor([5, 10, 15])` | `tf.constant([2, 4, 6])` |
| Forme | `torch.Size([3])` | `TensorShape([3])` |
| Multiplier | `t * 2` | `t * 3` |
| Matériel | `torch.cuda.is_available()` | `tf.config.list_physical_devices("GPU")` |

**Ce qui change** : trois noms de fonctions.
**Ce qui reste** : absolument tout le reste — la notion de tenseur, la forme,
le type déduit, la vectorisation, l'idée que le calcul peut migrer sur GPU.

C'est exactement le message du chapitre 26 : le choix du framework est une
question de contexte, pas de compétence à réapprendre de zéro.

### Les messages rouges au démarrage

```text
I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is
optimized to use available CPU instructions...
```

Une ligne qui commence par `I` est une **information** (`W` = avertissement,
`E` = erreur). TensorFlow vous dit simplement quelles instructions
processeur il utilise. Pour retrouver un terminal calme :

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # AVANT l'import de tensorflow
import tensorflow as tf
```

L'ordre compte : la variable doit être posée avant l'import, sinon elle arrive
trop tard.

### Le bonus utile : un `Sequential` minimal

```python
from tensorflow import keras

modele = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax"),
])
modele.summary()
```

Une pile de couches, dans l'ordre. C'est exactement ce que vous construirez au
chapitre 36 sur les manchots — avec, ici, 4 entrées et 3 classes en sortie.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | `pip install tensorflow` suffit : Keras est livré avec, sous `tf.keras`. |
| 2 | **b** | La convention universelle est `import tensorflow as tf`. |
| 3 | **b** | Non : ce sont le plus souvent de simples informations sur le matériel disponible. |
| 4 | **a** | `keras.Sequential([...])` décrit une pile de couches traversées l'une après l'autre. |

## Ce qu'il faut retenir

Un tenseur reste un tenseur, quel que soit le logo. Une fois le chapitre 24
digéré, le chapitre 25 ne coûte qu'un changement de vocabulaire.

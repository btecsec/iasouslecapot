"""Chapitre 25 — corrigé de l'exercice : premiers pas avec TensorFlow/Keras.

Usage :
    python exercice_keras.py
"""

from __future__ import annotations

import os

# À poser AVANT l'import de tensorflow, sinon la consigne arrive trop tard.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402


def version() -> str:
    """La version installée, par exemple '2.16.1'."""
    return tf.__version__


def creer_tenseur() -> tf.Tensor:
    """Le tenseur de l'énoncé : [2, 4, 6]."""
    return tf.constant([2, 4, 6])


def forme(tenseur: tf.Tensor) -> tuple[int, ...]:
    """La forme en tuple ordinaire : (3,)."""
    return tuple(tenseur.shape)


def tripler(tenseur: tf.Tensor) -> tf.Tensor:
    """Multiplication vectorisée, comme le `* 2` de PyTorch au chapitre 24."""
    return tenseur * 3


def gpu_disponible() -> bool:
    """L'équivalent TensorFlow de torch.cuda.is_available()."""
    return len(tf.config.list_physical_devices("GPU")) > 0


def petit_reseau(n_entrees: int = 4, n_classes: int = 3) -> keras.Model:
    """Une pile de couches : c'est tout ce qu'est un `Sequential`.

    Deux couches seulement, mais la structure est déjà celle du chapitre 36 :
    une couche cachée `relu`, une sortie `softmax` avec un neurone par classe.
    """
    return keras.Sequential(
        [
            keras.layers.Input(shape=(n_entrees,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


def main() -> None:
    print(f"Version de TensorFlow : {version()}")

    t = creer_tenseur()
    print(f"Tenseur               : {t.numpy()}")
    print(f"Forme                 : {t.shape}  (soit {forme(t)})")
    print(f"Type des éléments     : {t.dtype}")
    print(f"Multiplié par 3       : {tripler(t).numpy()}")
    print(f"GPU disponible        : {gpu_disponible()}")

    print("\n--- un Sequential minimal ---")
    petit_reseau().summary()


if __name__ == "__main__":
    main()

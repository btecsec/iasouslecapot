"""Chapitre 22 — corrigé de l'exercice : l'attention de « dort », à la main.

Le fichier calcule deux fois la même chose :
  * `attention_a_la_main`  : la version du tableau, terme à terme ;
  * `attention_matricielle`: la formule softmax(Q x K^T / sqrt(dk)) x V.

Elles doivent donner le même résultat au dernier chiffre près. C'est le seul
moyen honnête de se convaincre que la formule compacte ne cache rien.

Usage :
    python exercice_attention.py
"""

from __future__ import annotations

import numpy as np

MOTS = ["Le", "chat", "dort"]

# Les embeddings du chapitre : deux dimensions pour tenir sur une feuille.
EMBEDDINGS = {
    "Le": np.array([0.1, 0.2]),
    "chat": np.array([0.7, 0.3]),
    "dort": np.array([0.4, 0.8]),
}

# La matrice des embeddings : une ligne par mot.
X = np.array([EMBEDDINGS[mot] for mot in MOTS])

DK = X.shape[1]  # la dimension des clés, ici 2


def softmax(scores: np.ndarray) -> np.ndarray:
    """Des scores vers des poids qui somment à 1."""
    exponentielles = np.exp(scores - np.max(scores))
    return exponentielles / exponentielles.sum()


# ------------------------------------------------------------- à la main
def produits_scalaires(mot: str) -> dict[str, float]:
    """Étape 1 : la Query `mot` comparée à chacune des trois Keys."""
    q = EMBEDDINGS[mot]
    return {autre: float(np.dot(q, EMBEDDINGS[autre])) for autre in MOTS}


def mise_a_lechelle(scores: dict[str, float], racine_dk: float = None) -> dict[str, float]:
    """Étape 2 : diviser par √dk.

    Le chapitre arrondit √2 à 1.41 ; passez `racine_dk=1.41` pour retrouver
    exactement les chiffres du livre, ou laissez la valeur exacte.
    """
    racine_dk = racine_dk if racine_dk is not None else np.sqrt(DK)
    return {mot: score / racine_dk for mot, score in scores.items()}


def poids_attention(mot: str, racine_dk: float = None) -> dict[str, float]:
    """Étape 3 : les poids d'attention du mot sur toute la phrase."""
    scores = mise_a_lechelle(produits_scalaires(mot), racine_dk)
    valeurs = softmax(np.array([scores[m] for m in MOTS]))
    return dict(zip(MOTS, valeurs.tolist()))


def attention_a_la_main(mot: str, racine_dk: float = None) -> np.ndarray:
    """Étape 4 : la moyenne des embeddings, pondérée par l'attention."""
    poids = poids_attention(mot, racine_dk)
    return sum(poids[autre] * EMBEDDINGS[autre] for autre in MOTS)


# ------------------------------------------------------------ en matrices
def attention_matricielle(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """softmax(Q x K^T / sqrt(dk)) x V — toute la phrase d'un seul coup.

    K^T (la transposée) sert exactement à ça : produire d'un coup tous les
    produits scalaires de tous les mots avec tous les mots.
    """
    scores = q @ k.T / np.sqrt(k.shape[1])
    exponentielles = np.exp(scores - scores.max(axis=-1, keepdims=True))
    poids = exponentielles / exponentielles.sum(axis=-1, keepdims=True)
    return poids @ v


def main() -> None:
    print("--- attention du mot « dort » ---")

    scores = produits_scalaires("dort")
    for mot, valeur in scores.items():
        print(f"1. dort . {mot:<5} = {valeur:.2f}")

    echelle = mise_a_lechelle(scores, racine_dk=1.41)
    for mot, valeur in echelle.items():
        print(f"2. {mot:<5} -> {valeur:.3f}")

    poids = poids_attention("dort")
    for mot, valeur in poids.items():
        print(f"3. {mot:<5} : {valeur * 100:.1f} %")

    sortie = attention_a_la_main("dort")
    print(f"4. sortie = {np.round(sortie, 4)}  (annonce : [0.42, 0.48])")

    autre_que_soi = max((m for m in MOTS if m != "dort"), key=lambda m: poids[m])
    print(f"5. hors lui-meme, « dort » regarde surtout « {autre_que_soi} »")
    print("   c'est le sujet du verbe : la grammaire emerge de la geometrie")

    print("\n--- verification : la formule matricielle, toute la phrase ---")
    sorties = attention_matricielle(X, X, X)
    for mot, ligne in zip(MOTS, sorties):
        print(f"   {mot:<5} -> {np.round(ligne, 4)}")


if __name__ == "__main__":
    main()

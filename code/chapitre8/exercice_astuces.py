# -*- coding: utf-8 -*-
"""Chapitre 8 — corrigé de l'exercice : filtrer un million de commandes.

Énoncé : on a un million d'identifiants de commandes et dix mille
identifiants « déjà traités ». Il faut produire la liste des commandes
restant à traiter, en quatre versions de plus en plus efficaces.

Lancer :
    python exercice_astuces.py

Les durées affichées viennent de votre machine : ce sont les écarts
entre les versions qui comptent, pas les valeurs absolues.
"""

import random
import sys
import time
from collections.abc import Iterator

N_COMMANDES = 1_000_000
N_TRAITEES = 10_000

# La version naïve est en O(n × m) : 1 000 000 × 10 000, cela ferait dix
# milliards de comparaisons, soit plusieurs heures. On la mesure donc sur
# un petit échantillon, puis on extrapole. Savoir estimer plutôt que
# subir fait partie du métier.
TAILLE_ECHANTILLON = 5_000


def chrono(fonction, *args):
    """Exécute la fonction et renvoie (résultat, durée en secondes)."""
    debut = time.perf_counter()
    resultat = fonction(*args)
    return resultat, time.perf_counter() - debut


def fabriquer_donnees() -> tuple[list[int], list[int]]:
    """Un million de commandes, dont dix mille déjà traitées."""
    random.seed(0)                       # résultats reproductibles
    commandes = list(range(N_COMMANDES))
    deja_traitees = random.sample(commandes, N_TRAITEES)
    return commandes, deja_traitees


# --------------------------------------------------------------- étape 1
def restantes_naif(commandes: list[int], deja: list[int]) -> list[int]:
    """La version que tout le monde écrit d'abord.

    `identifiant not in deja` relit la liste `deja` du début à chaque
    tour : c'est là que se perd tout le temps.
    """
    resultat = []
    for identifiant in commandes:
        if identifiant not in deja:
            resultat.append(identifiant)
    return resultat


# --------------------------------------------------------------- étape 2
def restantes_avec_set(commandes: list[int], deja: list[int]) -> list[int]:
    """Un seul mot change : `deja` devient un set.

    La conversion coûte un passage sur les dix mille éléments, puis
    chaque test d'appartenance est quasi instantané.
    """
    deja_set = set(deja)
    resultat = []
    for identifiant in commandes:
        if identifiant not in deja_set:
            resultat.append(identifiant)
    return resultat


# --------------------------------------------------------------- étape 3
def restantes_comprehension(
    commandes: list[int], deja: list[int]
) -> list[int]:
    """La même chose en une ligne, annotée.

    Le `set(deja)` est calculé UNE fois, avant la compréhension : écrit à
    l'intérieur, il serait refabriqué à chaque tour et tout le gain
    disparaîtrait.
    """
    deja_set = set(deja)
    return [i for i in commandes if i not in deja_set]


# --------------------------------------------------------------- étape 4
def restantes_generateur(
    commandes: list[int], deja: list[int]
) -> Iterator[int]:
    """La version qui ne garde rien en mémoire.

    `yield` renvoie les identifiants un par un. Rien n'est calculé tant
    que personne ne demande la valeur suivante.
    """
    deja_set = set(deja)
    for identifiant in commandes:
        if identifiant not in deja_set:
            yield identifiant


def main():
    print("Fabrication des données…")
    commandes, deja = fabriquer_donnees()
    echantillon = commandes[:TAILLE_ECHANTILLON]

    print("\n--- Étape 1 : version naïve (sur un échantillon) ---")
    _, duree_naif = chrono(restantes_naif, echantillon, deja)
    estimation = duree_naif * N_COMMANDES / TAILLE_ECHANTILLON
    print(f"{TAILLE_ECHANTILLON:,} commandes : {duree_naif:.3f}s")
    print(f"Extrapolation à {N_COMMANDES:,} : {estimation:.0f}s "
          f"(~{estimation / 60:.0f} min)")

    print("\n--- Étape 2 : `deja` devient un set (jeu complet) ---")
    restantes, duree_set = chrono(restantes_avec_set, commandes, deja)
    print(f"{N_COMMANDES:,} commandes : {duree_set:.3f}s")
    print(f"Gain estimé : ~{estimation / duree_set:.0f}x")
    print(f"Commandes restantes : {len(restantes):,}")

    print("\n--- Étape 3 : une seule compréhension ---")
    restantes_2, duree_comp = chrono(
        restantes_comprehension, commandes, deja)
    print(f"{N_COMMANDES:,} commandes : {duree_comp:.3f}s")
    print(f"Même résultat que l'étape 2 : {restantes == restantes_2}")

    print("\n--- Étape 4 : générateur, mémoire constante ---")
    generateur = restantes_generateur(commandes, deja)
    taille_liste = sys.getsizeof(restantes)
    taille_gen = sys.getsizeof(generateur)
    print(f"Liste      : {taille_liste:>12,} octets")
    print(f"Générateur : {taille_gen:>12,} octets")
    print(f"Rapport    : ~{taille_liste / taille_gen:,.0f}x plus léger")

    # Un générateur se consomme comme une liste, mais sans la stocker.
    total = sum(1 for _ in restantes_generateur(commandes, deja))
    print(f"Comptées via le générateur : {total:,}")


if __name__ == "__main__":
    main()

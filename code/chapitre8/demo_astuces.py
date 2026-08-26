# -*- coding: utf-8 -*-
"""Chapitre 8 — toutes les astuces du chapitre, exécutables d'un coup.

Ce fichier n'est pas l'exercice : c'est le « bac à sable » du chapitre.
Chaque section reprend un exemple du livre pour que vous puissiez le
modifier et voir ce qui change.

Lancer :
    python demo_astuces.py
"""

import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import batched, islice
from pathlib import Path


def titre(texte):
    print(f"\n{'=' * 62}\n{texte}\n{'=' * 62}")


# ------------------------------------------------ écrire moins, dire mieux
def comprehensions():
    titre("Compréhensions, zip et enumerate")

    carres = [x**2 for x in range(10) if x % 2 == 0]
    print("carrés pairs :", carres)

    noms = ["Alice", "Bob", "Chris"]
    ages_liste = [25, 30, 35]
    ages = {nom: age for nom, age in zip(noms, ages_liste)}
    print("dictionnaire :", ages)

    for i, valeur in enumerate(noms, start=1):
        print(" ", i, valeur)


def fstrings():
    titre("f-strings, mode débogage et formatage")

    x = 42
    print(f"{x=}")
    print(f"{x * 2 = }")

    duree = 0.0123456
    print(f"{duree:.3f}s")
    print(f"{1234567:,}")


def morse_et_ternaire():
    titre("Opérateur morse, ternaire et `or` de repli")

    ma_liste = list(range(15))
    if (n := len(ma_liste)) > 10:
        print(f"Liste trop longue : {n} éléments")

    connecte = False
    print("statut :", "actif" if connecte else "inactif")

    nom_saisi = ""
    print("nom :", nom_saisi or "Anonyme")


def depaquetage():
    titre("Dépaqueter avec * et **")

    premier, *reste = [1, 2, 3, 4]
    print("premier :", premier, "| reste :", reste)

    a, b, *milieu, dernier = [1, 2, 3, 4, 5, 6]
    print(f"a={a} b={b} milieu={milieu} dernier={dernier}")

    def afficher_infos(nom, age):
        print(f"  {nom} a {age} ans")

    infos = {"nom": "Alice", "age": 25}
    afficher_infos(**infos)

    def tout_prendre(**kwargs):
        print("  kwargs :", kwargs)

    tout_prendre(**{"a": 1}, **{"b": 2})


# ---------------------------------------------- les bonnes structures
def set_contre_list():
    titre("set contre list : le facteur ~200")

    grande_liste = list(range(1_000_000))
    grand_set = set(grande_liste)
    cible = 999_999

    debut = time.perf_counter()
    cible in grande_liste
    duree_liste = time.perf_counter() - debut

    debut = time.perf_counter()
    cible in grand_set
    duree_set = time.perf_counter() - debut

    print(f"Dans une liste : {duree_liste:.6f}s")
    print(f"Dans un set    : {duree_set:.6f}s")
    print(f"Facteur ~{duree_liste / duree_set:.0f}x")


def get_et_counter():
    titre(".get() et Counter")

    mon_dict = {"cle": "valeur"}
    print(mon_dict.get("cle", "défaut"))
    print(mon_dict.get("absente", "défaut"))

    compteur = Counter(["a", "b", "a", "c", "b", "a"])
    print(compteur)
    print(compteur.most_common(2))


@dataclass
class Utilisateur:
    """Définie au niveau du module : son `repr` affiche alors le nom court.

    Déclarée à l'intérieur d'une fonction, elle s'afficherait
    « ma_fonction.<locals>.Utilisateur(...) ».
    """

    nom: str
    age: int


def dataclass_et_pathlib():
    titre("dataclass et pathlib")

    u = Utilisateur("Alice", 25)
    print(u)
    print("comparaison :", u == Utilisateur("Alice", 25))

    fichier = Path("donnees") / "brut" / "clients.csv"
    print("chemin  :", fichier)
    print("existe  :", fichier.exists())
    print("suffixe :", fichier.suffix)


# ------------------------------------------------------- ne pas recalculer
def memoisation():
    titre("functools.cache")

    @cache
    def fibonacci(n):
        return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)

    def fibo_sans_cache(n):
        if n < 2:
            return n
        return fibo_sans_cache(n - 1) + fibo_sans_cache(n - 2)

    debut = time.perf_counter()
    r1 = fibonacci(35)
    d1 = time.perf_counter() - debut
    print(f"{r1} en {d1:.6f}s (avec cache, n=35)")

    debut = time.perf_counter()
    r2 = fibo_sans_cache(30)
    d2 = time.perf_counter() - debut
    print(f"{r2} en {d2:.6f}s (sans cache, n=30)")


# ------------------------------------------------------ passer à l'échelle
def vectorisation():
    titre("NumPy : vectoriser plutôt que boucler")
    try:
        import numpy as np
    except ImportError:
        print("NumPy n'est pas installé : pip install numpy")
        return

    n = 5_000_000
    donnees = list(range(n))
    donnees_np = np.arange(n)

    debut = time.perf_counter()
    [x * 2 for x in donnees]
    d_boucle = time.perf_counter() - debut

    debut = time.perf_counter()
    donnees_np * 2
    d_np = time.perf_counter() - debut

    print(f"Compréhension : {d_boucle:.4f}s")
    print(f"NumPy         : {d_np:.4f}s")
    print(f"Facteur ~{d_boucle / d_np:.0f}x")


def generateurs():
    titre("Générateurs, batched et islice")

    def carres_liste(n):
        return [x**2 for x in range(n)]

    def carres_generateur(n):
        for x in range(n):
            yield x**2

    n = 1_000_000
    liste = carres_liste(n)
    gen = carres_generateur(n)
    print(f"Liste       : {sys.getsizeof(liste):,} octets")
    print(f"Générateur  : {sys.getsizeof(gen):,} octets")
    print(f"Somme : {sum(carres_generateur(n))}")

    print("lots de 3 :", list(batched(range(10), 3)))
    print("échantillon :", list(islice(carres_generateur(n), 5)))


def pandas_sans_iterrows():
    titre("pandas : fuir iterrows()")
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("pandas absent : pip install pandas numpy")
        return

    np.random.seed(0)
    df = pd.DataFrame({
        "categorie": np.random.choice(["A", "B", "C"], size=1_000_000),
        "valeur": np.random.randint(1, 100, size=1_000_000),
    })

    debut = time.perf_counter()
    [row["valeur"] * 2 for _, row in df.head(2000).iterrows()]
    d_iter = time.perf_counter() - debut

    debut = time.perf_counter()
    df["valeur"] * 2
    d_vect = time.perf_counter() - debut

    print(f"iterrows() sur 2 000 lignes : {d_iter:.4f}s")
    print(f"Vectorisé sur 1 000 000     : {d_vect:.4f}s")
    print(df.groupby("categorie")["valeur"].mean())

    # Le type `category` : trois chaînes stockées une fois, puis des entiers.
    avant = df["categorie"].memory_usage(deep=True)
    apres = df["categorie"].astype("category").memory_usage(deep=True)
    print(f"Colonne texte     : {avant:,} octets")
    print(f"Colonne category  : {apres:,} octets "
          f"(~{avant / apres:.0f}x plus légère)")


def main():
    comprehensions()
    fstrings()
    morse_et_ternaire()
    depaquetage()
    set_contre_list()
    get_et_counter()
    dataclass_et_pathlib()
    memoisation()
    vectorisation()
    generateurs()
    pandas_sans_iterrows()


if __name__ == "__main__":
    main()

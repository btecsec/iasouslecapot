# -*- coding: utf-8 -*-
"""Chapitre 9 — le code à tester.

Deux fonctions volontairement minuscules : tout l'intérêt est dans les
tests qui les accompagnent (dossier `tests/`).
"""


def moyenne(valeurs: list[float]) -> float:
    """Moyenne arithmétique.

    Lève ValueError sur une liste vide : mieux vaut une erreur claire
    qu'une ZeroDivisionError incompréhensible remontée d'ailleurs.
    """
    if not valeurs:
        raise ValueError("La liste ne peut pas être vide")
    return sum(valeurs) / len(valeurs)


def normaliser(valeurs: list[float]) -> list[float]:
    """Ramène une liste de nombres entre 0 et 1 (normalisation min-max).

    Trois décisions de conception, chacune testée dans
    `tests/test_calculs.py` :

    - liste vide  -> ValueError, comme `moyenne` (cohérence du module) ;
    - toutes les valeurs identiques -> une liste de 0.0. L'étendue vaut
      zéro, donc la division est impossible ; renvoyer des zéros est le
      choix retenu par scikit-learn dans `MinMaxScaler`, autant s'aligner
      sur ce que le lecteur croisera plus tard ;
    - le résultat est une nouvelle liste : on ne modifie jamais l'entrée
      de l'appelant.
    """
    if not valeurs:
        raise ValueError("La liste ne peut pas être vide")

    minimum = min(valeurs)
    etendue = max(valeurs) - minimum
    if etendue == 0:
        return [0.0] * len(valeurs)

    return [(v - minimum) / etendue for v in valeurs]

"""Corrigé exécutable du chapitre 6 — listes, ensembles et dictionnaires.

Une fonction par question de l'énoncé. Lancez `python exercice_commandes.py`
pour voir les sorties, `pytest -q` pour vérifier le corrigé.
"""

COMMANDES = [
    {"client": "Alice", "produit": "clavier", "prix": 45},
    {"client": "Bob", "produit": "souris", "prix": 25},
    {"client": "Alice", "produit": "écran", "prix": 180},
    {"client": "Chris", "produit": "souris", "prix": 25},
]


# --- 1. Afficher chaque commande numérotée -------------------------------
def lignes_numerotees(commandes):
    """['1. Alice a commandé un clavier (45 €)', ...]

    enumerate(..., start=1) donne le numéro d'affichage sans avoir à
    manipuler un compteur à la main ni à écrire `indice + 1`.
    """
    return [
        f"{numero}. {c['client']} a commandé un {c['produit']} ({c['prix']} €)"
        for numero, c in enumerate(commandes, start=1)
    ]


# --- 2. Les clients distincts --------------------------------------------
def clients_distincts(commandes):
    """L'ensemble des clients : les doublons disparaissent tout seuls.

    Alice apparaît deux fois dans les commandes, une seule fois ici.
    """
    return {c["client"] for c in commandes}


# --- 3. Le total par client ----------------------------------------------
def total_par_client(commandes):
    """{'Alice': 225, 'Bob': 25, 'Chris': 25}

    `.get(client, 0)` règle le cas du premier passage : sans lui, la première
    commande d'un client lèverait une KeyError sur une clé encore absente.
    """
    totaux = {}
    for c in commandes:
        totaux[c["client"]] = totaux.get(c["client"], 0) + c["prix"]
    return totaux


# --- 4. Le produit le plus cher et son acheteur --------------------------
def commande_la_plus_chere(commandes):
    """La commande dont le prix est le plus élevé.

    `key=` dit à max() sur QUOI comparer : ici le prix, pas le dictionnaire
    entier (que Python ne saurait pas comparer).
    """
    return max(commandes, key=lambda c: c["prix"])


# --- 5. Une question sans boucle ------------------------------------------
def produits_de(commandes, client):
    """L'ensemble des produits commandés par un client donné."""
    return {c["produit"] for c in commandes if c["client"] == client}


def a_commande(commandes, client, produit):
    """Répond à « X a-t-il commandé Y ? » par un test d'appartenance.

    L'ensemble est construit une fois, puis interrogé autant qu'on veut : le
    test `in` sur un set est immédiat, quelle que soit la taille.
    """
    return produit in produits_de(commandes, client)


# --- Bonus : augmenter les prix sans toucher à l'original -----------------
def prix_augmentes(commandes, pourcentage=10):
    """Une NOUVELLE liste, avec de NOUVEAUX dictionnaires.

    Le piège : `dict(c)` (ou `{**c}`) est indispensable. Sans lui on
    réutiliserait les dictionnaires d'origine, et modifier une copie
    modifierait la liste de départ — les dictionnaires sont mutables et
    partagés par référence.
    """
    facteur = 1 + pourcentage / 100
    return [{**c, "prix": round(c["prix"] * facteur, 2)} for c in commandes]


def main():
    print("1. Les commandes numérotées")
    for ligne in lignes_numerotees(COMMANDES):
        print("  ", ligne)

    clients = clients_distincts(COMMANDES)
    print(f"\n2. {len(clients)} clients distincts : {sorted(clients)}")

    print("\n3. Total par client")
    for client, total in total_par_client(COMMANDES).items():
        print(f"   {client} : {total} €")

    chere = commande_la_plus_chere(COMMANDES)
    print(f"\n4. Produit le plus cher : {chere['produit']} "
          f"({chere['prix']} €), acheté par {chere['client']}")

    print(f"\n5. Alice a-t-elle commandé une souris ? "
          f"{a_commande(COMMANDES, 'Alice', 'souris')}")
    print(f"   Et un écran ? {a_commande(COMMANDES, 'Alice', 'écran')}")

    augmentees = prix_augmentes(COMMANDES)
    print("\nBonus : prix +10 %")
    for avant, apres in zip(COMMANDES, augmentees):
        print(f"   {avant['produit']:8} {avant['prix']:>6} € -> {apres['prix']:>6} €")
    print(f"   L'originale est intacte : {COMMANDES[0]['prix']} €")


if __name__ == "__main__":
    main()

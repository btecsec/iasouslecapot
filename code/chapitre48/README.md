# Chapitre 48 — Surveiller son modèle en production

## L'énoncé

1. Listez trois indicateurs techniques à suivre.
2. Calculez moyenne et écart-type d'une mesure clé sur les données
   d'entraînement : ce sont vos références.
3. Écrivez une fonction qui compare la moyenne reçue à cette référence et
   alerte au-delà d'un seuil.
4. Donnez un exemple de dérive des données et un de dérive du concept.
5. **Défi** : comment obtiendriez-vous la « vraie réponse » plus tard ?

## Le code de ce dossier

| Fichier | À quoi ça sert |
|---|---|
| `monitoring_exercice.py` | Le script du chapitre, en affichage direct. |
| `solution.md` | Le corrigé rédigé (dérives, défi de la vérité terrain). |
| `surveillance.py` | Le corrigé en **fonctions testables** : références, alertes, indicateurs. |
| `test_surveillance.py` | Les tests, dont la démonstration du piège statistique. |

```bash
python surveillance.py
pytest -q
```

## Corrigé

### 1. Les trois indicateurs techniques

| Indicateur | Ce qu'il révèle |
|---|---|
| **Latence** (moyenne **et** p95) | la lenteur ressentie |
| **Taux d'erreur** (5xx / total) | les pannes |
| **Débit** (requêtes/min) | les pics de charge — et les chutes suspectes |

**Pourquoi le p95 et pas seulement la moyenne ?** Voici la démonstration,
tirée du script :

```text
requetes             100
latence_moyenne_ms   126.0
latence_p95_ms       846.5
```

90 requêtes à 50 ms et 10 à 900 ms. La moyenne (126 ms) laisse croire que tout
va bien ; le p95 (846 ms) montre qu'un client sur dix attend presque une
seconde. **La moyenne cache toujours les pics** — et ce sont les pics dont les
utilisateurs se plaignent.

À ces trois indicateurs techniques s'ajoutent des indicateurs **métier** :
distribution des classes prédites, taux de réponses « je ne sais pas »,
proportion d'entrées hors plage. Un modèle peut être parfaitement rapide et
disponible tout en prédisant n'importe quoi.

### 2. Les références

```python
reference = calculer_reference(df_train, "masse")
# moyenne 4176.9 g, ecart-type 791.4 g
```

**Elles se calculent une fois, sur les données d'entraînement, et ne bougent
plus.** C'est pour cela que `Reference` est un `dataclass(frozen=True)` : si
vous les recalculiez sur les données de production, la « normale » suivrait la
dérive, et vous ne détecteriez jamais rien.

### 3. La fonction d'alerte — et le piège statistique

Le script du chapitre compare l'écart à `2 × écart-type`. C'est intuitif, et
c'est **beaucoup trop peu sensible**. Voici les deux formules sur les mêmes
données :

| Lot reçu (200 manchots) | Formule naïve | Formule correcte | Verdict |
|---|---|---|---|
| production normale (4 250 g) | +0,10 σ | +1,46 σ | rien à signaler |
| nouvelle colonie (4 600 g) | +0,49 σ | **+6,91 σ** | dérive |
| jeunes manchots (2 900 g) | −1,71 σ | **−24,22 σ** | dérive massive |
| capteur en panne (0 g) | −5,28 σ | **−74,64 σ** | panne évidente |

Ligne 3 : une chute de **1 300 grammes** sur la masse moyenne — un changement
de population énorme — **ne déclenche pas** l'alerte naïve.

**Pourquoi ?** Parce qu'on compare une *moyenne de 200 mesures* à l'écart-type
d'un *individu*. Ce sont deux questions différentes :

- « ce manchot-ci est-il inhabituel ? » → écart-type individuel (σ) ;
- « ce **lot** de 200 manchots est-il anormal ? » → écart-type de la moyenne,
  qui vaut **σ / √n**, l'*erreur type*.

Sur 200 mesures, l'erreur type est √200 ≈ 14 fois plus petite que σ. Une
moyenne est bien plus stable qu'un individu, donc un petit écart de moyenne est
déjà très significatif.

```python
def zscore_moyenne(self, moyenne_observee, n):
    erreur_type = self.ecart_type / np.sqrt(n)
    return (moyenne_observee - self.moyenne) / erreur_type
```

Conséquence pratique : **le seuil dépend de la taille du lot**. Un contrôle sur
10 000 requêtes détecte des dérives invisibles sur 20 — et déclenche aussi plus
de fausses alertes. En production, on surveille des fenêtres de taille fixe
(l'heure écoulée, le dernier millier de requêtes) pour que le seuil garde le
même sens d'un jour à l'autre.

Deux détails d'implémentation qui comptent :

- la fonction **renvoie** une `Alerte` au lieu d'imprimer : un `print` dans un
  conteneur n'alerte personne. C'est cet objet qu'on sérialise vers Slack, un
  e-mail ou Prometheus ;
- `bool(...)` autour de la comparaison : sans lui, NumPy renvoie un `np.bool_`,
  qui casse un `is True` et se sérialise mal en JSON.

### 4. Dérive des données ou dérive du concept ?

| | Dérive des données | Dérive du concept |
|---|---|---|
| Ce qui change | la **forme des entrées** | le **lien entrée → réponse** |
| Manchots | un nouveau protocole pèse les juvéniles : la masse moyenne chute | le réchauffement modifie la morphologie : à mesures égales, l'espèce n'est plus la même |
| Banque | une appli mobile attire une clientèle plus jeune | les fraudeurs passent des gros virements aux micro-transactions |
| Détectable sans étiquettes ? | **oui** — il suffit de regarder les entrées | **non** — il faut connaître les vraies réponses |

Cette dernière ligne est la plus importante : la dérive des données se
surveille gratuitement et en continu ; la dérive du concept ne se voit que
lorsque la vérité terrain arrive. D'où la question 5.

### 5. Défi : obtenir la vraie réponse

Trois mécanismes, du moins au plus coûteux :

1. **Le retour naturel, différé.** Dans certains domaines, la réalité arrive
   toute seule : le client a-t-il remboursé son prêt (6 mois), le colis
   est-il arrivé (3 jours), l'utilisateur a-t-il cliqué (immédiat). Il suffit
   de **stocker chaque prédiction avec un identifiant**, et de la rapprocher
   plus tard du résultat. Si vous ne le faites pas dès le départ, ce sera
   impossible rétrospectivement.
2. **Le retour utilisateur.** Un pouce en haut / en bas, un « signaler comme
   spam ». Gratuit, mais **biaisé** : les mécontents répondent bien plus que
   les satisfaits.
3. **L'échantillonnage humain.** Un expert vérifie 1 % des prédictions tirées
   au hasard. Coûteux, mais c'est la seule méthode non biaisée — et la seule
   qui fonctionne quand la vérité n'arrive jamais d'elle-même.

En pratique, on combine : le retour naturel pour le volume, l'échantillonnage
humain pour l'étalonnage.

## Corrigé du quiz

| # | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le monde change, alors que le modèle a appris sur des données passées. |
| 2 | **a** | La dérive des données : les entrées ne ressemblent plus à celles de l'entraînement. |
| 3 | **b** | La dérive du concept : la relation entre les entrées et la bonne réponse a changé. |
| 4 | **b** | Une alerte prévient automatiquement l'équipe quand un seuil est franchi — personne ne regarde un tableau de bord en permanence. |

## Ce qu'il faut retenir

Surveillez trois choses : la santé technique (latence p95, erreurs), la forme
des entrées (dérive des données), et la qualité réelle quand la vérité arrive
(dérive du concept). Et vérifiez votre formule d'alerte : comparer une moyenne
à un écart-type individuel laisse passer des dérives massives.

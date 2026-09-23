# Chapitre 28 — Entraîner sans carte graphique — Google Colab

Chapitre de méthode : **pas de code propre à ce dossier**. Le corrigé
ci-dessous se recopie tel quel dans un notebook Colab.

Le réflexe à emporter : avant de brûler des heures de GPU, faites tourner
votre boucle **à vide**, sans charger le moindre modèle. Trente secondes
suffisent à valider la tuyauterie — découpage, tranches, reprise.

## Corrigé de l'exercice

**1 et 2 — le GPU et le Drive.** Rien à corriger : ce sont des manipulations.
Notez seulement le modèle et la mémoire obtenus, ils varient selon l'affluence.

**3 — la boucle reprenable.** Le point à vérifier est qu'elle saute ce qui
existe déjà :

```python
import os, time

DOSSIER = "essai"
os.makedirs(DOSSIER, exist_ok=True)

def chemin(numero):
    return os.path.join(DOSSIER, f"tranche_{numero:05d}.txt")

for numero in range(20):
    if os.path.exists(chemin(numero)):      # deja fait, on passe
        continue
    time.sleep(1)
    temporaire = chemin(numero) + ".tmp"
    with open(temporaire, "w", encoding="utf-8") as f:
        f.write(f"tranche {numero}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporaire, chemin(numero))
    print("écrit", numero)
```

Interrompez au milieu, relancez : l'affichage repart au bon numéro.

**4 — l'écriture non atomique.** Remplacez le bloc ci-dessus par un simple
`open(chemin(numero), "w")` et interrompez pendant une écriture. Le fichier
**existe** mais il est vide ou tronqué. Au redémarrage, la boucle le voit,
le considère comme fait, et le saute : le trou est définitif et silencieux.

C'est tout l'intérêt de `os.replace()` : le fichier final n'apparaît que
complet, jamais entre les deux.

**5 — le trousseau.** `userdata.get('HF_TOKEN')` relit la valeur sans qu'elle
apparaisse nulle part dans le notebook. Un `grep` sur le fichier `.ipynb` ne
doit rien trouver — c'est le test qui compte, puisque c'est ce fichier que
vous partagerez.

**6 — le temps restant.** Une seule ligne, mais elle change tout sur un
travail de plusieurs heures :

```python
depart, faites = time.time(), 0
# ... dans la boucle, apres chaque tranche :
faites += 1
reste = (total - faites) * (time.time() - depart) / faites
print(f"reste ~{reste / 60:.0f} min")
```

La moyenne glissante depuis le départ vaut mieux que la durée de la dernière
tranche : elle absorbe les à-coups de la machine partagée.

## Réponses du quiz

| Question | Réponse | Pourquoi |
|---|---|---|
| 1 | **b** | Le disque de la session est effacé. Seul ce qui a été écrit sur le Drive monté survit. |
| 2 | **b** | Le renommage est atomique : il réussit entièrement ou pas du tout. Un fichier à moitié écrit ne peut donc jamais passer pour un fichier fini. |
| 3 | **c** | Poids, optimiseur et itération. Adam garde deux moyennes mobiles par paramètre ; sans elles, la perte remonte à la reprise. |
| 4 | **c** | Le trousseau. Un notebook est un fichier que vous partagerez, et un jeton reste dans l'historique même après suppression. |

"""Chapitre 48 — la surveillance, en fonctions testables.

`monitoring_exercice.py` (le script du chapitre) affiche ses résultats ; ici,
les mêmes calculs **renvoient** une structure. C'est ce qui permet de les
tester, et surtout de les brancher dans une vraie API : une alerte qui ne sait
que faire un `print` n'alerte personne.

Usage :
    python surveillance.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reference:
    """Ce qu'on a observé à l'entraînement : la « normale » du modèle."""

    colonne: str
    moyenne: float
    ecart_type: float

    def zscore(self, valeur: float) -> float:
        """De combien d'écarts-types **un individu** s'éloigne-t-il du normal ?

        C'est la formule du script du chapitre. Elle répond à la question
        « ce manchot-ci est-il inhabituel ? ».
        """
        if self.ecart_type == 0:
            return 0.0
        return (valeur - self.moyenne) / self.ecart_type

    def zscore_moyenne(self, moyenne_observee: float, n: int) -> float:
        """De combien d'écarts-types **une moyenne de n valeurs** s'éloigne-t-elle ?

        C'est une autre question, et c'est celle de la surveillance : « le lot
        de 200 manchots reçus ce matin est-il anormal ? ».

        Une moyenne est bien plus stable qu'un individu : son écart-type vaut
        `sigma / racine(n)` — c'est l'erreur type. Sur 200 mesures, un écart de
        1,7 sigma **individuel** représente 24 sigma pour la moyenne : énorme,
        et parfaitement invisible avec la première formule.
        """
        if self.ecart_type == 0 or n <= 0:
            return 0.0
        erreur_type = self.ecart_type / np.sqrt(n)
        return (moyenne_observee - self.moyenne) / erreur_type


@dataclass
class Alerte:
    """Le résultat d'un contrôle. `declenchee` est ce qu'on teste."""

    declenchee: bool
    colonne: str
    moyenne_observee: float
    zscore: float
    seuil: float
    message: str = ""


# --------------------------------------------------------------- question 2
def calculer_reference(df_train: pd.DataFrame, colonne: str) -> Reference:
    """Les références se calculent **une fois**, sur les données d'entraînement.

    Les recalculer sur les données de production reviendrait à redéfinir la
    normale au fur et à mesure qu'elle dérive — et à ne jamais rien détecter.
    """
    return Reference(
        colonne=colonne,
        moyenne=float(df_train[colonne].mean()),
        ecart_type=float(df_train[colonne].std()),
    )


# --------------------------------------------------------------- question 3
def verifier_derive(
    donnees: pd.DataFrame,
    reference: Reference,
    seuil_sigma: float = 2.0,
    tenir_compte_de_la_taille: bool = True,
) -> Alerte:
    """Compare la moyenne reçue à la référence, et alerte au-delà du seuil.

    `tenir_compte_de_la_taille=True` (recommandé) compare la moyenne à
    l'erreur type ; à False, on retrouve la formule naïve du script du
    chapitre, beaucoup moins sensible.
    """
    colonne = donnees[reference.colonne]
    moyenne = float(colonne.mean())
    zscore = (
        reference.zscore_moyenne(moyenne, len(colonne.dropna()))
        if tenir_compte_de_la_taille
        else reference.zscore(moyenne)
    )
    # `bool(...)` : sans lui, NumPy renvoie un np.bool_, qui échoue sur un
    # `is True` et se sérialise mal en JSON.
    declenchee = bool(abs(zscore) > seuil_sigma)

    message = (
        f"Derive sur {reference.colonne} : moyenne {moyenne:.1f} contre "
        f"{reference.moyenne:.1f} attendu ({zscore:+.2f} sigma)"
        if declenchee
        else f"{reference.colonne} : rien a signaler ({zscore:+.2f} sigma)"
    )
    return Alerte(declenchee, reference.colonne, moyenne, zscore, seuil_sigma, message)


# --------------------------------------------------------------- question 1
@dataclass
class Indicateurs:
    """Les trois indicateurs techniques de la question 1, tenus à jour."""

    latences_ms: list[float] = field(default_factory=list)
    erreurs: int = 0
    requetes: int = 0

    def enregistrer(self, latence_ms: float, en_erreur: bool = False) -> None:
        self.requetes += 1
        self.latences_ms.append(latence_ms)
        if en_erreur:
            self.erreurs += 1

    @property
    def latence_moyenne(self) -> float:
        return float(np.mean(self.latences_ms)) if self.latences_ms else 0.0

    @property
    def latence_p95(self) -> float:
        """Le p95 dit ce que vit le client le plus mal servi sur vingt.

        La moyenne, elle, masque les pics : dix requêtes à 50 ms et une à
        3 secondes donnent une moyenne rassurante de 320 ms.
        """
        return float(np.percentile(self.latences_ms, 95)) if self.latences_ms else 0.0

    @property
    def taux_erreur(self) -> float:
        return self.erreurs / self.requetes if self.requetes else 0.0

    def resume(self) -> dict[str, float]:
        return {
            "requetes": self.requetes,
            "latence_moyenne_ms": round(self.latence_moyenne, 1),
            "latence_p95_ms": round(self.latence_p95, 1),
            "taux_erreur": round(self.taux_erreur, 4),
        }


def main() -> None:
    generateur = np.random.default_rng(42)

    entrainement = pd.DataFrame(
        {"masse": generateur.normal(4200, 800, size=1000)}
    )
    reference = calculer_reference(entrainement, "masse")
    print(
        f"Reference etablie : moyenne {reference.moyenne:.1f} g, "
        f"ecart-type {reference.ecart_type:.1f} g\n"
    )

    cas = {
        "production normale": generateur.normal(4250, 800, size=200),
        "derive douce (nouvelle colonie)": generateur.normal(4600, 800, size=200),
        "derive franche (jeunes manchots)": generateur.normal(2900, 800, size=200),
        "capteur en panne (tout a zero)": np.zeros(200),
    }

    for nom, masses in cas.items():
        lot = pd.DataFrame({"masse": masses})
        alerte = verifier_derive(lot, reference)
        naive = verifier_derive(lot, reference, tenir_compte_de_la_taille=False)
        marque = "ALERTE" if alerte.declenchee else "  ok  "
        print(f"[{marque}] {nom:<34} {alerte.zscore:+8.2f} sigma sur la moyenne")
        print(f"{'':<9} {'':<34} {naive.zscore:+8.2f} sigma avec la formule naive")

    print(
        "\n  La formule naive compare une moyenne de 200 mesures a l'ecart-type\n"
        "  d'un individu : elle laisse passer une derive pourtant massive."
    )

    print("\n--- indicateurs techniques (question 1) ---")
    indicateurs = Indicateurs()
    for _ in range(90):
        indicateurs.enregistrer(generateur.normal(45, 8))
    for _ in range(10):  # dix requetes lentes, toutes en erreur
        indicateurs.enregistrer(generateur.normal(900, 100), en_erreur=True)
    for cle, valeur in indicateurs.resume().items():
        print(f"  {cle:<20} {valeur}")
    print(
        "\n  La moyenne (126 ms) reste rassurante alors qu'une requete sur dix\n"
        "  prend pres d'une seconde : c'est le p95 (846 ms) qui le revele."
    )


if __name__ == "__main__":
    main()

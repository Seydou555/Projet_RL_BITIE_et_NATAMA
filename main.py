# -*- coding: utf-8 -*-
"""
==============================================================================
  PROJET 4 : Q-LEARNING CONTRE SARSA SUR FROZENLAKE ET CLIFFWALKING
  Module Apprentissage par Renforcement, Université Aube Nouvelle, 2025-2026
==============================================================================

Tout le projet tient dans ce fichier : les trois algorithmes, les deux
environnements, les expériences du rapport, la démonstration visuelle et le
rejeu d'une politique sauvegardée.

Les algorithmes sont écrits en NumPy pur. Gymnasium ne sert qu'à fournir les
environnements, aucune bibliothèque d'apprentissage par renforcement n'intervient
dans l'apprentissage.

  INSTALLATION
  ------------
    pip install -r requirements.txt

  Sous Windows, si « pip » n'est pas reconnu :  python -m pip install -r requirements.txt

  EXÉCUTION SIMPLE
  ----------------
  Lancer le fichier python main.py
  Déroulement de la démonstration complète : FrozenLake d'abord, CliffWalking ensuite.
  Dans chaque cas les trois algorithmes apprennent côte à côte, et l'on voit l'agent agir au hasard,
  puis progresser au fil de l'entraînement, puis jouer la politique apprise.

  Chaque démonstration écrit dans resultats/ sa vidéo MP4, sa courbe d'apprentissage,
  la politique de chaque agent et, sur CliffWalking, les trajectoires comparées.
  Les courbes multi-graines du rapport viennent des commandes ci-dessous, plus longues.
  --------------------------------------------------------------------------
  DANS SPYDER : une seule chose a regler UNE fois pour que la fenetre s'anime :
     Outils -> Preferences -> Console IPython -> Graphiques -> Backend
        -> choisis "Automatique"  (et PAS "En ligne / Inline")
     Puis redemarre le noyau (Console -> Redemarrer le noyau).
  Le script essaie deja de forcer ce reglage tout seul, mais si l'image ne
  bouge pas, c'est ce parametre qu'il faut changer.

  ORGANISATION DU FICHIER
  -----------------------
    1. Environnements          configuration des deux tâches
    2. Agents                  classe de base, Q-learning, SARSA, Expected SARSA
    3. Entraînement            boucle unifiée, évaluation gloutonne, multi-graines
    4. Figures                 courbes, trajectoires, politiques
    5. Expériences             les trois expériences du rapport
    6. Démonstration visuelle  animation matplotlib, export MP4 ou GIF
    7. Rejeu                   relecture d'une Q-table sauvegardée
    8. Ligne de commande
==============================================================================
"""

import argparse
import csv
import hashlib
import os
import sys

try:
    import numpy as np
    import gymnasium as gym
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as manquant:
    # Message explicite plutôt qu'une trace d'erreur : c'est presque toujours une
    # dépendance absente, et la commande d'installation est la seule chose utile ici.
    raise SystemExit(
        "Dépendance manquante : {}\n\n"
        "Installez les paquets du projet avec :\n"
        "    pip install -r requirements.txt\n"
        "ou, si pip n'est pas reconnu :\n"
        "    python -m pip install -r requirements.txt".format(manquant.name))

# Enchaînement joué quand le script est lancé sans argument (F5 dans Spyder). Chaque
# entrée est un environnement de démonstration : retirez-en une ou changez l'ordre si
# vous ne voulez en voir qu'une seule.
PARCOURS_PAR_DEFAUT = ["frozenlake4x4", "cliffwalking"]

ICI = os.path.dirname(os.path.abspath(__file__))
RESULTATS = os.path.join(ICI, "resultats")
CACHE = os.path.join(RESULTATS, "cache")

# =============================================================================
#  1. ENVIRONNEMENTS
# =============================================================================
ENVIRONNEMENTS = {
    "frozenlake4x4": dict(
        candidats=["FrozenLake-v1"],
        kwargs=dict(map_name="4x4", is_slippery=True),
        max_pas=100,
        # Succès = avoir ramassé la récompense finale de 1 (atteindre le cadeau).
        succes=lambda r, term: bool(r == 1.0),
        grille=(4, 4),
        titre="FrozenLake 4x4 glissant",
    ),
    "frozenlake8x8": dict(
        candidats=["FrozenLake-v1"],
        kwargs=dict(map_name="8x8", is_slippery=True),
        max_pas=200,
        succes=lambda r, term: bool(r == 1.0),
        grille=(8, 8),
        titre="FrozenLake 8x8 glissant",
    )
}


# Repères de la grille 4 x 12 de CliffWalking.
# DEPART, ARRIVEE = 36, 47
# FALAISE = set(range(37, 47))


def creer_env(nom_env, render_mode=None):
    """Instancie l'environnement en essayant les identifiants dans l'ordre."""
    cfg = ENVIRONNEMENTS[nom_env]
    derniere_erreur = None
    for eid in cfg["candidats"]:
        try:
            return gym.make(eid, render_mode=render_mode, **cfg["kwargs"]), eid, cfg
        except Exception as err:  # version absente ou dépréciée
            derniere_erreur = err
    raise SystemExit("Aucune version disponible pour {} ({}). Essayez : "
                     "pip install -U gymnasium".format(nom_env, derniere_erreur))


# =============================================================================
#  2. AGENTS
# =============================================================================
class AgentTabulaire:
    """
    Contient la table Q, la politique d'exploration et le calendrier de epsilon.
    La seule chose que les sous-classes redéfinissent, c'est la règle de mise à jour.
    """

    # Passe à True chez SARSA : seul lui a besoin de A' AVANT sa mise à jour.
    besoin_action_suivante = False
    nom = "base"

    def __init__(self, n_etats, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_debut=1.0, epsilon_fin=0.01, epsilon_decr=0.999,
                 mode_epsilon="exponentiel", graine=0):
        self.n_etats, self.n_actions = n_etats, n_actions
        self.alpha, self.gamma = alpha, gamma

        # Initialisation à zéro. Sur nos deux environnements toutes les récompenses
        # sont négatives ou nulles, donc zéro est déjà une valeur optimiste : une
        # action jamais essayée paraît meilleure que celles déjà évaluées.
        self.Q = np.zeros((n_etats, n_actions), dtype=np.float64)

        self.epsilon_debut, self.epsilon_fin = epsilon_debut, epsilon_fin
        self.epsilon_decr, self.mode_epsilon = epsilon_decr, mode_epsilon
        self.epsilon = epsilon_debut
        self.rng = np.random.default_rng(graine)

    # ---- choix de l'action ----------------------------------------------------
    def argmax_aleatoire(self, etat):
        """
        argmax avec départage aléatoire des ex aequo.

        np.argmax renvoie toujours le premier maximum. Avec une table initialisée
        à zéro, l'agent choisirait systématiquement l'action 0 tant qu'il n'a rien
        appris, soit « haut » sur CliffWalking, ce qui biaise les premiers épisodes.
        """
        valeurs = self.Q[etat]
        meilleures = np.flatnonzero(valeurs == valeurs.max())
        return int(meilleures[0]) if meilleures.size == 1 else int(self.rng.choice(meilleures))

    def choisir(self, etat):
        """Avec probabilité epsilon : action au hasard. Sinon : action gloutonne."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.argmax_aleatoire(etat)

    def probabilites_politique(self, etat):
        """Distribution pi(a | etat) de la politique epsilon-gloutonne."""
        p = np.full(self.n_actions, self.epsilon / self.n_actions)
        valeurs = self.Q[etat]
        meilleures = np.flatnonzero(valeurs == valeurs.max())
        p[meilleures] += (1.0 - self.epsilon) / meilleures.size
        return p

    # ---- calendrier de epsilon, appelé une fois par épisode --------------------
    def maj_epsilon(self, episode, nb_episodes):
        if self.mode_epsilon == "fixe":
            return
        if self.mode_epsilon == "exponentiel":
            self.epsilon = max(self.epsilon_fin, self.epsilon * self.epsilon_decr)
        elif self.mode_epsilon == "lineaire":
            frac = min(1.0, (episode + 1) / max(1, nb_episodes))
            self.epsilon = self.epsilon_debut + frac * (self.epsilon_fin - self.epsilon_debut)
        elif self.mode_epsilon == "inverse":
            # epsilon ~ 1/t : décroissance lente, compatible avec les conditions
            # de convergence qui demandent une exploration maintenue.
            self.epsilon = max(self.epsilon_fin, self.epsilon_debut / (1.0 + episode))
        else:
            raise ValueError("mode_epsilon inconnu : " + str(self.mode_epsilon))

    # ---- mise à jour : c'est ICI que les trois algorithmes diffèrent -----------
    def maj(self, etat, action, recompense, etat_suivant, action_suivante, termine):
        raise NotImplementedError


class QLearning(AgentTabulaire):
    """
    Q(S,A) <- Q(S,A) + alpha [ R + gamma max_a Q(S',a) - Q(S,A) ]

    La cible utilise le maximum sur S', donc la valeur de la politique GLOUTONNE,
    alors que l'action jouée vient de la politique epsilon-gloutonne. Politique de
    comportement et politique cible diffèrent : c'est le sens de « hors-politique ».

    """
    nom = "Q-learning"

    def maj(self, etat, action, recompense, etat_suivant, action_suivante, termine):
        # Un état terminal vaut 0 par définition, d'où le facteur (not termine).
        cible = recompense + self.gamma * np.max(self.Q[etat_suivant]) * (not termine)
        self.Q[etat, action] += self.alpha * (cible - self.Q[etat, action])


class SARSA(AgentTabulaire):
    """
    Q(S,A) <- Q(S,A) + alpha [ R + gamma Q(S',A') - Q(S,A) ]

    A' est l'action réellement jouée ensuite, tirée de la même politique
    epsilon-gloutonne. La cible intègre donc le coût de l'exploration : les cases
    bordant la falaise voient leur valeur baissée par les chutes accidentelles, ce
    qui pousse la politique vers le chemin sûr de la ligne du haut.
    """
    nom = "SARSA"
    besoin_action_suivante = True

    def maj(self, etat, action, recompense, etat_suivant, action_suivante, termine):
        q_suivant = 0.0 if termine else self.Q[etat_suivant, action_suivante]
        cible = recompense + self.gamma * q_suivant
        self.Q[etat, action] += self.alpha * (cible - self.Q[etat, action])


class ExpectedSARSA(AgentTabulaire):
    """
    Q(S,A) <- Q(S,A) + alpha [ R + gamma sum_a pi(a|S') Q(S',a) - Q(S,A) ]

    L'échantillon Q(S',A') de SARSA est remplacé par son espérance sous pi. La cible
    ne dépend plus du tirage de A', donc sa variance chute et l'algorithme supporte
    des alpha plus grands. Avec epsilon = 0, l'espérance se réduit au maximum et
    Expected SARSA redevient exactement Q-learning.
    """
    nom = "Expected SARSA"

    def maj(self, etat, action, recompense, etat_suivant, action_suivante, termine):
        if termine:
            valeur_attendue = 0.0
        else:
            p = self.probabilites_politique(etat_suivant)
            valeur_attendue = float(np.dot(p, self.Q[etat_suivant]))
        cible = recompense + self.gamma * valeur_attendue
        self.Q[etat, action] += self.alpha * (cible - self.Q[etat, action])


AGENTS = {"qlearning": QLearning, "sarsa": SARSA, "expected_sarsa": ExpectedSARSA}
NOMS = {c: AGENTS[c].nom for c in AGENTS}
ALGOS = ["qlearning", "sarsa", "expected_sarsa"]
COULEURS = {"Q-learning": "#2a78d6", "SARSA": "#e0662b", "Expected SARSA": "#1baf7a"}


# =============================================================================
#  3. ENTRAÎNEMENT ET ÉVALUATION
# =============================================================================
def decroissance_pour(episodes, eps_debut=1.0, eps_fin=0.01, fraction=0.7):
    """
    Facteur multiplicatif tel que epsilon atteigne eps_fin après `fraction` des
    épisodes. Fixer 0,999 au jugé convient à 5 000 épisodes et rate complètement
    à 50 000 : on déduit le facteur du budget réellement alloué.
    """
    return (eps_fin / eps_debut) ** (1.0 / max(1, int(fraction * episodes)))


def entrainer(nom_env, nom_agent, episodes=5000, graine=0, hyper=None):
    """
    Entraîne un agent et renvoie l'historique épisode par épisode.

    La boucle est unique pour les trois algorithmes. Seul SARSA a besoin de choisir
    A' avant sa mise à jour ; les deux autres mettent à jour puis choisissent sur la
    table déjà corrigée. Écrire une seule boucle sans cette distinction
    introduirait un décalage silencieux dans Q-learning.
    """
    env, eid, cfg = creer_env(nom_env)
    env.action_space.seed(graine)
    agent = AGENTS[nom_agent](env.observation_space.n, env.action_space.n,
                              graine=graine, **dict(hyper or {}))

    recompenses = np.zeros(episodes)
    succes = np.zeros(episodes)
    longueurs = np.zeros(episodes)

    for ep in range(episodes):
        # La graine n'est passée qu'au premier reset : l'environnement poursuit
        # ensuite sa propre séquence, ce qui donne des épisodes variés tout en
        # restant reproductible d'une exécution à l'autre.
        etat, _ = env.reset(seed=graine if ep == 0 else None)
        action = agent.choisir(etat)
        fini, pas, total, reussi = False, 0, 0.0, False

        while not fini and pas < cfg["max_pas"]:
            etat_suivant, r, termine, tronque, _ = env.step(action)
            fini = termine or tronque

            if agent.besoin_action_suivante:  # SARSA
                action_suivante = agent.choisir(etat_suivant)
                agent.maj(etat, action, r, etat_suivant, action_suivante, termine)
                action = action_suivante
            else:  # les deux autres
                agent.maj(etat, action, r, etat_suivant, None, termine)
                action = agent.choisir(etat_suivant)

            etat = etat_suivant
            total += r
            pas += 1
            if cfg["succes"](r, termine):
                reussi = True

        agent.maj_epsilon(ep, episodes)
        recompenses[ep], succes[ep], longueurs[ep] = total, float(reussi), pas

    env.close()
    return dict(Q=agent.Q, recompenses=recompenses, succes=succes,
                longueurs=longueurs, env_id=eid)


def evaluer_glouton(nom_env, Q, n_episodes=100, graine=1234):
    """
    Rejoue la politique apprise sans exploration ni apprentissage.

    C'est cette mesure qui doit figurer dans le rapport. La courbe d'entraînement
    mesure une politique qui explore encore, donc pas celle qu'on prétend livrer.
    """
    env, _, cfg = creer_env(nom_env)
    rng = np.random.default_rng(graine)
    retours, longueurs, reussites = [], [], 0

    for _ in range(n_episodes):
        etat, _ = env.reset(seed=int(rng.integers(1 << 31)))
        fini, pas, total, reussi = False, 0, 0.0, False
        while not fini and pas < cfg["max_pas"]:
            valeurs = Q[etat]
            meilleures = np.flatnonzero(valeurs == valeurs.max())
            action = int(rng.choice(meilleures)) if meilleures.size > 1 else int(meilleures[0])
            etat, r, termine, tronque, _ = env.step(action)
            fini = termine or tronque
            total += r
            pas += 1
            if cfg["succes"](r, termine):
                reussi = True
        retours.append(total)
        longueurs.append(pas)
        reussites += int(reussi)

    env.close()
    return dict(taux_succes=100.0 * reussites / n_episodes,
                retour_moyen=float(np.mean(retours)),
                retour_ecart=float(np.std(retours)),
                longueur_moyenne=float(np.mean(longueurs)))


def _cle_cache(nom_env, nom_agent, episodes, hyper, graine):
    """Identifiant stable d'un entraînement : deux réglages différents, deux clés."""
    signature = repr((nom_env, nom_agent, episodes, sorted((hyper or {}).items())))
    return "{}_{}_{}_g{}".format(nom_env, nom_agent,
                                 hashlib.md5(signature.encode()).hexdigest()[:8], graine)


def campagne(nom_env, nom_agent, episodes, graines, hyper=None, verbeux=True, cache=True):
    """
    Relance le même entraînement sur plusieurs graines et empile les historiques.

    Le sujet impose au moins 10 graines sur FrozenLake : sur un environnement
    stochastique, deux graines peuvent donner des résultats opposés par simple
    chance de tirage.

    Chaque graine terminée est écrite dans resultats/cache/. Une campagne
    interrompue reprend là où elle s'était arrêtée, ce qui compte pour FrozenLake
    8x8 qui demande une vingtaine de minutes. Supprimer le dossier force un
    recalcul complet.
    """
    if cache:
        os.makedirs(CACHE, exist_ok=True)

    rec, suc, lon, tables, evals = [], [], [], [], []
    for g in graines:
        chemin = os.path.join(CACHE, _cle_cache(nom_env, nom_agent, episodes, hyper, g) + ".npz")

        if cache and os.path.exists(chemin):
            d = np.load(chemin)
            if verbeux:
                print("   [{}] {} | graine {} (cache)".format(nom_env, nom_agent, g))
            rec.append(d["recompenses"]), suc.append(d["succes"]), lon.append(d["longueurs"])
            tables.append(d["Q"])
            evals.append({k: float(d[k]) for k in
                          ("taux_succes", "retour_moyen", "retour_ecart", "longueur_moyenne")})
            continue

        if verbeux:
            print("   [{}] {} | graine {}".format(nom_env, nom_agent, g))
        h = entrainer(nom_env, nom_agent, episodes=episodes, graine=g, hyper=hyper)
        ev = evaluer_glouton(nom_env, h["Q"], n_episodes=100, graine=10_000 + g)
        rec.append(h["recompenses"]), suc.append(h["succes"]), lon.append(h["longueurs"])
        tables.append(h["Q"])
        evals.append(ev)
        if cache:
            np.savez_compressed(chemin, recompenses=h["recompenses"], succes=h["succes"],
                                longueurs=h["longueurs"], Q=h["Q"], **ev)

    return dict(recompenses=np.array(rec), succes=np.array(suc), longueurs=np.array(lon),
                tables=tables, evaluations=evals, graines=list(graines),
                nom_agent=nom_agent, nom_env=nom_env)


def resume_evaluations(resultat):
    """Moyenne et écart-type inter-graines des évaluations gloutonnes finales."""
    ev = resultat["evaluations"]
    taux = np.array([e["taux_succes"] for e in ev])
    retours = np.array([e["retour_moyen"] for e in ev])
    longueurs = np.array([e["longueur_moyenne"] for e in ev])
    return dict(succes_moy=taux.mean(), succes_std=taux.std(),
                retour_moy=retours.mean(), retour_std=retours.std(),
                longueur_moy=longueurs.mean(), longueur_std=longueurs.std())


# =============================================================================
#  4. FIGURES ET TABLEAUX
# =============================================================================
FLECHES = {0: "^", 1: ">", 2: "v", 3: "<"}  # CliffWalking : haut, droite, bas, gauche


def fichier_resultat(nom):
    os.makedirs(RESULTATS, exist_ok=True)
    return os.path.join(RESULTATS, nom)


def moyenne_glissante(x, fenetre=100):
    """Sans lissage, les courbes brutes sont illisibles."""
    if fenetre <= 1 or len(x) < fenetre:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(fenetre) / fenetre, mode="valid")


def courbe_multi_graines(ax, donnees, label, couleur=None, fenetre=100, echelle=1.0):
    """
    Trace la moyenne inter-graines et une bande à plus ou moins un écart-type.

    Chaque graine est lissée AVANT le calcul de la moyenne. Lisser après
    mélangerait le bruit interne d'une graine avec la dispersion entre graines,
    et la bande n'aurait plus le sens qu'on lui prête.
    """
    lisse = np.array([moyenne_glissante(l, fenetre) for l in donnees]) * echelle
    moyenne, ecart = lisse.mean(axis=0), lisse.std(axis=0)
    x = np.arange(len(moyenne)) + fenetre
    ax.plot(x, moyenne, lw=2, label=label, color=couleur)
    ax.fill_between(x, moyenne - ecart, moyenne + ecart, alpha=0.18, color=couleur)
    ax.grid(alpha=0.25)
    return moyenne


def trajectoire_gloutonne(nom_env, Q, max_pas=200):
    """Rejoue la politique gloutonne et renvoie la liste des états visités."""
    env, _, cfg = creer_env(nom_env)
    etat, _ = env.reset(seed=0)
    chemin, fini, pas = [int(etat)], False, 0
    while not fini and pas < min(max_pas, cfg["max_pas"]):
        etat, _, termine, tronque, _ = env.step(int(np.argmax(Q[etat])))
        chemin.append(int(etat))
        fini = termine or tronque
        pas += 1
    env.close()
    return chemin


def ligne_moyenne(chemin, colonnes=12):
    """
    Ligne moyenne parcourue par la politique. Sur la grille 4 x 12, la ligne 3 est
    celle de la falaise. Plus la moyenne est haute, plus l'agent prend de risques.
    Cela chiffre le « chemin risqué contre chemin sûr » au lieu de le décrire.
    """
    return float(np.mean([s // colonnes for s in chemin]))


def tracer_politique(Q, lignes, colonnes, titre, fichier, etats_speciaux=None):
    """Politique gloutonne en flèches, sur fond de la valeur d'état max_a Q(s,a)."""
    valeurs = Q.max(axis=1).reshape(lignes, colonnes)
    politique = np.argmax(Q, axis=1).reshape(lignes, colonnes)

    fig, ax = plt.subplots(figsize=(1.0 * colonnes, 1.1 * lignes))
    im = ax.imshow(valeurs, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8, label="max_a Q(s,a)")

    for i in range(lignes):
        for j in range(colonnes):
            s = i * colonnes + j
            texte = (etats_speciaux or {}).get(s, FLECHES[int(politique[i, j])])
            ax.text(j, i, texte, ha="center", va="center", color="w",
                    fontsize=12, weight="bold")

    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(titre, fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(fichier, dpi=140)
    plt.close(fig)
    print("   [figure]", fichier)


def sauver_tableau(entetes, lignes, nom_fichier, titre=""):
    """Le même tableau en CSV pour retraitement et en Markdown à coller au rapport."""
    chemin_csv = fichier_resultat(nom_fichier + ".csv")
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(entetes)
        w.writerows(lignes)

    chemin_md = fichier_resultat(nom_fichier + ".md")
    with open(chemin_md, "w", encoding="utf-8") as f:
        if titre:
            f.write("### " + titre + "\n\n")
        f.write("| " + " | ".join(entetes) + " |\n")
        f.write("|" + "|".join([" --- "] * len(entetes)) + "|\n")
        for ligne in lignes:
            f.write("| " + " | ".join(str(c) for c in ligne) + " |\n")
    print("   [tableau]", chemin_csv)
    print("   [tableau]", chemin_md)


# =============================================================================
#  5. EXPÉRIENCES DU RAPPORT
# =============================================================================
def experience_frozenlake(args):
    """
    Expérience 1 : FrozenLake glissant, 4x4 contre 8x8, trois algorithmes.
    Répond aux points 3 et 6 du sujet.
    """
    if args.rapide:
        args.episodes, args.episodes8, args.graines = 2000, 3000, 3

    graines = list(range(args.graines))
    toutes = {"4x4": ("frozenlake4x4", args.episodes),
              "8x8": ("frozenlake8x8", args.episodes8)}
    cartes = [toutes[c.strip()] for c in args.cartes.split(",") if c.strip() in toutes]
    lignes_tableau = []

    for nom_env, episodes in cartes:
        # alpha faible et gamma proche de 1 : sur glace glissante, une mise à jour
        # trop agressive fait osciller Q à cause de la stochasticité des transitions.
        hyper = dict(alpha=args.alpha, gamma=args.gamma, epsilon_debut=1.0,
                     epsilon_fin=0.01, epsilon_decr=decroissance_pour(episodes),
                     mode_epsilon="exponentiel")

        print("\n=== {} | {} épisodes | {} graines ===".format(nom_env, episodes, len(graines)))
        resultats = {}
        for algo in ALGOS:
            resultats[algo] = campagne(nom_env, algo, episodes, graines, hyper=hyper)
            r = resume_evaluations(resultats[algo])
            print("   -> évaluation gloutonne : {:.1f} % ± {:.1f} de succès sur 100 épisodes"
                  .format(r["succes_moy"], r["succes_std"]))
            lignes_tableau.append([
                nom_env, NOMS[algo], episodes, len(graines),
                "{:.1f} ± {:.1f}".format(r["succes_moy"], r["succes_std"]),
                "{:.3f} ± {:.3f}".format(r["retour_moy"], r["retour_std"]),
                "{:.1f}".format(r["longueur_moy"])])
            np.save(fichier_resultat("{}_{}.npy".format(nom_env, algo)),
                    resultats[algo]["tables"][0])

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
        for algo in ALGOS:
            courbe_multi_graines(a1, resultats[algo]["succes"], NOMS[algo],
                                 COULEURS[NOMS[algo]], fenetre=500, echelle=100.0)
            courbe_multi_graines(a2, resultats[algo]["recompenses"], NOMS[algo],
                                 COULEURS[NOMS[algo]], fenetre=500)
        a1.set_title("Taux de succès pendant l'entraînement", fontsize=11, weight="bold")
        a1.set_xlabel("Épisode"), a1.set_ylabel("% de succès"), a1.legend()
        a2.set_title("Récompense moyenne par épisode", fontsize=11, weight="bold")
        a2.set_xlabel("Épisode"), a2.set_ylabel("Récompense"), a2.legend()
        fig.suptitle("{} : moyenne sur {} graines, bande = ± 1 écart-type"
                     .format(nom_env, len(graines)), fontsize=12, weight="bold")
        fig.tight_layout()
        chemin = fichier_resultat("exp1_{}.png".format(nom_env))
        fig.savefig(chemin, dpi=140)
        plt.close(fig)
        print("   [figure]", chemin)

        cote = ENVIRONNEMENTS[nom_env]["grille"][0]
        for algo in ALGOS:
            tracer_politique(resultats[algo]["tables"][0], cote, cote,
                             "{} sur {}".format(NOMS[algo], nom_env),
                             fichier_resultat("exp1_politique_{}_{}.png".format(nom_env, algo)))

    suffixe = "" if len(cartes) == 2 else "_" + args.cartes.replace(",", "_")
    sauver_tableau(["Environnement", "Algorithme", "Épisodes", "Graines",
                    "Succès glouton (%)", "Retour moyen", "Longueur moyenne"],
                   lignes_tableau, "exp1_frozenlake_resume" + suffixe,
                   titre="FrozenLake : évaluation gloutonne finale (100 épisodes par graine)")


def experience_epsilon(args):
    """
    Expérience 3 : effet du calendrier d'exploration (point 5 du sujet).

    La théorie prévoit que SARSA rejoint la politique optimale quand epsilon tend
    vers 0, puisque sa cible Q(S',A') tend alors vers le maximum. Les mesures ne le
    confirment qu'à moitié, ce qui est le résultat le plus intéressant du projet.
    """
    if args.rapide:
        args.episodes, args.graines = 400, 5

    if args.balayage:
        return balayage_epsilon(args)

    graines = list(range(args.graines))
    calendriers = {
        "fixe": dict(alpha=args.alpha, gamma=args.gamma, mode_epsilon="fixe",
                     epsilon_debut=args.epsilon, epsilon_fin=args.epsilon),
        "decroissant": dict(alpha=args.alpha, gamma=args.gamma, mode_epsilon="exponentiel",
                            epsilon_debut=1.0, epsilon_fin=0.01,
                            epsilon_decr=decroissance_pour(args.episodes)),
    }
    styles = {"fixe": "-", "decroissant": "--"}

    print("\n=== {} | epsilon fixe {} contre décroissant | {} épisodes, {} graines ==="
          .format(args.env, args.epsilon, args.episodes, len(graines)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    lignes_tableau, chemins, titres = [], [], []

    for algo in ["qlearning", "sarsa"]:
        for nom_cal, hyper in calendriers.items():
            res = campagne(args.env, algo, args.episodes, graines, hyper=hyper, verbeux=False)
            r = resume_evaluations(res)
            etiquette = "{} / epsilon {}".format(NOMS[algo], nom_cal)
            print("   {:<32} glouton {:>7.2f} ± {:.2f} | {:>5.1f} % de réussite"
                  .format(etiquette, r["retour_moy"], r["retour_std"], r["succes_moy"]))

            courbe_multi_graines(ax, res["recompenses"], etiquette, COULEURS[NOMS[algo]],
                                 fenetre=max(20, args.episodes // 50))
            ax.lines[-1].set_linestyle(styles[nom_cal])

            lignes_tableau.append([
                NOMS[algo], nom_cal,
                "{:.2f}".format(res["recompenses"][:, int(0.75 * args.episodes):].mean()),
                "{:.2f} ± {:.2f}".format(r["retour_moy"], r["retour_std"]),
                "{:.1f}".format(r["succes_moy"]), "{:.1f}".format(r["longueur_moy"])])

            if args.env == "cliffwalking":
                chemins.append(trajectoire_gloutonne("cliffwalking", res["tables"][0]))
                titres.append(etiquette)

    ax.set_xlabel("Épisode"), ax.set_ylabel("Somme des récompenses")
    ax.set_title("{} : effet du calendrier d'exploration\nmoyenne sur {} graines"
                 .format(args.env, len(graines)), fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    if args.env == "cliffwalking":
        ax.set_ylim(-200, 0)
    fig.tight_layout()
    chemin = fichier_resultat("exp3_epsilon_{}.png".format(args.env))
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    print("   [figure]", chemin)

    sauver_tableau(["Algorithme", "Calendrier de epsilon", "Récompense d'entraînement (fin)",
                    "Retour glouton", "Réussite (%)", "Longueur"],
                   lignes_tableau, "exp3_epsilon_{}_resume".format(args.env),
                   titre="{} : epsilon fixe contre epsilon décroissant".format(args.env))


def balayage_epsilon(args):
    """
    Version chiffrée de la question « que deviennent les politiques si epsilon
    tend vers 0 ». Deux indicateurs par valeur : le retour de la politique
    gloutonne et la ligne moyenne du chemin, qui mesure la prise de risque.
    """
    valeurs = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    graines = list(range(max(5, args.graines // 2)))
    episodes = max(args.episodes, 5000)
    print("\n=== CliffWalking | balayage de epsilon fixe | {} épisodes, {} graines ==="
          .format(episodes, len(graines)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lignes = []
    for algo in ["qlearning", "sarsa"]:
        retours, lignes_moy, ecarts = [], [], []
        for eps in valeurs:
            hyper = dict(alpha=args.alpha, gamma=args.gamma, mode_epsilon="fixe",
                         epsilon_debut=eps, epsilon_fin=eps)
            res = campagne("cliffwalking", algo, episodes, graines, hyper=hyper, verbeux=False)
            r = resume_evaluations(res)
            lm = np.mean([ligne_moyenne(trajectoire_gloutonne("cliffwalking", Q))
                          for Q in res["tables"]])
            retours.append(r["retour_moy"]), ecarts.append(r["retour_std"])
            lignes_moy.append(lm)
            print("   {:<15} epsilon {:.2f} : retour {:>7.2f} ± {:.2f}, ligne moyenne {:.2f}"
                  .format(NOMS[algo], eps, r["retour_moy"], r["retour_std"], lm))
            lignes.append([NOMS[algo], eps,
                           "{:.2f} ± {:.2f}".format(r["retour_moy"], r["retour_std"]),
                           "{:.2f}".format(lm), "{:.1f}".format(r["longueur_moy"])])

        a1.errorbar(valeurs, retours, yerr=ecarts, marker="o", capsize=3,
                    color=COULEURS[NOMS[algo]], label=NOMS[algo])
        # Une graine qui boucle donne -200 et écrase l'échelle : on borne l'axe sur
        # la zone utile, la barre d'erreur signale l'accident.
        a1.set_ylim(-45, -5)
        a2.plot(valeurs, lignes_moy, marker="o", color=COULEURS[NOMS[algo]], label=NOMS[algo])

    for ax, titre, ylab in ((a1, "Retour de la politique gloutonne", "Retour"),
                            (a2, "Position moyenne du chemin", "Ligne moyenne (3 = falaise)")):
        ax.set_xscale("log")
        ax.set_xlabel("epsilon fixe (échelle log)"), ax.set_ylabel(ylab)
        ax.set_title(titre, fontsize=11, weight="bold")
        ax.grid(alpha=0.25), ax.legend()
    fig.suptitle("CliffWalking : la politique apprise en fonction de l'exploration "
                 "({} épisodes, {} graines)".format(episodes, len(graines)),
                 fontsize=12, weight="bold")
    fig.tight_layout()
    chemin = fichier_resultat("exp3_balayage_epsilon.png")
    fig.savefig(chemin, dpi=140)
    plt.close(fig)
    print("   [figure]", chemin)

    sauver_tableau(["Algorithme", "epsilon", "Retour glouton", "Ligne moyenne", "Longueur"],
                   lignes, "exp3_balayage_epsilon_resume",
                   titre="CliffWalking : effet de la valeur de epsilon fixe")


# =============================================================================
#  6. DÉMONSTRATION VISUELLE
# =============================================================================
# Réglages de la démonstration. La taille suit la forme de la grille : CliffWalking
# est un rectangle très large, FrozenLake un carré. Les checkpoints sont resserrés au
# début pour voir l'agent passer de mauvais à bon.
DEMO = {
    "cliffwalking": dict(taille=(7.5, 3.0), vertical=True,
                         episodes=1000, alpha=0.25, gamma=1.0,
                         epsilon=0.1, mode="fixe", n_avant=3, n_test=4, dt=0.12,
                         checkpoints=[1, 10, 25, 50, 80, 120, 200, 300, 450, 650, 1000]),
    "frozenlake4x4": dict(taille=(5.0, 5.4), episodes=8000, alpha=0.1, gamma=0.99,
                          epsilon=None, mode="exponentiel", n_avant=3, n_test=6, dt=0.25,
                          checkpoints=[1, 200, 600, 1200, 2000, 3000, 4500, 6000, 8000]),
    "frozenlake8x8": dict(taille=(5.0, 5.4), episodes=20000, alpha=0.1, gamma=0.99,
                          epsilon=None, mode="exponentiel", n_avant=2, n_test=4, dt=0.15,
                          checkpoints=[1, 2000, 8000, 20000]),
}


def choisir_backend():
    """
    Cherche un backend matplotlib à fenêtre en testant réellement l'ouverture d'une
    figure. pygame refuse souvent de s'afficher depuis Spyder, d'où ce détour par
    matplotlib. Sans backend interactif, on retombe sur Agg et l'animation est
    remplacée par l'export vidéo.

    Dans Spyder : Outils > Préférences > Console IPython > Graphiques > Backend >
    Automatique, puis redémarrer le noyau.
    """
    for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            fig = plt.figure()
            plt.close(fig)
            plt.ion()
            return True
        except Exception:
            continue
    plt.switch_backend("Agg")
    return False


class Ecran:
    """
    Affiche une à trois vues et peut capturer les images.

    Les vues sont empilées verticalement quand la grille est large et basse, comme
    CliffWalking : trois de ces rectangles côte à côte donneraient une image huit fois
    plus large que haute, illisible une fois réduite pour la vidéo.
    """

    def __init__(self, titre_fenetre, n_vues=1, taille=(6, 6), vertical=False):
        if vertical:
            self.fig, axes = plt.subplots(n_vues, 1, figsize=(taille[0], taille[1] * n_vues))
        else:
            self.fig, axes = plt.subplots(1, n_vues, figsize=(taille[0] * n_vues, taille[1]))
        self.axes = [axes] if n_vues == 1 else list(axes)
        if vertical and n_vues > 1:
            # Les titres tiennent sur deux lignes : sans cet écart ils viennent
            # mordre sur l'image de la vue précédente.
            self.fig.subplots_adjust(hspace=0.55, top=0.90, bottom=0.03)
        try:
            self.fig.canvas.manager.set_window_title(titre_fenetre)
        except Exception:
            pass
        for ax in self.axes:
            ax.axis("off")
        self.images = [None] * n_vues
        self.bandeaux = [ax.set_title("", fontsize=12, weight="bold") for ax in self.axes]
        # Images du film, gardées compressées : une capture brute pèse 2 Mo, et une
        # démonstration en produit près d'un millier. Les conserver telles quelles
        # saturait la mémoire et faisait tuer le processus avant l'écriture du fichier.
        self.captures = []

    def montrer(self, cadres, textes, dt, enregistrer=False):
        for i, (cadre, texte) in enumerate(zip(cadres, textes)):
            if self.images[i] is None:
                self.images[i] = self.axes[i].imshow(cadre)
            else:
                self.images[i].set_data(cadre)
            self.bandeaux[i].set_text(texte)
        plt.pause(max(0.001, dt))  # dessine ET rafraîchit la fenêtre
        if enregistrer:
            self.fig.canvas.draw()
            image = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
            self.captures.append(comprimer_image(image))

    def fermer(self):
        plt.close(self.fig)


LARGEUR_VIDEO = 1000  # largeur maximale des images du film, en pixels


def comprimer_image(image, largeur_max=LARGEUR_VIDEO):
    """
    Réduit une capture et la renvoie encodée en PNG. Le stockage passe d'environ
    2 Mo à 100 Ko par image, sans perte de qualité, ce qui permet de garder un film
    entier en mémoire avant d'en fixer la cadence.

    Les dimensions sont ramenées à des nombres pairs : la plupart des codecs vidéo
    refusent les tailles impaires.
    """
    from io import BytesIO
    from PIL import Image

    img = Image.fromarray(np.asarray(image))
    if img.width > largeur_max:
        img = img.resize((largeur_max, max(2, round(img.height * largeur_max / img.width))),
                         Image.LANCZOS)
    if img.width % 2 or img.height % 2:
        img = img.resize((img.width - img.width % 2, img.height - img.height % 2))
    tampon = BytesIO()
    img.save(tampon, format="PNG")
    return tampon.getvalue()


def decomprimer_image(donnees):
    """Inverse de comprimer_image : renvoie un tableau numpy."""
    from io import BytesIO
    from PIL import Image
    return np.asarray(Image.open(BytesIO(donnees)).convert("RGB"))


def jouer_et_afficher(envs, tables, ecran, textes, dt, cfgs, au_hasard=False,
                      rng=None, max_pas_vue=80, enregistrer=False):
    """
    Joue un épisode sur chaque environnement de la liste, image par image et en
    parallèle, pour que la comparaison Q-learning / SARSA reste lisible à l'écran.
    """
    etats = [env.reset(seed=int(rng.integers(1 << 31)) if rng is not None else None)[0]
             for env in envs]
    finis, pas = [False] * len(envs), 0
    if ecran is not None:
        ecran.montrer([e.render() for e in envs], textes, dt, enregistrer)

    plafond = min(min(c["max_pas"] for c in cfgs), max_pas_vue)
    while not all(finis) and pas < plafond:
        for i, env in enumerate(envs):
            if finis[i]:
                continue
            a = env.action_space.sample() if au_hasard else int(np.argmax(tables[i][etats[i]]))
            etats[i], _, termine, tronque, _ = env.step(a)
            finis[i] = termine or tronque
        pas += 1
        if ecran is not None:
            ecran.montrer([e.render() for e in envs], textes, dt, enregistrer)
    return pas


def demonstration(args):
    """
    Trois temps, comme demandé pour la vidéo du livrable :
      1) l'agent agit au hasard et échoue ;
      2) aperçus à des épisodes de plus en plus tardifs ;
      3) la politique apprise, jouée en glouton pur.
    """
    backend_ok = choisir_backend()
    cfg = dict(DEMO[args.env])
    if getattr(args, "dt", None):
        cfg["dt"] = args.dt
    episodes = args.episodes or cfg["episodes"]
    nb_apercus = getattr(args, "apercus", None)
    if nb_apercus:
        # Aperçus espacés géométriquement : serrés au début, où la politique change
        # vite, puis de plus en plus écartés quand elle ne bouge presque plus.
        checkpoints = {int(round(episodes ** ((i + 1) / nb_apercus)))
                       for i in range(nb_apercus)}
        checkpoints = {max(1, min(episodes, c)) for c in checkpoints} | {1, episodes}
    else:
        checkpoints = set(c for c in cfg["checkpoints"] if c <= episodes) | {episodes}
    noms_agents = tuple(ALGOS) if args.comparer else (args.agent,)

    # epsilon fixe sur CliffWalking, c'est ce qui sépare les deux politiques ;
    # décroissant sur FrozenLake, où il faut beaucoup explorer au début.
    if cfg["mode"] == "fixe":
        hyper = dict(alpha=cfg["alpha"], gamma=cfg["gamma"], mode_epsilon="fixe",
                     epsilon_debut=cfg["epsilon"], epsilon_fin=cfg["epsilon"])
    else:
        hyper = dict(alpha=cfg["alpha"], gamma=cfg["gamma"], mode_epsilon="exponentiel",
                     epsilon_debut=1.0, epsilon_fin=0.01,
                     epsilon_decr=decroissance_pour(episodes))

    print("Environnement :", args.env, "|", ENVIRONNEMENTS[args.env]["titre"])
    print("Agents :", ", ".join(NOMS[n] for n in noms_agents))
    if not backend_ok and not args.video:
        print("[info] aucun backend interactif. Réglez Spyder sur Préférences >")
        print("       Console IPython > Graphiques > Automatique, ou utilisez --video.")

    # Deux environnements par agent : un rapide sans image pour apprendre, un avec
    # rendu pour les aperçus. Le rendu coûte cher, on le réserve aux épisodes montrés.
    envs_train, envs_vue, agents, cfgs = [], [], [], []
    for nom_agent in noms_agents:
        e1, _, c = creer_env(args.env)
        e2, _, _ = creer_env(args.env, render_mode="rgb_array")
        e1.action_space.seed(0)
        envs_train.append(e1), envs_vue.append(e2), cfgs.append(c)
        agents.append(AGENTS[nom_agent](e1.observation_space.n, e1.action_space.n,
                                        graine=0, **hyper))

    ecran = Ecran("RL - " + args.env, len(noms_agents), cfg["taille"],
                  vertical=cfg.get("vertical", False))
    rng = np.random.default_rng(0)
    enregistrer = args.video is not None
    lisibles = [NOMS[n] for n in noms_agents]

    print("\n=== 1) AVANT entraînement : l'agent agit AU HASARD ===")
    for i in range(cfg["n_avant"]):
        textes = ["{}\nAVANT entraînement, actions au hasard ({}/{})".format(n, i + 1, cfg["n_avant"])
                  for n in lisibles]
        pas = jouer_et_afficher(envs_vue, [a.Q for a in agents], ecran, textes, cfg["dt"],
                                cfgs, au_hasard=True, rng=rng, max_pas_vue=45,
                                enregistrer=enregistrer)
        print("   essai {}/{} : {} pas".format(i + 1, cfg["n_avant"], pas))

    print("\n=== 2) ENTRAÎNEMENT ({} épisodes) ===".format(episodes))
    historiques = [np.zeros(episodes) for _ in agents]
    for ep in range(episodes):
        for idx, (env, agent) in enumerate(zip(envs_train, agents)):
            etat, _ = env.reset(seed=int(rng.integers(1 << 31)) if ep == 0 else None)
            action = agent.choisir(etat)
            fini, pas, total = False, 0, 0.0
            while not fini and pas < cfgs[idx]["max_pas"]:
                etat_suivant, r, termine, tronque, _ = env.step(action)
                fini = termine or tronque
                if agent.besoin_action_suivante:
                    action_suivante = agent.choisir(etat_suivant)
                    agent.maj(etat, action, r, etat_suivant, action_suivante, termine)
                    action = action_suivante
                else:
                    agent.maj(etat, action, r, etat_suivant, None, termine)
                    action = agent.choisir(etat_suivant)
                etat, total, pas = etat_suivant, total + r, pas + 1
            agent.maj_epsilon(ep, episodes)
            historiques[idx][ep] = total

        if (ep + 1) in checkpoints:
            moyennes = [h[max(0, ep - 99):ep + 1].mean() for h in historiques]
            textes = ["{}\nÉpisode {}/{} (récompense récente {:.1f})".format(n, ep + 1, episodes, m)
                      for n, m in zip(lisibles, moyennes)]
            print("   " + " | ".join("{} : {:.1f}".format(n, m)
                                     for n, m in zip(lisibles, moyennes)))
            jouer_et_afficher(envs_vue, [a.Q for a in agents], ecran, textes, cfg["dt"],
                              cfgs, max_pas_vue=80, enregistrer=enregistrer)

    print("\n=== 3) TEST : la politique apprise, 100 % gloutonne ===")
    for i in range(cfg["n_test"]):
        textes = ["{}\nTEST, politique apprise ({}/{})".format(n, i + 1, cfg["n_test"])
                  for n in lisibles]
        pas = jouer_et_afficher(envs_vue, [a.Q for a in agents], ecran, textes, cfg["dt"],
                                cfgs, rng=rng, max_pas_vue=cfgs[0]["max_pas"],
                                enregistrer=enregistrer)
        print("   test {}/{} : {} pas".format(i + 1, cfg["n_test"], pas))

    for e in envs_train + envs_vue:
        e.close()

    # Courbe d'apprentissage. Le sujet demande qu'elle apparaisse dans la vidéo, elle
    # est donc dessinée au format exact des images capturées puis maintenue à l'écran
    # quelques secondes à la fin du film.
    # Figures de la démonstration : politique de chaque agent, trajectoires comparées
    # sur CliffWalking, Q-tables réutilisables avec la commande « rejouer ».
    figures_demo(args.env, noms_agents, [a.Q for a in agents])

    courbe = tracer_courbe_demo(historiques, lisibles, args.env,
                                taille=ecran.fig.get_size_inches())
    if args.video and ecran.captures:
        maintien = int(6.0 / max(0.04, cfg["dt"]))  # environ six secondes
        images = ecran.captures + [comprimer_image(courbe)] * maintien
        ecrire_video(images, args.video, cfg["dt"], duree_cible=args.duree)
    ecran.fermer()
    plt.show(block=True)


def figures_demo(nom_env, noms_agents, tables):
    """
    Sauvegarde ce qui reste utile après l'animation : les Q-tables apprises, la
    politique de chaque agent, et sur CliffWalking les trajectoires comparées, qui
    sont la figure la plus parlante du projet.
    """
    lignes, colonnes = ENVIRONNEMENTS[nom_env]["grille"]

    chemins, titres = [], []
    for nom_agent, Q in zip(noms_agents, tables):
        np.save(fichier_resultat("demo_{}_{}.npy".format(nom_env, nom_agent)), Q)
        tracer_politique(Q, lignes, colonnes,
                         "{} sur {}".format(NOMS[nom_agent], nom_env),
                         fichier_resultat("demo_politique_{}.png".format(nom_env, nom_agent)))


def tracer_courbe_demo(historiques, noms, nom_env, taille=(8, 4.4)):
    """
    Enregistre la courbe de récompense et renvoie son image, au même format que les
    captures de l'écran de démonstration pour pouvoir l'ajouter à la vidéo.
    """
    fig, ax = plt.subplots(figsize=tuple(taille))
    for h, n in zip(historiques, noms):
        ax.plot(moyenne_glissante(h, 50), lw=2, label=n, color=COULEURS.get(n))
    ax.set_xlabel("Épisode"), ax.set_ylabel("Récompense (moyenne glissante)")
    ax.set_title("Apprentissage sur " + nom_env, fontsize=12, weight="bold")
    ax.grid(alpha=0.25), ax.legend()
    fig.tight_layout()
    chemin = fichier_resultat("demo_courbes_{}.png".format(nom_env))
    fig.savefig(chemin, dpi=130)
    print("   [figure]", chemin)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def ecrire_video(images, chemin, dt, duree_cible=None):
    """
    Écrit la vidéo de démonstration à partir des images compressées par
    comprimer_image. MP4 par défaut, GIF si le nom de fichier se termine par .gif.

    Les images sont décodées une par une et transmises au fur et à mesure à
    l'encodeur : tout charger d'un coup demanderait plus d'un gigaoctet de mémoire
    pour une démonstration complète.

    duree_cible fixe la durée voulue en secondes. La cadence en découle, bornée
    entre 4 et 20 images par seconde. Le livrable demande 2 à 3 minutes ; si la
    capture est trop courte pour y arriver, le script le signale.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("Écriture vidéo indisponible. Installez les dépendances :\n"
                         "    pip install -r requirements.txt")

    fps = 1.0 / max(0.04, dt)
    if duree_cible:
        fps = min(20.0, max(4.0, len(images) / duree_cible))
    duree = len(images) / fps

    if chemin.lower().endswith(".gif"):
        ecrivain = imageio.get_writer(chemin, mode="I", duration=1.0 / fps, loop=0)
    else:
        ecrivain = imageio.get_writer(chemin, fps=fps, macro_block_size=None)
    with ecrivain:
        for donnees in images:
            ecrivain.append_data(decomprimer_image(donnees))

    print("[vidéo] {} : {} images, {:.0f} images/s, {:.0f} s"
          .format(chemin, len(images), fps, duree))
    if duree_cible and duree < 0.9 * duree_cible:
        print("        durée inférieure à la cible. Augmentez --episodes, ou le nombre")
        print("        d'aperçus avec --apercus, pour capturer plus d'images.")


# =============================================================================
#  7. REJEU D'UNE POLITIQUE SAUVEGARDÉE
# =============================================================================
def rejouer(args):
    """
    Rejoue une Q-table sans aucun réentraînement, comme demandé par le sujet.
    L'environnement est déduit du nom du fichier (<env>_<algo>.npy) ; --env force
    un autre choix.
    """
    Q = np.load(args.model)
    nom_env = args.env
    if nom_env is None:
        base = os.path.basename(args.model).lower()
        nom_env = next((n for n in ENVIRONNEMENTS if base.startswith(n)), None)
    if nom_env is None:
        raise SystemExit("Impossible de deviner l'environnement. Ajoutez --env.")

    # Sans GIF on utilise le rendu natif de Gymnasium (fenêtre pygame). Avec GIF on
    # récupère les images en tableaux numpy et Pillow assemble le fichier.
    mode = "rgb_array" if args.gif else "human"
    env, eid, cfg = creer_env(nom_env, render_mode=mode)
    print("Environnement :", eid, "| table :", args.model, "| forme :", Q.shape)

    images, rng = [], np.random.default_rng(0)
    for ep in range(args.episodes):
        etat, _ = env.reset(seed=int(rng.integers(1 << 31)))
        fini, pas, total, reussi = False, 0, 0.0, False
        while not fini and pas < cfg["max_pas"]:
            etat, r, termine, tronque, _ = env.step(int(np.argmax(Q[etat])))
            fini = termine or tronque
            total, pas = total + r, pas + 1
            if cfg["succes"](r, termine):
                reussi = True
            if args.gif:
                images.append(env.render())
            else:
                import time
                time.sleep(args.pause)
        print("   épisode {}/{} : {} pas, retour {:.1f} -> {}"
              .format(ep + 1, args.episodes, pas, total, "réussi" if reussi else "échoué"))
    env.close()

    if args.gif and images:
        from PIL import Image
        cadres = [Image.fromarray(im) for im in images]
        cadres[0].save(args.gif, save_all=True, append_images=cadres[1:],
                       duration=int(1000 * args.pause), loop=0)
        print("GIF écrit :", args.gif, "({} images)".format(len(cadres)))


# =============================================================================
#  8. LIGNE DE COMMANDE
# =============================================================================
def construire_parseur():
    p = argparse.ArgumentParser(
        description="Projet 4 : Q-learning contre SARSA sur FrozenLake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemple : python main.py frozenLake --rapide")
    sous = p.add_subparsers(dest="commande", required=True)

    f = sous.add_parser("frozenlake", help="expérience 1 : cartes 4x4 et 8x8")
    f.add_argument("--episodes", type=int, default=20000, help="épisodes pour la carte 4x4")
    f.add_argument("--episodes8", type=int, default=40000, help="épisodes pour la carte 8x8")
    f.add_argument("--graines", type=int, default=10)
    f.add_argument("--alpha", type=float, default=0.10)
    f.add_argument("--gamma", type=float, default=0.99)
    f.add_argument("--cartes", default="4x4,8x8", help="par exemple --cartes 4x4")
    f.add_argument("--rapide", action="store_true")
    f.set_defaults(fonction=experience_frozenlake)

    e = sous.add_parser("epsilon", help="expérience 3 : calendrier d'exploration")
    e.add_argument("--env", default="cliffwalking", choices=list(ENVIRONNEMENTS))
    e.add_argument("--episodes", type=int, default=2000)
    e.add_argument("--graines", type=int, default=20)
    e.add_argument("--alpha", type=float, default=0.25)
    e.add_argument("--gamma", type=float, default=1.0)
    e.add_argument("--epsilon", type=float, default=0.1, help="valeur du epsilon fixe")
    e.add_argument("--balayage", action="store_true",
                   help="faire varier epsilon fixe au lieu de comparer les calendriers")
    e.add_argument("--rapide", action="store_true")
    e.set_defaults(fonction=experience_epsilon)

    d = sous.add_parser("demo", help="animation de l'agent avant, pendant et après")
    d.add_argument("--env", default="frozenLake", choices=list(DEMO))
    d.add_argument("--agent", default="qlearning", choices=list(AGENTS))
    d.add_argument("--comparer", action="store_true",
                   help="les trois algorithmes côte à côte")
    d.add_argument("--episodes", type=int, default=None)
    d.add_argument("--apercus", type=int, default=None,
                   help="nombre d'aperçus pendant l'entraînement (défaut : liste "
                        "prédéfinie par environnement)")
    d.add_argument("--video", default=None, help="fichier .mp4 ou .gif à écrire")
    d.add_argument("--dt", type=float, default=None,
                   help="secondes entre deux images à l'écran. Baisser cette valeur "
                        "accélère la fabrication d'une vidéo, sans changer sa cadence "
                        "finale, qui est fixée par --duree")
    d.add_argument("--duree", type=float, default=150.0,
                   help="durée visée de la vidéo en secondes (livrable : 120 à 180)")
    d.set_defaults(fonction=demonstration)

    r = sous.add_parser("rejouer", help="rejouer une Q-table sauvegardée")
    r.add_argument("--model", required=True, help="fichier .npy contenant la Q-table")
    r.add_argument("--env", choices=list(ENVIRONNEMENTS), default=None)
    r.add_argument("--episodes", type=int, default=3)
    r.add_argument("--gif", default=None, help="enregistrer le rejeu dans un GIF")
    r.add_argument("--pause", type=float, default=0.25)
    r.set_defaults(fonction=rejouer)

    q = sous.add_parser("parcours", help="démonstration des deux environnements à la suite")
    q.add_argument("--envs", nargs="+", default=None, choices=list(DEMO),
                   help="environnements à enchaîner, dans l'ordre voulu")
    q.add_argument("--episodes", type=int, default=None)
    q.add_argument("--apercus", type=int, default=None,
                   help="nombre d'aperçus pendant l'entraînement (défaut : 14)")
    q.add_argument("--sans-video", dest="sans_video", action="store_true",
                   help="ne pas enregistrer les vidéos MP4")
    q.add_argument("--dt", type=float, default=None,
                   help="secondes entre deux images à l'écran")
    q.set_defaults(fonction=parcours)

    t = sous.add_parser("tout", help="enchaîne les trois expériences du rapport")
    t.add_argument("--rapide", action="store_true")
    t.set_defaults(fonction=tout_lancer)

    return p


def tout_lancer(args):
    """Enchaîne les trois expériences avec leurs réglages par défaut."""
    parseur = construire_parseur()
    for commande in ("frozenlake", "epsilon"):
        arguments = [commande] + (["--rapide"] if args.rapide else [])
        sous_args = parseur.parse_args(arguments)
        sous_args.fonction(sous_args)
    # Le balayage n'a pas de réglage par défaut commun avec le reste : on le relance
    # explicitement, sinon la figure correspondante manquerait au rapport.
    sous_args = parseur.parse_args(["epsilon", "--balayage"] +
                                   (["--rapide"] if args.rapide else []))
    sous_args.fonction(sous_args)


def parcours(args=None):
    """
    Démonstration complète, sans rien avoir à taper : chaque environnement de
    PARCOURS_PAR_DEFAUT défile à son tour, Q-learning et SARSA côte à côte.

    Entre deux environnements, la courbe d'apprentissage reste affichée : fermez la
    fenêtre pour enchaîner. Sous un backend sans fenêtre, l'enchaînement est immédiat
    et seules les figures écrites dans resultats/ témoignent du passage.
    """
    parseur = construire_parseur()
    liste = getattr(args, "envs", None) or PARCOURS_PAR_DEFAUT
    for numero, nom_env in enumerate(liste, start=1):
        print("\n" + "=" * 78)
        print("  DÉMONSTRATION {}/{} : {}".format(numero, len(liste),
                                                  ENVIRONNEMENTS[nom_env]["titre"]))
        print("=" * 78)
        arguments = ["demo", "--env", nom_env, "--comparer"]
        if not getattr(args, "sans_video", False):
            arguments += ["--video", fichier_resultat("demo_{}.mp4".format(nom_env))]
        if getattr(args, "episodes", None):
            arguments += ["--episodes", str(args.episodes)]
        arguments += ["--apercus", str(getattr(args, "apercus", None) or 14)]
        if getattr(args, "dt", None):
            arguments += ["--dt", str(args.dt)]
        sous_args = parseur.parse_args(arguments)
        if numero < len(liste):
            print("(fermez la fenêtre de la courbe pour passer à l'environnement suivant)")
        demonstration(sous_args)

    print("\n" + "=" * 78)
    print("  Terminé. Figures, vidéos et Q-tables écrites dans resultats/")
    print("=" * 78)
    print("Les courbes multi-graines du rapport demandent les expériences chiffrées,")
    print("plus longues, à lancer depuis un terminal :")
    print("   python main.py frozenlake")
    print("   python main.py epsilon")
    print("   python main.py tout")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # Aucun argument : on déroule la démonstration complète, pour qu'un simple F5
        # dans Spyder montre quelque chose d'utile sans rien avoir à taper.
        argv = ["parcours"]
        print("Démonstration complète : " + " puis ".join(PARCOURS_PAR_DEFAUT) + ".")
        print("Pour les expériences chiffrées du rapport, lancez depuis un terminal :")
        print("   python main.py frozenlake | epsilon | tout\n")
    args = construire_parseur().parse_args(argv)
    # Les expériences n'ouvrent aucune fenêtre : backend sans affichage, ce qui
    # évite les plantages sur un serveur. La démonstration, elle, choisit son
    # backend elle-même au moment où elle démarre.
    if args.commande not in ("demo", "parcours"):
        matplotlib.use("Agg")
    args.fonction(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

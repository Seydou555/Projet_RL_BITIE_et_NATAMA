# -*- coding: utf-8 -*-
"""
==============================================================================
  Q-LEARNING VISUEL (version qui S'AFFICHE DANS SPYDER)
  Taxi / FrozenLake / CliffWalking
==============================================================================

  Ici on N'UTILISE PAS pygame (qui refuse de s'afficher depuis Spyder).
  On dessine l'agent avec MATPLOTLIB : une fenetre d'image s'ouvre et se met
  a jour, ce qui fonctionne aussi bien dans Spyder (F5 / %runfile) que dans un
  terminal.

  Tu verras, avec un TITRE ecrit sur l'image a chaque instant :
     1) AVANT      : l'agent agit au hasard (il echoue).
     2) ENTRAINEMENT : apercus a l'episode 1, puis de plus en plus tard
                       -> tu vois l'agent passer de NUL a BON.
     3) TEST       : la politique apprise, il reussit tout de suite.

  --------------------------------------------------------------------------
  DANS SPYDER : une seule chose a regler UNE fois pour que la fenetre s'anime :
     Outils -> Preferences -> Console IPython -> Graphiques -> Backend
        -> choisis "Automatique"  (et PAS "En ligne / Inline")
     Puis redemarre le noyau (Console -> Redemarrer le noyau).
  Le script essaie deja de forcer ce reglage tout seul, mais si l'image ne
  bouge pas, c'est ce parametre qu'il faut changer.
  --------------------------------------------------------------------------

  Choix de l'environnement : change la ligne  ENV_CHOISI  ci-dessous.
  Dependances :  pip install numpy matplotlib gymnasium
==============================================================================
"""


import matplotlib
import matplotlib.pyplot as plt

_backend_ok = False
for _bk in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        plt.switch_backend(_bk)
        _f = plt.figure(); plt.close(_f)     # test reel : la fenetre peut-elle s'ouvrir ?
        _backend_ok = True
        break
    except Exception:
        continue
if not _backend_ok:
    plt.switch_backend("Agg")                # aucun backend graphique -> pas d'animation

import numpy as np
import gymnasium as gym

if _backend_ok:
    plt.ion()   # mode interactif : la fenetre se met a jour en direct


# ==============================================================================
#  CHOIX DE L'ENVIRONNEMENT  (change juste cette ligne)
# ==============================================================================
ENV_CHOISI = "CliffWalking"        #"FrozenLake"  |  "CliffWalking"


# ==============================================================================
#  Reglages par environnement
#   - checkpoints : les episodes ou on montre un apercu (tot -> pour voir progresser)
# ==============================================================================
CONFIGS = {
    "FrozenLake": dict(
        candidats=["FrozenLake-v1"], kwargs=dict(is_slippery=False),
        episodes=3000, alpha=0.80, gamma=0.95,
        eps_debut=1.0, eps_fin=0.02, eps_decr=0.9990, max_pas=100,
        checkpoints=[1, 300, 800, 1500, 3000],
        n_avant=2, n_test=4, dt=0.25,
        succes=lambda r, term: bool(r == 1),
        titre="FrozenLake : atteindre le cadeau sans tomber",
    ),
    "CliffWalking": dict(
        candidats=["CliffWalking-v1", "CliffWalking-v0"], kwargs={},
        episodes=1200, alpha=0.50, gamma=0.99,
        eps_debut=1.0, eps_fin=0.02, eps_decr=0.990, max_pas=200,
        checkpoints=[1, 30, 100, 300, 700, 1200],
        n_avant=2, n_test=4, dt=0.12,
        succes=lambda r, term: bool(term),
        titre="CliffWalking : rejoindre l'arrivee en longeant la falaise",
    ),
}


# ==============================================================================
#  Petite classe pour AFFICHER les images de l'agent avec matplotlib
# ==============================================================================
class Ecran:
    def __init__(self, titre_fenetre):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.canvas.manager.set_window_title(titre_fenetre) if hasattr(
            self.fig.canvas, "manager") else None
        self.ax.axis("off")
        self.img = None
        self.bandeau = self.ax.set_title("", fontsize=13, weight="bold")

    def montrer(self, frame, texte, dt):
        if self.img is None:
            self.img = self.ax.imshow(frame)
        else:
            self.img.set_data(frame)
        self.bandeau.set_text(texte)
        plt.pause(max(0.001, dt))   # dessine ET rafraichit la fenetre

    def fermer(self):
        plt.close(self.fig)


# ==============================================================================
#  Outils environnement
# ==============================================================================
def trouver_env(cfg):
    for eid in cfg["candidats"]:
        try:
            gym.make(eid, **cfg["kwargs"]).close()
            return eid
        except Exception:
            continue
    raise SystemExit("Aucune version trouvee. Fais :  pip install -U gymnasium")


def jouer_et_afficher(env, Q, cfg, ecran, texte, dt, au_hasard=False,
                      rng=None, max_pas_vue=80):
    """Joue UN episode et l'affiche image par image. Renvoie (pas, reussi)."""
    if au_hasard:
        etat, _ = env.reset(seed=int(rng.integers(1 << 31)))
    else:
        etat, _ = env.reset()
    fini, pas, reussi = False, 0, False
    plafond = min(cfg["max_pas"], max_pas_vue)
    if ecran is not None:
        ecran.montrer(env.render(), texte, dt)
    while not fini and pas < plafond:
        if au_hasard:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[etat]))
        etat, r, term, trunc, _ = env.step(a)
        fini = term or trunc
        pas += 1
        if cfg["succes"](r, term):
            reussi = True
        if ecran is not None:
            ecran.montrer(env.render(), texte, dt)
    return pas, reussi


# ==============================================================================
#  PROGRAMME PRINCIPAL
# ==============================================================================
def lancer(env_choisi=ENV_CHOISI, episodes_override=None, afficher=True):
    cfg = CONFIGS[env_choisi]
    eid = trouver_env(cfg)
    episodes = episodes_override or cfg["episodes"]
    checkpoints = set(c for c in cfg["checkpoints"] if c <= episodes) | {episodes}
    print("Environnement :", eid, "|", cfg["titre"])
    if afficher and not _backend_ok:
        print("[info] backend interactif non force ; si l'image ne s'anime pas,")
        print("       regle Spyder : Preferences > Console IPython > Graphiques > Automatique.")

    env_train = gym.make(eid, **cfg["kwargs"])                       # rapide, sans image
    env_vis = gym.make(eid, render_mode="rgb_array", **cfg["kwargs"])# pour dessiner
    ecran = Ecran("RL - " + eid) if afficher else None

    n_etats = env_train.observation_space.n
    n_actions = env_train.action_space.n
    Q = np.zeros((n_etats, n_actions))
    rng = np.random.default_rng(0)

    # ---------- 1) AVANT : au hasard ----------
    print("\n=== 1) AVANT entrainement : l'agent agit AU HASARD ===")
    for i in range(cfg["n_avant"]):
        txt = "AVANT entrainement - actions AU HASARD  (essai {}/{})".format(i + 1, cfg["n_avant"])
        pas, _ = jouer_et_afficher(env_vis, Q, cfg, ecran, txt, cfg["dt"],
                                   au_hasard=True, rng=rng, max_pas_vue=45)
        print("   essai au hasard {}/{} : {} pas".format(i + 1, cfg["n_avant"], pas))

    # ---------- 2) ENTRAINEMENT (+ apercus aux checkpoints) ----------
    print("\n=== 2) ENTRAINEMENT ({} episodes) - apercus dans la fenetre ===".format(episodes))
    epsilon = cfg["eps_debut"]
    hist_rec = np.zeros(episodes)
    hist_succes = np.zeros(episodes)

    for ep in range(episodes):
        etat, _ = env_train.reset(seed=int(rng.integers(1 << 31)))
        fini, pas, total, reussi = False, 0, 0.0, False
        while not fini and pas < cfg["max_pas"]:
            if rng.random() < epsilon:
                a = env_train.action_space.sample()
            else:
                a = int(np.argmax(Q[etat]))
            s2, r, term, trunc, _ = env_train.step(a)
            fini = term or trunc
            Q[etat, a] += cfg["alpha"] * (r + cfg["gamma"] * np.max(Q[s2]) * (not term) - Q[etat, a])
            etat = s2; total += r; pas += 1
            if cfg["succes"](r, term):
                reussi = True
        epsilon = max(cfg["eps_fin"], epsilon * cfg["eps_decr"])
        hist_rec[ep] = total
        hist_succes[ep] = 1.0 if reussi else 0.0

        if (ep + 1) in checkpoints:
            taux = 100 * hist_succes[max(0, ep - 199):ep + 1].mean()
            txt = "ENTRAINEMENT - episode {}/{}   (succes recents {:.0f} %)".format(
                ep + 1, episodes, taux)
            print("   " + txt)
            if ecran is not None:
                jouer_et_afficher(env_vis, Q, cfg, ecran, txt, cfg["dt"], max_pas_vue=80)

    # ---------- 3) TEST ----------
    print("\n=== 3) TEST : la politique apprise (100 % glouton) ===")
    for i in range(cfg["n_test"]):
        txt = "TEST - l'agent applique ce qu'il a appris  (essai {}/{})".format(i + 1, cfg["n_test"])
        pas, reussi = jouer_et_afficher(env_vis, Q, cfg, ecran, txt, cfg["dt"], max_pas_vue=cfg["max_pas"])
        print("   test {}/{} : {} pas  ->  {}".format(i + 1, cfg["n_test"], pas,
                                                       "REUSSI" if reussi else "echoue"))

    # evaluation chiffree (rapide, sans image)
    succ, pas_moy = 0, []
    for _ in range(1000):
        etat, _ = env_train.reset(seed=int(rng.integers(1 << 31)))
        fini, pas, reussi = False, 0, False
        while not fini and pas < cfg["max_pas"]:
            etat, r, term, trunc, _ = env_train.step(int(np.argmax(Q[etat])))
            fini = term or trunc; pas += 1
            if cfg["succes"](r, term):
                reussi = True
        succ += int(reussi); pas_moy.append(pas)
    print("   -> sur 1000 tests : {:.1f} % de reussite, {:.1f} pas en moyenne".format(
        100 * succ / 1000, np.mean(pas_moy)))

    env_train.close(); env_vis.close()
    if ecran is not None:
        ecran.fermer()

    tracer(hist_rec, hist_succes, eid)
    return Q


def moyenne_glissante(x, f=100):
    if f <= 1 or len(x) < f:
        return x
    return np.convolve(x, np.ones(f) / f, mode="valid")


def tracer(hist_rec, hist_succes, eid, sauver=True):
    BLEU, VERT = "#2a78d6", "#1baf7a"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(moyenne_glissante(hist_rec, 100), color=BLEU, lw=2)
    a1.set_title("Recompense par episode (elle monte = il apprend)", fontsize=11, weight="bold")
    a1.set_xlabel("Episode"); a1.set_ylabel("Recompense"); a1.grid(alpha=0.25)
    a2.plot(100 * moyenne_glissante(hist_succes, 100), color=VERT, lw=2)
    a2.set_title("Taux de reussite (%)", fontsize=11, weight="bold")
    a2.set_xlabel("Episode"); a2.set_ylabel("% reussite"); a2.set_ylim(-2, 105); a2.grid(alpha=0.25)
    fig.suptitle("Apprentissage sur " + eid, fontsize=13, weight="bold")
    fig.tight_layout()
    if sauver:
        nom = "courbes_" + eid.replace("/", "_") + ".png"
        fig.savefig(nom, dpi=130); print("[figure]", nom)
    plt.show(block=True)   # garde les courbes affichees a la fin


if __name__ == "__main__":
    lancer()

# Projet 4 : Q-learning contre SARSA sur FrozenLake et CliffWalking

Module Apprentissage par Renforcement, Université Aube Nouvelle, 2025-2026.
Encadrant : Dr SOMDA Augustin.

Les trois algorithmes de contrôle TD (Q-learning, SARSA, Expected SARSA) sont
implémentés en NumPy pur. Gymnasium ne sert qu'à fournir les environnements,
aucune bibliothèque de RL n'intervient dans l'apprentissage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Si la commande `pip` n'est pas reconnue, notamment sous Windows :

```bash
python -m pip install -r requirements.txt
```

Contenu de requirements.txt : numpy, matplotlib, gymnasium[toy-text], pillow, imageio,
imageio-ffmpeg. Les deux derniers ne servent qu'à écrire la vidéo de démonstration.

Python 3.10 ou plus. Sous Gymnasium 1.2 et suivants, CliffWalking s'appelle `CliffWalking-v1` ;
le code essaie `v1` puis retombe sur `v0`, donc les deux versions fonctionnent.

## Arborescence

```
Projet_RL_BITIE_et_NATAMA/
├── main.py             tout le projet : agents, environnements, expériences, démo, rejeu
├── requirements.txt
├── README.md
└── resultats/             figures PNG, tableaux CSV et Markdown, Q-tables .npy
    
```

## Ce que produit l'exécution simple

Dans `resultats/`, pour chaque environnement : la vidéo MP4, la courbe d'apprentissage,
la politique gloutonne de chacun des trois algorithmes, les Q-tables rejouables, et sur
CliffWalking les trajectoires comparées. Les courbes multi-graines du rapport viennent
des expériences chiffrées, plus longues, listées plus bas.

## Exécution simple

Lancer le fichier tel quel, avec F5 dans Spyder ou `python main.py`, déroule la
démonstration complète sans rien avoir à taper : FrozenLake d'abord, CliffWalking
ensuite. Pour chaque environnement on voit l'agent agir au hasard, puis progresser au
fil de l'entraînement, puis jouer la politique apprise, avec Q-learning et SARSA côte
à côte.

Pour que l'animation s'affiche dans Spyder : Outils, Préférences, Console IPython,
Graphiques, Backend, choisir Automatique plutôt qu'En ligne, puis redémarrer le noyau.
Sans ce réglage la fenêtre reste figée, le script le signale au démarrage.

L'ordre des environnements se change à la ligne `PARCOURS_PAR_DEFAUT` en tête de
fichier. Le nombre d'aperçus montrés pendant l'entraînement se règle avec `--apercus`.

Un seul fichier Python porte le projet. Il est découpé en huit sections repérées par
des bandeaux de commentaires : environnements, agents, entraînement, figures,
expériences, démonstration visuelle, rejeu, ligne de commande.

## Commandes

Une commande par expérience du rapport. Ajouter `--rapide` pour une exécution
courte de vérification (quelques minutes deviennent quelques secondes).

```bash
# Expérience 1 : FrozenLake 4x4 et 8x8, taux de succès et récompense, 10 graines
python main.py frozenlake
python main.py frozenlake --cartes 4x4        # une seule carte

# Expérience 2 : CliffWalking, epsilon fixe à 0,1, 20 graines, trajectoires finales
python main.py cliffwalking

# Expérience 3 : effet du calendrier d'exploration
python main.py epsilon
python main.py epsilon --balayage             # valeurs de epsilon fixe de 0,3 à 0,01

# Les trois d'un coup
python main.py tout

# Démonstration visuelle (agent au hasard, puis apprentissage, puis politique finale)
python main.py parcours                       # FrozenLake puis CliffWalking
python main.py parcours --apercus 15          # plus d'aperçus pendant l'apprentissage
python main.py demo --env frozenlake8x8 --comparer
python main.py demo --comparer
python main.py demo --comparer --video resultats/demo_cliffwalking.mp4
python main.py demo --comparer --video resultats/demo.mp4 --duree 180

# Rejeu d'une politique sauvegardée, sans réentraînement
python main.py rejouer --model resultats/cliffwalking_qlearning.npy
python main.py rejouer --model resultats/frozenlake4x4_sarsa.npy --episodes 5
```

## Vidéos de démonstration

`resultats/demo_frozenlake4x4.mp4` et `resultats/demo_cliffwalking.mp4` durent
150 secondes chacune et montrent les trois temps demandés par le sujet : agent
aléatoire, aperçus à quatorze moments de l'apprentissage, puis les politiques finales
en glouton pur, avec la courbe d'apprentissage maintenue six secondes à la fin. Les
trois algorithmes y figurent côte à côte.

Elles sont produites automatiquement par l'exécution simple du script. La cadence est
calculée pour atteindre la durée passée à `--duree` ; si la capture est trop courte, le
script le signale et indique quoi augmenter. `--dt` règle la vitesse de l'animation à
l'écran : la baisser accélère la fabrication du fichier sans changer sa cadence finale.

```bash
python main.py demo --env cliffwalking --comparer --dt 0.01 --video sortie.mp4
```

L'écriture du MP4 demande `imageio` et `imageio-ffmpeg`, tous deux dans
requirements.txt. Un nom de fichier terminant par `.gif` produit un GIF à la place.

## Cache des campagnes

Chaque graine terminée est enregistrée dans `resultats/cache/` sous une clé qui encode
l'environnement, l'algorithme, le nombre d'épisodes et les hyperparamètres. Une campagne
interrompue reprend donc là où elle s'était arrêtée, ce qui compte pour FrozenLake 8x8
qui demande une vingtaine de minutes. Changer un seul hyperparamètre change la clé, le
cache ne peut pas resservir un résultat obtenu avec d'autres réglages. Supprimez le
dossier pour forcer un recalcul complet.

## Reproductibilité

Chaque exécution fixe trois graines : celle de l'environnement (`env.reset(seed=...)`),
celle de l'espace d'actions (`env.action_space.seed(...)`) et celle du générateur interne
de l'agent (`np.random.default_rng(graine)`). Relancer une commande deux fois donne
les mêmes chiffres.

Les campagnes utilisent les graines 0 à N-1. Sur FrozenLake, une seule graine ne prouve
rien : l'écart entre deux graines dépasse souvent l'écart entre deux algorithmes.

## Résultats attendus

CliffWalking, epsilon fixe à 0,1, alpha 0,25, gamma 1, 2000 épisodes, 20 graines :

| Algorithme | Récompense pendant l'entraînement | Retour de la politique gloutonne | Chemin |
| --- | --- | --- | --- |
| Q-learning | -50,5 | -13,0 | longe la falaise (optimal) |
| SARSA | -23,6 | -16,9 | passe par la ligne du haut (sûr) |
| Expected SARSA | -20,9 | -15,0 | chemin intermédiaire |

Q-learning apprend la meilleure politique et récolte la moins bonne récompense pendant
l'apprentissage : sa cible ignore le fait que epsilon le fera tomber.

FrozenLake glissant, évaluation gloutonne sur 100 épisodes par graine, 10 graines :

| Algorithme | 4x4 (20 000 épisodes) | 8x8 (40 000 épisodes) |
| --- | --- | --- |
| Q-learning | 73,6 % ± 3,2 | 59,2 % ± 5,6 |
| SARSA | 73,8 % ± 3,7 | 59,2 % ± 3,9 |
| Expected SARSA | 72,2 % ± 6,5 | 61,5 % ± 4,6 |

Aucun algorithme ne domine sur FrozenLake : la dispersion entre graines dépasse l'écart
entre méthodes. Les figures correspondantes sont dans `resultats/`.

## Travail en binôme

Le barème demande un dépôt Git avec des contributions visibles des deux membres.
Le travail a été réparti comme suit:
- NATAMA Ferdiand: FrozenLake
- BITIE Seydou : Cliffwalking

Nous avons ensuite fusionné les travaux en un seul avant de faire le push sur git

## Webographie

- https://medium.com/@priya61197/q-learning-vs-sarsa-b9e433dec930
- https://www.geeksforgeeks.org/artificial-intelligence/differences-between-q-learning-and-sarsa/
- https://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html
- https://tcnguyen.github.io/reinforcement_learning/sarsa_vs_q_learning.html
- https://www.baeldung.com/cs/q-learning-vs-sarsa
- https://gymnasium.farama.org/environments/toy_text/cliff_walking/
- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://medium.com/@mdmohsinkamal/reinforcement-learning-in-action-surviving-the-frozenlake-with-q-learning-and-sarsa-ce1725da42b1
- https://sesen.ai/blog/q-learning-frozen-lake-from-scratch
- https://github.com/NiravRaiyani/Reinforcement_Learning
- https://colab.research.google.com/github/yfletberliac/rlss-2019/blob/master/labs/solutions/RL.DP%2BQLearning%2BSARSA_solution.ipynb
- https://builtin.com/machine-learning/sarsa
- https://www.mysimulator.uk/ai-ml/ds-topic-33/
- https://github.com/SwamiKannan/CliffWalk
- https://www.geeksforgeeks.org/machine-learning/expected-sarsa-in-reinforcement-learning/
- https://campus.datacamp.com/fr/courses/reinforcement-learning-with-gymnasium-in-python/advanced-strategies-in-model-free-rl?ex=2
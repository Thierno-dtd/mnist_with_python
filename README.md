# TP MNIST - Reconnaissance de chiffres manuscrits

TP de Deep Learning  réalisé par **Thierno-dtd**.

Le but est d'entraîner un réseau de neurones avec Keras pour reconnaître des chiffres écrits à la main (de 0 à 9) à partir de la base MNIST, puis de comparer deux architectures : un réseau simple avec une seule couche cachée et un réseau plus profond avec trois couches cachées.

On peut utiliser le programme de deux façons : en mode console ou avec une interface graphique.

## La base MNIST

MNIST contient 70 000 images de chiffres manuscrits en niveaux de gris, de 28 × 28 pixels :

- 60 000 images pour l'entraînement ;
- 10 000 images pour le test.

Elle est téléchargée automatiquement par Keras au premier lancement (environ 11 Mo), il n'y a donc rien à récupérer à la main.

Avant l'entraînement, chaque image est :

- aplatie en un vecteur de 784 valeurs (28 × 28) ;
- normalisée en divisant les pixels par 255, pour avoir des valeurs entre 0 et 1.

Les étiquettes sont transformées en vecteurs one-hot (le chiffre 3 devient `[0,0,0,1,0,0,0,0,0,0]`).

## Les deux modèles

| | Modèle simple | Modèle profond |
|---|---|---|
| Couches cachées | 1 couche de 784 neurones | 3 couches de 784, 300 et 100 neurones |
| Activation | ReLU | ReLU |
| Couche de sortie | 10 neurones, softmax | 10 neurones, softmax |
| Nombre de paramètres | 623 290 | 882 050 |

Les deux sont compilés de la même façon : optimiseur Adam, perte `categorical_crossentropy`, métrique `accuracy`. Ils sont entraînés sur 10 epochs avec des batchs de 200 images, comme demandé dans l'énoncé.

## Organisation du code

Le projet est découpé en trois couches pour que les calculs ne soient pas mélangés avec l'affichage :

```
tp_mnist/
├── main.py                  # point d'entrée, choix console ou graphique
├── requirements.txt
├── metier/                  # tout ce qui concerne le réseau de neurones
│   ├── donnees.py           # chargement et prétraitement de MNIST
│   ├── modele.py            # construction des deux modèles
│   └── entrainement.py      # entraînement, évaluation, prédiction
├── application/
│   └── controleur.py        # fait le lien entre l'interface et le métier
└── interface/
    ├── console.py           # menu en ligne de commande
    └── graphique.py         # fenêtre avec customtkinter
```

Les deux interfaces passent par le même `ControleurMNIST`. Elles ne font aucun calcul elles-mêmes, ce qui permet de changer d'interface sans toucher au reste.

## Installation

Le projet a été développé avec Python 3.11 et TensorFlow 2.21. Attention, TensorFlow ne s'installe pas forcément avec les toutes dernières versions de Python.

```bash
python -m venv venv
```

Sous Windows :

```bash
venv\Scripts\activate
```

Sous Linux / macOS :

```bash
source venv/bin/activate
```

Puis :

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py
```

Le programme demande d'abord le mode de lancement : `1` pour la console, `2` pour l'interface graphique. Dans les deux cas on retrouve les mêmes actions :

1. Entraîner le modèle simple
2. Entraîner le modèle profond
3. Évaluer le modèle sur les 10 000 images de test
4. Prédire le chiffre d'une image de test (on donne son numéro, ou on en tire une au hasard dans l'interface graphique)
5. Afficher les courbes de perte et de précision pendant l'entraînement

Il faut entraîner un modèle (1 ou 2) avant de pouvoir utiliser les options 3, 4 et 5.

Dans l'interface graphique, l'entraînement tourne dans un thread séparé pour que la fenêtre ne se fige pas. La barre de progression avance à chaque epoch, et le détail epoch par epoch s'affiche à la fin avec le résultat. Pour la prédiction, on voit l'image testée et la probabilité donnée par le modèle pour chacun des 10 chiffres.

## Résultats obtenus

Sur une exécution, avec la graine fixée à 42 dans `main.py` :

| | Modèle simple | Modèle profond |
|---|---|---|
| Précision sur le test | 97,91 % | 97,69 % |
| Perte sur le test | 0,070 | 0,095 |
| Précision en entraînement (dernière epoch) | 99,85 % | 99,53 % |

Les chiffres peuvent varier légèrement d'une machine à l'autre, même avec la graine fixée.

Ce qu'on retient :

- les deux réseaux dépassent 97 % de bonnes réponses, ce qui est bien pour des réseaux entièrement connectés ;
- le modèle profond n'est **pas** meilleur que le modèle simple, alors qu'il a 40 % de paramètres en plus ;
- pour les deux modèles, la précision en entraînement monte presque à 100 % alors que celle du test plafonne autour de 98 %. Sur les courbes, la perte de validation du modèle profond remonte dès la 3e epoch : il commence à apprendre les images d'entraînement par cœur (surapprentissage). Pour le modèle simple ça arrive plus tard, vers la 8e epoch.

Pour améliorer ça, on pourrait ajouter du Dropout, arrêter l'entraînement plus tôt (EarlyStopping) ou passer à un réseau convolutif (CNN), plus adapté aux images.

## Remarque

Pendant l'entraînement, le jeu de test sert aussi de jeu de validation (`validation_data` dans `metier/entrainement.py`). Dans un vrai projet il faudrait plutôt séparer une partie des 60 000 images d'entraînement pour la validation, et garder le jeu de test uniquement pour l'évaluation finale.

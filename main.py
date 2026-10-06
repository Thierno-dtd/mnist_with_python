import numpy as np
import tensorflow as tf
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def demander_mode():
    print("=" * 50)
    print(" TP MNIST - Reconnaissance de chiffres manuscrits")
    print("=" * 50)
    print("1. Interface console")
    print("2. Interface graphique")

    while True:
        choix = input("Choisissez le mode de lancement (1 ou 2) : ").strip()
        if choix in ("1", "2"):
            return choix
        print("Merci de repondre par 1 ou 2.")


if __name__ == "__main__":
    mode = demander_mode()

    if mode == "2":
        from interface.graphique import lancer
    else:
        from interface.console import lancer

    lancer()
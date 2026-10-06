from tensorflow import keras
from tensorflow.keras.layers import Dense


def construire_modele_simple():
   
    modele = keras.Sequential()
    modele.add(Dense(784, input_dim=784, activation="relu"))
    modele.add(Dense(10, activation="softmax"))

    modele.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modele


def construire_modele_profond():
   
    modele = keras.Sequential()
    modele.add(Dense(784, input_dim=784, activation="relu"))
    modele.add(Dense(300, activation="relu"))
    modele.add(Dense(100, activation="relu"))
    modele.add(Dense(10, activation="softmax"))

    modele.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modele

from tensorflow import keras


def charger_donnees_brutes():
   
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def pretraiter_images(images):

    nb_images = images.shape[0]
    images_aplaties = images.reshape(nb_images, 784)
    images_normalisees = images_aplaties.astype("float32") / 255.0
    return images_normalisees


def pretraiter_etiquettes(etiquettes, nb_classes=10):
    """
    Exemple : l'étiquette 3 devient [0,0,0,1,0,0,0,0,0,0].
    """
    return keras.utils.to_categorical(etiquettes, nb_classes)


def preparer_jeu_de_donnees():
   
    x_train_brut, y_train_brut, x_test_brut, y_test_brut = charger_donnees_brutes()

    return {
        "x_train": pretraiter_images(x_train_brut),
        "y_train": pretraiter_etiquettes(y_train_brut),
        "x_test": pretraiter_images(x_test_brut),
        "y_test": pretraiter_etiquettes(y_test_brut),
        "x_test_brut": x_test_brut,
        "y_test_brut": y_test_brut,
    }

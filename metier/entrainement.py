def entrainer_modele(modele, donnees, epochs=10, batch_size=200):

    historique = modele.fit(
        donnees["x_train"],
        donnees["y_train"],
        validation_data=(donnees["x_test"], donnees["y_test"]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    return historique


def evaluer_modele(modele, donnees):
   
    perte, precision = modele.evaluate(
        donnees["x_test"], donnees["y_test"], verbose=0
    )
    return perte, precision


def predire_chiffre(modele, image_pretraitee):
    
    probabilites = modele.predict(image_pretraitee.reshape(1, 784), verbose=0)[0]
    chiffre_predit = probabilites.argmax()
    return chiffre_predit, probabilites
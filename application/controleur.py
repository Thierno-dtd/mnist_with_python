from metier import donnees as metier_donnees
from metier import modele as metier_modele
from metier import entrainement as metier_entrainement


class ControleurMNIST:


    def __init__(self):
        self.donnees = None
        self.modele = None
        self.historique = None
        self.nom_modele_courant = None


    def charger_les_donnees(self):
        if self.donnees is None:
            self.donnees = metier_donnees.preparer_jeu_de_donnees()
        return self.donnees


    def entrainer(self, type_modele="simple", epochs=10, batch_size=200):
       
        self.charger_les_donnees()

        if type_modele == "profond":
            self.modele = metier_modele.construire_modele_profond()
        else:
            self.modele = metier_modele.construire_modele_simple()

        self.nom_modele_courant = type_modele
        self.historique = metier_entrainement.entrainer_modele(
            self.modele, self.donnees, epochs=epochs, batch_size=batch_size
        )
        return self.historique


    def evaluer(self):
        if self.modele is None:
            return None
        return metier_entrainement.evaluer_modele(self.modele, self.donnees)


    def predire_sur_image_test(self, indice):
        if self.modele is None or self.donnees is None:
            return None

        nb_images_test = self.donnees["x_test"].shape[0]
        if indice < 0 or indice >= nb_images_test:
            return None

        image_pretraitee = self.donnees["x_test"][indice]
        chiffre_predit, probabilites = metier_entrainement.predire_chiffre(
            self.modele, image_pretraitee
        )

        return {
            "image_brute": self.donnees["x_test_brut"][indice],
            "etiquette_reelle": int(self.donnees["y_test_brut"][indice]),
            "chiffre_predit": int(chiffre_predit),
            "probabilites": probabilites,
        }


    def a_un_modele_entraine(self):
        return self.modele is not None
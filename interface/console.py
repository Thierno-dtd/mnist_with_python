import matplotlib.pyplot as plt

from application.controleur import ControleurMNIST


def afficher_menu():
    print("\n" + "=" * 50)
    print(" TP MNIST - Reconnaissance de chiffres manuscrits")
    print("=" * 50)
    print("1. Entrainer le modele simple (1 couche cachee - 784 neurones)")
    print("2. Entrainer le modele profond (3 couches cachees - 784/300/100)")
    print("3. Evaluer le modele courant sur les donnees de test")
    print("4. Predire un chiffre a partir d'une image de test")
    print("5. Afficher les courbes d'apprentissage (loss / accuracy)")
    print("0. Quitter")


def action_entrainer(controleur, type_modele):
    print(f"\nChargement des donnees et entrainement du modele '{type_modele}'...")
    print("(epochs=10, batch_size=200, comme demande dans l'enonce)\n")
    controleur.entrainer(type_modele=type_modele, epochs=10, batch_size=200)
    print("\nEntrainement termine.")


def action_evaluer(controleur):
    resultat = controleur.evaluer()
    if resultat is None:
        print("\nAucun modele entraine pour le moment. Choisissez d'abord 1 ou 2.")
        return
    perte, precision = resultat
    print(f"\nResultats sur le jeu de test :")
    print(f"  - perte (loss)      : {perte:.4f}")
    print(f"  - precision (accuracy) : {precision * 100:.2f} %")


def action_predire(controleur):
    if not controleur.a_un_modele_entraine():
        print("\nAucun modele entraine pour le moment. Choisissez d'abord 1 ou 2.")
        return

    
    while True:
        saisie = input("\nNumero de l'image de test a tester (0 a 9999) : ").strip()
        if not saisie.isdigit():
            print("Merci d'entrer un nombre entier.")
        else:
            resultat = controleur.predire_sur_image_test(int(saisie))
            if resultat is None:
                print("Indice invalide (doit etre compris entre 0 et 9999).")
            else:
                print(f"\nEtiquette reelle   : {resultat['etiquette_reelle']}")
                print(f"Chiffre predit     : {resultat['chiffre_predit']}")
                print("Probabilites du modele pour chaque chiffre :")
                for chiffre, proba in enumerate(resultat["probabilites"]):
                    print(f"  {chiffre} : {proba * 100:5.2f} %")

                # Affichage graphique de l'image testee (fenetre matplotlib)
                plt.figure(figsize=(3, 3))
                plt.imshow(resultat["image_brute"], cmap="gray")
                titre = (
                    f"Reel : {resultat['etiquette_reelle']} | "
                    f"Predit : {resultat['chiffre_predit']}"
                )
                plt.title(titre)
                plt.axis("off")
                plt.show()

        reponse = input("\nPredire une autre image ? (o/n) : ").strip().lower()
        if reponse != "o":
            break


def action_afficher_courbes(controleur):
    if controleur.historique is None:
        print("\nAucun entrainement effectue pour le moment. Choisissez d'abord 1 ou 2.")
        return

    historique = controleur.historique.history

    figure, (axe_perte, axe_precision) = plt.subplots(1, 2, figsize=(10, 4))

    axe_perte.plot(historique["loss"], label="apprentissage")
    axe_perte.plot(historique["val_loss"], label="validation")
    axe_perte.set_title("Perte (loss)")
    axe_perte.set_xlabel("epoch")
    axe_perte.legend()

    axe_precision.plot(historique["accuracy"], label="apprentissage")
    axe_precision.plot(historique["val_accuracy"], label="validation")
    axe_precision.set_title("Precision (accuracy)")
    axe_precision.set_xlabel("epoch")
    axe_precision.legend()

    figure.suptitle(f"Modele : {controleur.nom_modele_courant}")
    plt.tight_layout()
    plt.show()


def lancer():
    
    controleur = ControleurMNIST()

    while True:
        afficher_menu()
        choix = input("Votre choix : ").strip()

        if choix == "1":
            action_entrainer(controleur, "simple")
        elif choix == "2":
            action_entrainer(controleur, "profond")
        elif choix == "3":
            action_evaluer(controleur)
        elif choix == "4":
            action_predire(controleur)
        elif choix == "5":
            action_afficher_courbes(controleur)
        elif choix == "0":
            print("\nFin du programme. Au revoir !")
            break
        else:
            print("\nChoix invalide, merci de reessayer.")
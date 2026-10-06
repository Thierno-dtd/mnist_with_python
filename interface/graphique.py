# -*- coding: utf-8 -*-
"""
Couche INTERFACE - interface graphique moderne (customtkinter).

Comme pour interface/console.py, ce module ne fait AUCUN calcul
métier : il appelle uniquement le ControleurMNIST (couche application)
et affiche les résultats.

Particularité par rapport à la console : l'entraînement d'un réseau de
neurones prend du temps. Si on le lançait directement au clic d'un
bouton, la fenêtre se figerait ("ne répond pas") le temps de
l'entraînement. On lance donc l'entraînement dans un thread séparé
(threading.Thread), et on ne touche aux widgets graphiques que depuis
le thread principal, via self.after(...) — c'est la règle de base de
Tkinter (et donc de customtkinter, qui est construit par-dessus).
"""

import random
import re
import sys
import threading

import customtkinter as ctk
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from application.controleur import ControleurMNIST

# Apparence générale de l'application (thème sombre, accent bleu)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


import re


class CapteurProgressionKeras:

    MOTIF_EPOCH = re.compile(
        r"Epoch\s+(\d+)\s*/\s*(\d+)"
    )

    MOTIF_PROGRESS = re.compile(
        r"(?<!Epoch\s)(?<!Epoch)(\d+)\s*/\s*(\d+)"
    )

    MOTIF_ACCURACY = re.compile(
        r"accuracy:\s*([0-9.]+)"
    )

    MOTIF_LOSS = re.compile(
        r"loss:\s*([0-9.]+)"
    )

    MOTIF_VAL_ACCURACY = re.compile(
        r"val_accuracy:\s*([0-9.]+)"
    )

    MOTIF_VAL_LOSS = re.compile(
        r"val_loss:\s*([0-9.]+)"
    )

    def __init__(self, flux_original, sur_fin_epoch):
        self.flux_original = flux_original
        self.sur_fin_epoch = sur_fin_epoch

        self.tampon = ""

        self.epoch_actuel = 0
        self.total_epochs = 1

        # Empêche de déclencher deux fois le même epoch
        self.dernier_epoch_envoye = 0

    def write(self, texte):

        # -------------------------------------------------
        # Affichage normal dans le terminal
        # -------------------------------------------------

        self.flux_original.write(texte)
        self.flux_original.flush()

        # -------------------------------------------------
        # Accumulation
        # -------------------------------------------------

        self.tampon += texte

        # -------------------------------------------------
        # Epoch
        # -------------------------------------------------

        matches_epoch = list(
            self.MOTIF_EPOCH.finditer(self.tampon)
        )

        if matches_epoch:

            match_epoch = matches_epoch[-1]

            self.epoch_actuel = int(
                match_epoch.group(1)
            )

            self.total_epochs = int(
                match_epoch.group(2)
            )

        # -------------------------------------------------
        # Progression batch
        # -------------------------------------------------

        matches_progress = list(
            self.MOTIF_PROGRESS.finditer(self.tampon)
        )

        if not matches_progress:
            return len(texte)

        # On prend la dernière progression
        match_progress = matches_progress[-1]

        batch_actuel = int(
            match_progress.group(1)
        )

        total_batches = int(
            match_progress.group(2)
        )

        # Pas encore arrivé au dernier batch
        if batch_actuel != total_batches:
            return len(texte)

        # -------------------------------------------------
        # Éviter un double déclenchement
        # -------------------------------------------------

        if self.epoch_actuel == self.dernier_epoch_envoye:
            return len(texte)

        # -------------------------------------------------
        # Métriques
        # -------------------------------------------------

        match_accuracy = self.MOTIF_ACCURACY.search(
            self.tampon
        )

        match_loss = self.MOTIF_LOSS.search(
            self.tampon
        )

        if not match_accuracy or not match_loss:
            return len(texte)

        logs = {
            "accuracy": float(
                match_accuracy.group(1)
            ),
            "loss": float(
                match_loss.group(1)
            ),
        }

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        match_val_accuracy = self.MOTIF_VAL_ACCURACY.search(
            self.tampon
        )

        match_val_loss = self.MOTIF_VAL_LOSS.search(
            self.tampon
        )

        if match_val_accuracy:
            logs["val_accuracy"] = float(
                match_val_accuracy.group(1)
            )

        if match_val_loss:
            logs["val_loss"] = float(
                match_val_loss.group(1)
            )

        # -------------------------------------------------
        # Envoi vers Tkinter
        # -------------------------------------------------

        self.dernier_epoch_envoye = self.epoch_actuel

        self.sur_fin_epoch(
            self.epoch_actuel,
            self.total_epochs,
            logs
        )

        # -------------------------------------------------
        # Nettoyage du tampon
        # -------------------------------------------------

        self.tampon = self.tampon[
            match_progress.end():
        ]

        return len(texte)

    def flush(self):
        self.flux_original.flush()

        
class FenetrePrincipale(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.controleur = ControleurMNIST()
        self.epoques_log = []

        self.title("TP MNIST - Reconnaissance de chiffres manuscrits")
        self.geometry("1050x680")
        self.minsize(900, 600)

        # La fenêtre est divisée en 2 colonnes : le menu (gauche) et la
        # zone de contenu (droite). On utilise le système "grid" pour
        # que la zone de contenu s'agrandisse quand on redimensionne
        # la fenêtre.
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._construire_menu_lateral()
        self._construire_zone_contenu()

        self._afficher_accueil()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------
    def _construire_menu_lateral(self):
        menu = ctk.CTkFrame(self, width=230, corner_radius=0)
        menu.grid(row=0, column=0, sticky="nsw")
        menu.grid_propagate(False)

        titre = ctk.CTkLabel(
            menu, text="TP MNIST", font=ctk.CTkFont(size=22, weight="bold")
        )
        titre.pack(padx=20, pady=(25, 5), anchor="w")

        sous_titre = ctk.CTkLabel(
            menu,
            text="Reconnaissance de\nchiffres manuscrits",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
            justify="left",
        )
        sous_titre.pack(padx=20, pady=(0, 25), anchor="w")

        self.bouton_simple = ctk.CTkButton(
            menu, text="1. Entrainer modele simple",
            command=lambda: self._demarrer_entrainement("simple"),
        )
        self.bouton_simple.pack(padx=20, pady=8, fill="x")

        self.bouton_profond = ctk.CTkButton(
            menu, text="2. Entrainer modele profond",
            command=lambda: self._demarrer_entrainement("profond"),
        )
        self.bouton_profond.pack(padx=20, pady=8, fill="x")

        self.bouton_evaluer = ctk.CTkButton(
            menu, text="3. Evaluer le modele", command=self._action_evaluer
        )
        self.bouton_evaluer.pack(padx=20, pady=8, fill="x")

        self.bouton_predire = ctk.CTkButton(
            menu, text="4. Predire une image", command=self._action_predire
        )
        self.bouton_predire.pack(padx=20, pady=8, fill="x")

        self.bouton_courbes = ctk.CTkButton(
            menu, text="5. Afficher les courbes", command=self._action_courbes
        )
        self.bouton_courbes.pack(padx=20, pady=8, fill="x")

        self.tous_les_boutons = [
            self.bouton_simple,
            self.bouton_profond,
            self.bouton_evaluer,
            self.bouton_predire,
            self.bouton_courbes,
        ]

        # Zone basse du menu : statut + barre de progression + quitter
        bas_menu = ctk.CTkFrame(menu, fg_color="transparent")
        bas_menu.pack(side="bottom", fill="x", padx=20, pady=20)

        self.barre_progression = ctk.CTkProgressBar(bas_menu)
        self.barre_progression.set(0)
        self.barre_progression.pack(fill="x", pady=(0, 8))

        self.label_statut = ctk.CTkLabel(
            bas_menu, text="Aucun modele entraine", font=ctk.CTkFont(size=12),
            text_color="gray70", wraplength=190, justify="left",
        )
        self.label_statut.pack(anchor="w", pady=(0, 12))

        bouton_quitter = ctk.CTkButton(
            bas_menu, text="Quitter", fg_color="#8b2c2c",
            hover_color="#6e2222", command=self.destroy,
        )
        bouton_quitter.pack(fill="x")

    def _construire_zone_contenu(self):
        self.zone_contenu = ctk.CTkFrame(self, corner_radius=0,
                                          fg_color="transparent")
        self.zone_contenu.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        self.zone_contenu.grid_columnconfigure(0, weight=1)
        self.zone_contenu.grid_rowconfigure(0, weight=1)

    def _effacer_zone_contenu(self):
        for enfant in self.zone_contenu.winfo_children():
            enfant.destroy()

    def _afficher_accueil(self):
        self._effacer_zone_contenu()
        carte = ctk.CTkFrame(self.zone_contenu, corner_radius=15)
        carte.grid(row=0, column=0, sticky="nsew")
        carte.grid_columnconfigure(0, weight=1)
        carte.grid_rowconfigure(0, weight=1)

        message = ctk.CTkLabel(
            carte,
            text=(
                "Bienvenue !\n\n"
                "1) Entrainez un modele (bouton 1 ou 2 dans le menu)\n"
                "2) Evaluez-le, affichez les courbes, testez une prediction"
            ),
            font=ctk.CTkFont(size=16),
            justify="center",
        )
        message.grid(row=0, column=0)

    # ------------------------------------------------------------------
    # Utilitaires d'etat (thread-safe : appeles uniquement via self.after)
    # ------------------------------------------------------------------
    def _activer_boutons(self, actif):
        etat = "normal" if actif else "disabled"
        for bouton in self.tous_les_boutons:
            bouton.configure(state=etat)

    def _maj_statut(self, texte):
        self.label_statut.configure(text=texte)

    # ------------------------------------------------------------------
    # Action 1 / 2 : entrainement (dans un thread separe)
    # ------------------------------------------------------------------
    def _demarrer_entrainement(self, type_modele):
        self._activer_boutons(False)
        self.barre_progression.set(0)
        self._maj_statut("Chargement des donnees...")
        self._effacer_zone_contenu()

        # Le detail epoch par epoch est capture en arriere-plan (par
        # CapteurProgressionKeras) mais n'est PAS affiche pendant
        # l'entrainement : on ne montre ici qu'un ecran de chargement
        # simple. Le detail sera affiche d'un coup, juste apres, sur
        # l'ecran de resultat (voir _action_evaluer).
        self.epoques_log = []

        nom_affiche = "simple (1 couche cachee)" if type_modele == "simple" \
            else "profond (3 couches cachees)"

        carte = ctk.CTkFrame(self.zone_contenu, corner_radius=15)
        carte.grid(row=0, column=0, sticky="nsew")
        carte.grid_columnconfigure(0, weight=1)
        carte.grid_rowconfigure(0, weight=1)

        conteneur_centre = ctk.CTkFrame(carte, fg_color="transparent")
        conteneur_centre.grid(row=0, column=0)

        label = ctk.CTkLabel(
            conteneur_centre,
            text=f"Entrainement du modele {nom_affiche}\nen cours...",
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="center",
        )
        label.pack(pady=(0, 20))

        barre_chargement = ctk.CTkProgressBar(
            conteneur_centre, mode="indeterminate", width=260
        )
        barre_chargement.pack()
        barre_chargement.start()

        thread = threading.Thread(
            target=self._tache_entrainement, args=(type_modele,), daemon=True
        )
        thread.start()

    def _tache_entrainement(self, type_modele):
        """Cette fonction s'execute dans le thread d'arriere-plan."""

        def sur_fin_epoch(epoch, total_epochs, logs):
            # Ne JAMAIS toucher un widget directement depuis ce thread :
            # on planifie la mise a jour sur le thread principal.
            self.after(
                0,
                lambda: self._maj_progression_entrainement(
                    epoch, total_epochs, logs
                ),
            )

        # On remplace temporairement la sortie standard par notre
        # "espion" : le temps de l'entrainement, tout ce que Keras
        # imprime (verbose=1, tel quel dans metier/entrainement.py)
        # passe par CapteurProgressionKeras, qui met a jour la barre de
        # progression au passage puis relaie le texte vers le vrai
        # terminal. metier/entrainement.py et application/controleur.py
        # n'ont besoin de rien savoir de tout ca.
        flux_original = sys.stdout
        sys.stdout = CapteurProgressionKeras(flux_original, sur_fin_epoch)
        try:
            self.controleur.entrainer(
                type_modele=type_modele, epochs=10, batch_size=200
            )
        finally:
            sys.stdout = flux_original

        self.after(0, self._fin_entrainement)

    def _maj_progression_entrainement(self, epoch, total_epochs, logs):
        # On memorise la ligne (meme esprit que la console : "Epoch
        # X/Y" puis les metriques) sans l'afficher tout de suite - elle
        # sera affichee d'un bloc sur l'ecran de resultat final.
        ligne = f"Epoch {epoch}/{total_epochs}\n"
        ligne += (
            f"  accuracy: {logs.get('accuracy', 0):.4f} - "
            f"loss: {logs.get('loss', 0):.4f}"
        )
        if "val_accuracy" in logs:
            ligne += (
                f" - val_accuracy: {logs['val_accuracy']:.4f} - "
                f"val_loss: {logs['val_loss']:.4f}"
            )
        self.epoques_log.append(ligne)

        progression = epoch / total_epochs
        self.barre_progression.set(progression)
        self._maj_statut(f"Entrainement... epoch {epoch}/{total_epochs}")

    def _fin_entrainement(self):
        self._activer_boutons(True)
        self._maj_statut(
            f"Modele '{self.controleur.nom_modele_courant}' entraine."
        )
        # Redirection automatique vers l'ecran de resultat (equivalent
        # du menu "3. Evaluer le modele").
        self._action_evaluer()

    # ------------------------------------------------------------------
    # Action 3 : evaluation (resultat final + detail epoch par epoch)
    # ------------------------------------------------------------------
    def _action_evaluer(self):
        resultat = self.controleur.evaluer()
        self._effacer_zone_contenu()

        carte = ctk.CTkFrame(self.zone_contenu, corner_radius=15)
        carte.grid(row=0, column=0, sticky="nsew")
        carte.grid_columnconfigure(0, weight=1)

        if resultat is None:
            carte.grid_rowconfigure(0, weight=1)
            label = ctk.CTkLabel(
                carte, text="Aucun modele entraine.\nChoisissez d'abord 1 ou 2.",
                font=ctk.CTkFont(size=18), justify="center",
            )
            label.grid(row=0, column=0)
            return

        perte, precision = resultat

        # -- Encart en valeur : le resultat global, en premier --
        encart = ctk.CTkFrame(
            carte, corner_radius=12,
            fg_color="#1f4d2e", border_width=2, border_color="#3fae5c",
        )
        encart.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        encart.grid_columnconfigure((0, 1), weight=1)

        titre = ctk.CTkLabel(
            encart, text="Resultat final sur les 10 000 images de test",
            font=ctk.CTkFont(size=15, weight="bold"), text_color="#e8fff0",
        )
        titre.grid(row=0, column=0, columnspan=2, pady=(12, 4))

        label_precision = ctk.CTkLabel(
            encart, text=f"Precision : {precision * 100:.2f} %",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#7ee89a",
        )
        label_precision.grid(row=1, column=0, pady=(0, 14), padx=10)

        label_perte = ctk.CTkLabel(
            encart, text=f"Perte (loss) : {perte:.4f}",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#7ee89a",
        )
        label_perte.grid(row=1, column=1, pady=(0, 14), padx=10)

        # -- Detail epoch par epoch, en dessous du resultat global --
        if self.epoques_log:
            carte.grid_rowconfigure(2, weight=1)

            sous_titre = ctk.CTkLabel(
                carte, text="Detail de l'entrainement, epoch par epoch",
                font=ctk.CTkFont(size=13, weight="bold"), text_color="gray70",
            )
            sous_titre.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 5))

            zone_journal = ctk.CTkTextbox(
                carte, font=ctk.CTkFont(family="Courier New", size=13)
            )
            zone_journal.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 15))
            zone_journal.insert("end", "\n".join(self.epoques_log) + "\n")
            zone_journal.configure(state="disabled")

    # ------------------------------------------------------------------
    # Action 4 : prediction sur une image de test
    # ------------------------------------------------------------------
    def _action_predire(self):
        self._effacer_zone_contenu()

        conteneur = ctk.CTkFrame(self.zone_contenu, corner_radius=15)
        conteneur.grid(row=0, column=0, sticky="nsew")
        conteneur.grid_columnconfigure(0, weight=1)
        conteneur.grid_columnconfigure(1, weight=1)
        conteneur.grid_rowconfigure(1, weight=1)

        barre_saisie = ctk.CTkFrame(conteneur, fg_color="transparent")
        barre_saisie.grid(row=0, column=0, columnspan=2, pady=15, sticky="ew")
        barre_saisie.grid_columnconfigure(0, weight=1)

        self.champ_indice = ctk.CTkEntry(
            barre_saisie, placeholder_text="Numero d'image (0 a 9999)"
        )
        self.champ_indice.grid(row=0, column=0, padx=(15, 8), sticky="ew")

        bouton_aleatoire = ctk.CTkButton(
            barre_saisie, text="Image aleatoire", width=140,
            command=self._choisir_image_aleatoire,
        )
        bouton_aleatoire.grid(row=0, column=1, padx=8)

        bouton_predire = ctk.CTkButton(
            barre_saisie, text="Predire", width=100,
            command=self._lancer_prediction,
        )
        bouton_predire.grid(row=0, column=2, padx=(8, 15))

        # Zone ou l'image et les probabilites seront affichees
        self.zone_resultat_prediction = ctk.CTkFrame(
            conteneur, fg_color="transparent"
        )
        self.zone_resultat_prediction.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=15, pady=(0, 15)
        )
        self.zone_resultat_prediction.grid_columnconfigure(0, weight=1)
        self.zone_resultat_prediction.grid_columnconfigure(1, weight=2)

        if not self.controleur.a_un_modele_entraine():
            info = ctk.CTkLabel(
                self.zone_resultat_prediction,
                text="Aucun modele entraine. Choisissez d'abord 1 ou 2.",
                font=ctk.CTkFont(size=16),
            )
            info.grid(row=0, column=0, columnspan=2)

    def _choisir_image_aleatoire(self):
        indice = random.randint(0, 9999)
        self.champ_indice.delete(0, "end")
        self.champ_indice.insert(0, str(indice))
        self._lancer_prediction()

    def _lancer_prediction(self):
        if not self.controleur.a_un_modele_entraine():
            return

        saisie = self.champ_indice.get().strip()
        if not saisie.isdigit():
            return
        indice = int(saisie)

        resultat = self.controleur.predire_sur_image_test(indice)
        if resultat is None:
            return

        for enfant in self.zone_resultat_prediction.winfo_children():
            enfant.destroy()

        # -- Colonne de gauche : l'image du chiffre --
        image_pil = Image.fromarray(resultat["image_brute"]).convert("L")
        image_pil = image_pil.resize((220, 220), Image.NEAREST)
        image_ctk = ctk.CTkImage(light_image=image_pil, dark_image=image_pil,
                                  size=(220, 220))

        colonne_image = ctk.CTkFrame(self.zone_resultat_prediction,
                                      fg_color="transparent")
        colonne_image.grid(row=0, column=0, sticky="n", padx=20, pady=10)

        label_image = ctk.CTkLabel(colonne_image, image=image_ctk, text="")
        label_image.image = image_ctk  # garder une reference (sinon Tkinter
        # "oublie" l'image et l'affichage devient blanc)
        label_image.pack()

        correct = resultat["etiquette_reelle"] == resultat["chiffre_predit"]
        couleur = "#3fae5c" if correct else "#c0392b"
        texte_resultat = (
            f"Etiquette reelle : {resultat['etiquette_reelle']}\n"
            f"Chiffre predit : {resultat['chiffre_predit']}"
        )
        label_resultat = ctk.CTkLabel(
            colonne_image, text=texte_resultat, text_color=couleur,
            font=ctk.CTkFont(size=17, weight="bold"), justify="center",
        )
        label_resultat.pack(pady=10)

        # -- Colonne de droite : probabilites par chiffre (barres) --
        colonne_probas = ctk.CTkFrame(self.zone_resultat_prediction,
                                       fg_color="transparent")
        colonne_probas.grid(row=0, column=1, sticky="new", padx=20, pady=10)
        colonne_probas.grid_columnconfigure(1, weight=1)

        titre_probas = ctk.CTkLabel(
            colonne_probas, text="Probabilites du modele",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        titre_probas.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        for chiffre, proba in enumerate(resultat["probabilites"]):
            ligne = chiffre + 1
            ctk.CTkLabel(colonne_probas, text=str(chiffre), width=20).grid(
                row=ligne, column=0, sticky="w", pady=3
            )
            barre = ctk.CTkProgressBar(colonne_probas)
            barre.set(float(proba))
            barre.grid(row=ligne, column=1, sticky="ew", padx=8, pady=3)
            ctk.CTkLabel(colonne_probas, text=f"{proba * 100:5.1f} %",
                         width=55).grid(row=ligne, column=2, sticky="e")

    # ------------------------------------------------------------------
    # Action 5 : courbes d'apprentissage (matplotlib integre a la fenetre)
    # ------------------------------------------------------------------
    def _action_courbes(self):
        self._effacer_zone_contenu()

        carte = ctk.CTkFrame(self.zone_contenu, corner_radius=15)
        carte.grid(row=0, column=0, sticky="nsew")
        carte.grid_columnconfigure(0, weight=1)
        carte.grid_rowconfigure(0, weight=1)

        if self.controleur.historique is None:
            label = ctk.CTkLabel(
                carte,
                text="Aucun entrainement effectue.\nChoisissez d'abord 1 ou 2.",
                font=ctk.CTkFont(size=16), justify="center",
            )
            label.grid(row=0, column=0)
            return

        historique = self.controleur.historique.history

        # Couleurs alignees sur le theme sombre de l'application, pour
        # que les graphiques ne "jurent" pas avec le reste de la fenetre.
        couleur_fond = "#2b2b2b"
        couleur_texte = "#dcdcdc"
        couleur_grille = "#444444"

        figure = Figure(figsize=(8, 4), dpi=100)
        figure.patch.set_facecolor(couleur_fond)

        def styliser(axe, titre):
            axe.set_facecolor(couleur_fond)
            axe.set_title(titre, color=couleur_texte)
            axe.set_xlabel("epoch", color=couleur_texte)
            axe.tick_params(colors=couleur_texte)
            axe.grid(color=couleur_grille, linewidth=0.5)
            for bord in axe.spines.values():
                bord.set_color(couleur_grille)
            legende = axe.legend(facecolor=couleur_fond, edgecolor=couleur_grille)
            for texte in legende.get_texts():
                texte.set_color(couleur_texte)

        axe_perte = figure.add_subplot(1, 2, 1)
        axe_perte.plot(historique["loss"], label="apprentissage", color="#3fa7d6")
        axe_perte.plot(historique["val_loss"], label="validation", color="#ee6c4d")
        styliser(axe_perte, "Perte (loss)")

        axe_precision = figure.add_subplot(1, 2, 2)
        axe_precision.plot(historique["accuracy"], label="apprentissage",
                            color="#3fa7d6")
        axe_precision.plot(historique["val_accuracy"], label="validation",
                            color="#ee6c4d")
        styliser(axe_precision, "Precision (accuracy)")

        figure.tight_layout()

        canevas = FigureCanvasTkAgg(figure, master=carte)
        canevas.draw()
        canevas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10,
                                      pady=10)


def lancer():
    """Point d'entree de la couche interface graphique."""
    application = FenetrePrincipale()
    application.mainloop()
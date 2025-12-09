# -*- coding: utf-8 -*-
"""
8 decembre 2025

Justine Hamelin
"""
# %% Libraries
import os
import pandas as pd

from traiter_pluvios_sherb import ajoute_manquantes, krig_pluvio, visualiser_grilles_csv, format_pcswmm

# %% Read files
main_dir = os.path.realpath(os.path.dirname(__file__))

# %% 1. Completer les evenements avec tous les pas de temps

#ajoute_manquantes(fichier_o, fichier_modif, date_debut, date_fin, pas_temps)
fichier_modif = main_dir+"/precip_data/precip_complete.csv"

ajoute_manquantes(main_dir+"/precip_data/pluviometres.csv", fichier_modif,
                  "2025-05-17", "2025-05-19", "5min")

# %% 2. Flitrer les donnees aberrantes

#Pas encore de fonction pour cette etape

# %% 3. Kriger les donnees des pluviometres sur une grille "radar"

#krig_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios)
resultats = krig_pluvio(main_dir+'/radar_grid/radar_grid_xyz.csv',
                        main_dir+'/precipitations/Emplacement des pluviomètres/pluvio_xyz.csv',
                        fichier_modif)

#Enregister 1 pas de temps (Exemple)
pasdetemps = "2025-05-17 13:30:00"
grille = resultats[pd.to_datetime(pasdetemps)]
grille.to_csv(main_dir + "/data_krig/20250517_1330.csv")

# %% 4. Validation - Figure
# *Pour le moment, fonctionne pour 1 pas de temps qui autait ete enregistrer de "resultats" a l'etape 3

#visualiser_grilles_csv(grille_krigee, radar_grid, donnees_pluvios, emplacements_pluvios, date_heure)
visualiser_grilles_csv(main_dir+"/data_krig/20250517_1330.csv", main_dir+"/radar_grid/radar_grid_xyz.csv",
                       fichier_modif, main_dir+"/precipitations/Emplacement des pluviomètres/pluvio_xyz.csv",
                       pasdetemps)

# %% 5. Formater les donnees pour PCSWMM
index_garde = (list(range(811, 819)) + list(range(844, 852)) +
                list(range(877, 885)) + list(range(910, 918)) +
                list(range(943, 951)) + list(range(976, 984)))

#format_pcswmm(index_garde, chemin_resultats, chemin_timeseries)
timeseries_estim = format_pcswmm(index_garde, main_dir+"/data_krig/dict_df.pkl",
              main_dir+"/precip_data/20250517_20250519/timeseries2.csv")

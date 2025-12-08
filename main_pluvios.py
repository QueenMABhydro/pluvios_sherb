# -*- coding: utf-8 -*-
"""
8 decembre 2025

Justine Hamelin
"""
# %% Libraries
import os
import pandas as pd

from traiter_pluvios_sherb import ajoute_manquantes, krig_pluvio, visualiser_grilles_csv

# %% Read files
main_dir = os.path.realpath(os.path.dirname(__file__))

# %% 1. Completer les evenements avec tous les pas de temps

#ajoute_manquantes(fichier_o, fichier_modif, date_debut, date_fin, pas_temps)
fichier_modif = main_dir+"/precip_data/precip_complete_test2.csv"

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

index_garde = {812, 813, 814, 815, 816, 817, 818,           #Correspond aux index des cellules de la grille
               845, 846, 847, 848, 849, 850, 851,           #radar qui couvre le bassin versant a l'etude.
               878, 879, 880, 881, 882, 883, 884,
               911, 912, 913, 914, 915, 916, 917}           #On les a determiner dans PCSWMM

#Couper la grille pour garder seulement la region etudiee
for key, df in resultats.items():
    index = sorted(i for i in index_garde if i in df.index)
    resultats[key] = df.loc[index]

timeseries = pd.concat(resultats, names=["Temps", "points"])

timeseries_estim = timeseries["estimation"].unstack("points")
timeseries_estim = timeseries_estim.fillna(0)                      #Remplacer les nan par 0 pour PCSWMM
timeseries_estim.index = pd.to_datetime(timeseries_estim.index)

timeseries_estim.to_csv(main_dir + "/precip_data/20250517_20250519/timeseries3.csv")

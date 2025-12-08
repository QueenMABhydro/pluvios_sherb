#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des precipitometres de la
Ville de Sherbrooke.
- "ajoute_manquantes": Ajouter explicitement les donnees manquantes
- "filtre_precip_louche": filtrer les valeurs qui semblent aberrantes
(avant de coder cette fonction, verifier si le code existe deja)
- "krig_pluvio" : Kriger les donnees des pluviometres sur une grille "radar"
couvrant la region etudiee

@author: Marie-Amelie Boucher, USherbrooke
"""

import pandas as pd
import numpy as np
from pykrige import OrdinaryKriging
import matplotlib.pyplot as plt


def ajoute_manquantes(fichier_o, fichier_modif, date_debut, date_fin, pas_temps):
    """
    Parameters
    ----------
    fichier_o : chaine de caracteres
        Chemin vers le fichier .csv des donnees brutes des pluviometres
        Telecharger sur le site de la ville de Sherbrooke: https://pluviometres.ville.sherbrooke.qc.ca/
        *Le Excel provenant du site est .xls et doit etre change pour .csv prealablement
    fichier_modif : chaine de caracteres
        Chemin et nom du fichier de destination (AVEC les dates manquantes ajoutees). format csv
    date_debut : chaine de caracteres
        Date de debut de la periode d'interet. Exemple: '2018-01-01'
    date_fin : chaine de caracteres
        Date de fin de la periode d'interet. Exemple: '2018-12-31'
    pas_temps : chaine de caracteres
        Pas de temps a utiliser dans le fichier de destination. Exemple: 5 minutes = '5min'
        La liste des frequences possibles est ici:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Returns
    -------
    donnees_pluvio_complet: data frame (Enregistre en format csv)
        Contient une serie complete dans laquelle les donnees manquantes sont identifiees par des NaN
    """

    df = pd.read_csv(fichier_o, sep=',')                                                #Lire donnees brutes

    if str(df.iloc[-1, 0]).strip().upper() == "TOTAL":                                  #Enlever la derniere ligne "TOTAL" du fichier
        df = df.iloc[:-1]                                                               #Qui est propre au site de la ville de Sherbrooke

    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')         #Mettre les dates en index
    df = df.set_index('Date')

    df.index = pd.to_datetime(
        df.index.date.astype(str) + ' ' + df['Période'].str.split(' à ').str[0])        #Inclure le temps a l'index
    df = df.drop(columns='Période')

    serie_index = pd.date_range(start= date_debut , end= date_fin, freq= pas_temps)     #Index de la serie complete

    donnees_pluvio_complet = pd.DataFrame(index =serie_index, columns=df.columns)       #Serie vide complete
    donnees_pluvio_complet.update(df)                                                   #Ajout des donnees des pluviometres

    donnees_pluvio_complet.to_csv(fichier_modif, sep=',')                               #Enregistrer le dataframe en .csv

    return donnees_pluvio_complet


def krig_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios):
    """
    Parameters
    ----------
    radar_grid : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees des cellules ou les precip sont krigees.
        C'est une grille 500 x 500 m (*radar_grid_xyz.csv)
    emplacements_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees 'x' et 'y', en metre, des pluviometres
        (*pluvio_xyz.csv)
    donnees_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les donnees de tous les pluviometres et pour tous
        les pas de temps de la periode (*precip_complete.csv)

    Returns
    -------
    resultats : Dictionary
        Dictionnaire ou on retrouve un dataframe pour chaque pas de temps.
        Chaque df a une colonne 'x', 'y' (representant chaque case dans la grille radar), 'estimation' et 'variance'
        Pour selectionner 1 seule grille : exemple : resultats[pd.to_datetime("2025-05-17 13:30:00")]
    """
    # Grille radar - centroides xyz (500x500m)
    radar_grid = pd.read_csv(radar_grid)
    radar_grid = radar_grid.set_index('id')
    radar_grid = radar_grid[['X','Y','ELEV_1']]
    radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6

    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])

    # Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    # Donnees pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)

    #Rassembler les donnees des pluviometres aux pluviometres
    stations = donnees_pluvios.columns.values
    temps = donnees_pluvios.index
    df = pd.DataFrame(index=temps)

    for station in pluvio_xyz.index :
        df[f'X_{station}'] = pluvio_xyz.loc[station, 'X']
        df[f'Y_{station}'] = pluvio_xyz.loc[station, 'Y']
        df[f'precip_{station}'] = donnees_pluvios[station]

    #Krigeage
    resultats = {}
    for t in temps :
        ligne = df.loc[t]       #Info des pluviometres pour 1 pas de temps

        x_val = np.array([ligne[f"X_{st}"]     for st in stations])        #Coordonnees X des pluviometres
        y_val = np.array([ligne[f"Y_{st}"]     for st in stations])        #Coordonnees Y des pluviometres
        precip = np.array([ligne[f"precip_{st}"] for st in stations])      #Precip aux pluviometres

        #DataFrame du pas de temps
        result_t = pd.DataFrame({"x":gx, "y":gy, "estimation":np.nan, "variance":np.nan},
                                index= radar_grid.index)

        if len(precip) > 0 and not np.all(np.isnan(precip)):
            krig = OrdinaryKriging(x_val,y_val,precip,variogram_model='spherical',
                                   nlags=2, enable_plotting=False, verbose=False,
                                   enable_statistics=False, coordinates_type='euclidean',
                                   pseudo_inv=True, weight=False)

            estim, var = krig.execute("points", gx, gy)

            result_t["estimation"] = estim
            result_t["variance"] = var

        else : pass
        resultats[t] = result_t

    return resultats


def visualiser_grilles_csv(grille_krigee, radar_grid, donnees_pluvios, emplacements_pluvios, date_heure):
    """
    Parameters
    ----------
    grille_krigee : Chaine de caracteres
        Chemin vers le fichier .csv contenant les donnees krigees.
        Le fichier csv contient 5 colonnes, soient une colonne d'index sans titre
        une colonne 'x' avec les longitudes en metres, une colonne 'y' avec les latitudes en metres,
        une colonne 'estimation' avec les quantitees (absolues) de pluie estimees par krigeage en mm,
        et une colonne 'variance' avec la variance du krigeage en mm^2
    radar_grid : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees des cellules ou les precip sont krigees.
        C'est une grille 500 x 500 m
    donnees_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les donnees de tous les pluviometres et pour tous
        les pas de temps de la periode (*precip_complete.csv)
    emplacements_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees 'x' et 'y', en metre, des pluviometres
        (*pluvio_xyz.csv)
    date_heure : Chaine de caracteres
        Pas de temps observe, avec un format comme cet exemple : "2025-05-17 13:30:00"

    Returns
    -------
    None.

    """
    # Preparation des donnees
    donnees = pd.read_csv(grille_krigee)
    donnees = donnees['estimation']
    radar_grid = pd.read_csv(radar_grid)

    lon = np.array(radar_grid['X'])
    lat = np.array(radar_grid['Y'])

    lon_tot=np.arange(lon[0],lon[len(lon)-1]+500, 500)
    lat_tot=lat[0:33]
    x, y = np.meshgrid(lon_tot, lat_tot)

    # faire un reshape de la precip
    precip_reshape= np.reshape(donnees, (len(lon_tot),len(lat_tot)) )
    precip_reshape=np.transpose(precip_reshape)

    # Aller chercher les donnees correspondantes pour les pluvios
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)

    emplacements_pluvios= pd.read_csv(emplacements_pluvios)

    valeurs_pluvios_date=emplacements_pluvios[['X', 'Y', 'SONDEID']]
    valeurs_pluvios_date_classe=valeurs_pluvios_date.sort_values(by='SONDEID')
    valeurs_pluvio = valeurs_pluvios_date_classe.set_index('SONDEID')

    data_pluvio = donnees_pluvios.loc[pd.Timestamp(date_heure)]
    data_pluvio = data_pluvio.sort_index()
    data_pluvio = data_pluvio.rename('precip')

    valeurs_pluvio['precip'] = data_pluvio   #Dataframe X, Y, precip de chaque station

    # Tracer la grille
    plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues')
    plt.colorbar(label="Pluie (mm)")
    plt.xlabel("X coordinate (m)")
    plt.ylabel("Y coordinate (m)")

    # Ajouter les pluviometres
    plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), cmap='Blues', edgecolor='black',s=80)

    plt.show()

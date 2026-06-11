#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des pluviometres de la
Ville de Sherbrooke.
- "ajoute_manquantes": Ajouter explicitement les donnees manquantes
- "single_raingauge" : Creer un fichier des precipitations completes propre a chaque pluviometre
- "metadata" : Creer le fichier metadata utiliser par le module rainfallQC

- "krig_ordinaire_pluvio" : Effectuer un krigeage ordinaire des donnees des 
    pluviometres sur une grille radar couvrant la region etudiee
- "krig_derive_pluvio" : Effectuer un krigeage avec derive externe des donnees des 
    pluviometres sur une grille radar couvrant la region etudiee où la derive est l'altitude
- "valid_krig_pluvios" : Valider les resultats du krigeage en calculant des erreurs absolues, 
    erreurs relatives, RMSE, MAE

- "figures_periode" : Tracer une carte radar pour un seul pas de temps
    Option : Tracer des figures comparant les obsevations et les valeurs estimees

- "format_pcswmm" : Formater les resulats pour les rendre compatibles avec PCSWMM

@author: Marie-Amelie Boucher, USherbrooke
"""
from pathlib import Path
import pandas as pd
import numpy as np
from pykrige import OrdinaryKriging
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist
from pykrige.uk import UniversalKriging
from itertools import combinations
from scipy.optimize import curve_fit

def ajoute_manquantes(fichier_o, fichier_modif, date_debut, date_fin, pasdetemps):
    """
    Parameters
    ----------
    fichier_o : chaine de caracteres
        Chemin vers le fichier .csv des donnees brutes des pluviometres
        Telecharger sur le site de la ville de Sherbrooke: https://pluviometres.ville.sherbrooke.qc.ca/
        *Le fichier provenant du site est .xls et doit etre converti en .csv prealablement
    fichier_modif : chaine de caracteres
        Chemin et nom du fichier de destination (AVEC les dates manquantes ajoutees) en format csv
    date_debut : chaine de caracteres
        Date de debut de la periode d'interet. Exemple : '2018-01-01'
    date_fin : chaine de caracteres
        Date de fin de la periode d'interet. Exemple : '2018-12-31'
    pasdetemps : chaine de caracteres
        Resolution temporelle des donnees
        Valeurs supportees : "5min" et "1h"

    Returns
    -------
    donnees_pluvio_complet: pandas.DataFrame
        Serie temporelle complete couvrant la periode demandee
        Les pas de temps manquant sont ajoutes et remplis avec des NaN
        Le DataFrame est enregiste en format .csv
    """
    df = pd.read_csv(fichier_o, sep=';')

    if str(df.iloc[-1, 0]).strip().upper() == "TOTAL":
        df = df.iloc[:-1]

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df = df.set_index('Date')

    if pasdetemps == "5min":
        df.index = pd.to_datetime(
            df.index.date.astype(str) + ' ' + df['Période'].str.split(' à ').str[0])
        df = df.drop(columns='Période')

    elif pasdetemps == "1h" :
        df['Heure'] = (df['Période'].str.extract(r'(\d{1,2})h')[0].str.zfill(2) + ':00:00')
        df['Date'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Heure'])
        df = df.set_index('Date')
        df = df.drop(columns=['Période', 'Heure'])

    else :
        raise ValueError(f"pasdetemps='{pasdetemps}' non supporté. "
            "Utiliser '5min' ou '1h'.")

    serie_index = pd.date_range(start= date_debut , end= date_fin, freq= pasdetemps)

    donnees_pluvio_complet = pd.DataFrame(index =serie_index, columns=df.columns)
    donnees_pluvio_complet.update(df)

    donnees_pluvio_complet.to_csv(fichier_modif, sep=',')

    return donnees_pluvio_complet


def single_raingauge_metadata(precip_complete, emplacements_pluvios, dossier_sortie, fichier_metadata):
    """
    Parameters
    ----------
    precip_complete : chaine de caracteres
        Chemin vers le fichier csv des precipitations completes de toutes les stations
        (precip_complete.csv)
    emplacements_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des emplacements des pluviometres
        (pluvio_xyz.csv)
    dossier_sortie : chaine de caracteres
        Chemin et nom du dossier de destination des fichiers individuels
    fichier_metadata : chaine de caracteres
        Chemin et nom du fichier de destination (.csv)

    Returns
    -------
    metadata_df : pandas.DataFrame
        Tableau metadata contenant : station_id, latitude, longitude, start_datetime, end_datetime et path
    """
    # Donnees de precipitations
    donnees_precip_complet = pd.read_csv(precip_complete,sep=';',index_col=0,parse_dates=True)

    # Emplacement des pluviometres
    pluvio_xy = pd.read_csv(emplacements_pluvios)
    pluvio_xy = pluvio_xy.set_index('SONDEID')
    pluvio_xy = pluvio_xy[['Latitude','Longitude']]
    pluvio_xy = pluvio_xy.apply(pd.to_numeric)
    pluvio_xy.index = pluvio_xy.index.astype(str)

    # Creer le dossier de sortie
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(exist_ok=True)

    metadata_list = []

    for station in donnees_precip_complet.columns:
        # Fichier individuel
        df_station = donnees_precip_complet[[station]]
        chemin_fichier = dossier_sortie / f"precip_complete_{station}.csv"
        df_station.to_csv(chemin_fichier, sep=';')

        # Metadata
        metadata_list.append({
            'station_id': station,
            'latitude': pluvio_xy.loc[str(station), 'Latitude'],
            'longitude': pluvio_xy.loc[str(station), 'Longitude'],
            'start_datetime': df_station.index.min(),
            'end_datetime': df_station.index.max(),
            'path': str(chemin_fichier.resolve())})

    metadata_df = pd.DataFrame(metadata_list,
        columns=['station_id','latitude','longitude','start_datetime','end_datetime','path'])

    metadata_df.to_csv(fichier_metadata, sep=';', index=False)

    return metadata_df


def interpolation_IDW_grid(radar_grid, emplacements_pluvios, donnees_pluvios, rayon, chemin_resultats):
    """
    Parameters
    ----------
    radar_grid : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des points de la grille radar 
        (radar_grid_xyz.csv)
    emplacements_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des emplacements des pluviometres 
        (pluvio_xyz.csv)
    donnees_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des donnees completes des precipitations 
        (precip_complete.csv)
    rayon : Int
        Valeur du rayon d'influence en metre
    chemin_resultats : chaine de caracteres
        Chemin vers le dictionnaire de dataframe, soit "resultats" qui est retourne 
        par la fonction. Enregirster sous le format .pkl

    Returns
    -------
    donnees_pluvios : DataFrame
        DataFrame des donnees des pluviometres : une rangee par pas de temps et 
        une colonne par pluviometre
    resultats : Dictionnaire
        Dictionnaire contenant un Dataframe pour chaque pas de temps. 
        Chaque DataFrame contient 3 colonnes : "X", "Y" et "precip".
        Les precipitations sont les valeurs resultant de l'interpolation IDW
    """
    # Grille radar - centroides xyz (500x500m)
    radar_grid = pd.read_csv(radar_grid)
    radar_grid = radar_grid.set_index('id')
    radar_grid = radar_grid[['X','Y','ELEV_1']]
    radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6
    
    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])
    
    # Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6
    
    xpluvio = np.array(pluvio_xyz['X'])
    ypluvio = np.array(pluvio_xyz['Y'])
    
    # Donnees pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric) 
    donnees_pluvios = donnees_pluvios[pluvio_xyz.index] #Changer l'ordre des colonnes
    
    #Rassembler les donnees des pluviometres aux pluviometres
    temps = donnees_pluvios.index
    df = pd.DataFrame(index=temps)
    
    for station in pluvio_xyz.index :
        df[f'X_{station}'] = pluvio_xyz.loc[station, 'X']
        df[f'Y_{station}'] = pluvio_xyz.loc[station, 'Y']
        df[f'precip_{station}'] = donnees_pluvios[station]
    
    #interpolation
    resultats = {}
    for t in temps :
        precip = donnees_pluvios.loc[t].to_numpy()        
        interpol = inverse_distance_to_grid(xpluvio, ypluvio, precip, gx, gy, r=rayon, min_neighbors=3)        
        df_t = pd.DataFrame({'X':gx, 'Y':gy, 'precip':interpol})       
        resultats[t] = df_t
    
    #Save les resultats
    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return donnees_pluvios, resultats


def interpolation_IDW_points(radar_grid, emplacements_pluvios, donnees_pluvios, rayon, chemin_resultats):
    """
    Parameters
    ----------
    radar_grid : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des points de la grille radar 
        (radar_grid_xyz.csv)
    emplacements_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des emplacements des pluviometres 
        (pluvio_xyz.csv)
    donnees_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des donnees completes des precipitations 
        (precip_complete.csv)
    rayon : Int
        Valeur du rayon d'influence en metre
    chemin_resultats : chaine de caracteres
        Chemin vers le dictionnaire de dataframe, soit "resultats" qui est retourne 
        par la fonction. Enregirster sous le format .pkl

    Returns
    -------
    donnees_pluvios : DataFrame
        DataFrame des donnees des pluviometres : une rangee par pas de temps et 
        une colonne par pluviometre
    resultats : Dictionnaire
        Dictionnaire contenant un Dataframe pour chaque pas de temps. 
        Chaque DataFrame contient 3 colonnes : "X", "Y" et "precip".
        Les precipitations sont les valeurs resultant de l'interpolation IDW
    """
    # Grille radar - centroides xyz (500x500m)
    radar_grid = pd.read_csv(radar_grid)
    radar_grid = radar_grid.set_index('id')
    radar_grid = radar_grid[['X','Y','ELEV_1']]
    radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6
    
    xi = (radar_grid[['X', 'Y']].copy()).to_numpy()
    
    # Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6
    
    points = (pluvio_xyz[['X', 'Y']].copy()).to_numpy()
    
    # Donnees pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric) 
    donnees_pluvios = donnees_pluvios[pluvio_xyz.index]
    
    #interpolation
    resultats = {}
    for t in donnees_pluvios.index :
        precip = donnees_pluvios.loc[t].to_numpy()   
        interpol = inverse_distance_to_points(points, precip, xi, rayon)
        
        resultats[t] = pd.DataFrame({'X': xi[:, 0],
        'Y': xi[:, 1], 'precip': interpol})

    #Save les resultats
    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return donnees_pluvios, resultats


def krig_ordinaire_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios, chemin_resultats):
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
    chemin_resultats : Chaine de caracteres
        Chemin vers le dictionnaire de dataframe, soit "resultats" qui est retourne par la fonction

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
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6

    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])

    # Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    # Donnees pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric) 

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
                                   nlags=8, enable_plotting=False, verbose=False,
                                   enable_statistics=False, coordinates_type='euclidean',
                                   pseudo_inv=True, weight=False)

            estim, var = krig.execute("points", gx, gy)

            result_t["estimation"] = estim
            result_t["variance"] = var

        else : pass
        resultats[t] = result_t

    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return resultats, donnees_pluvios


def krig_derive_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios, chemin_resultats):
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
    chemin_resultats : Chaine de caracteres
        Chemin vers le dictionnaire de dataframe, soit "resultats" qui est retourne par la fonction

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
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6

    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])
    gz = np.array(radar_grid['Z'])
    
    # Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6
    
    # Donnees pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric) 
    
    #Rassembler les donnees des pluviometres aux pluviometres
    stations = donnees_pluvios.columns.values
    temps = donnees_pluvios.index
    df = pd.DataFrame(index=temps)

    for station in pluvio_xyz.index :
        df[f'X_{station}'] = pluvio_xyz.loc[station, 'X']
        df[f'Y_{station}'] = pluvio_xyz.loc[station, 'Y']
        df[f'Z_{station}'] = pluvio_xyz.loc[station, 'Z']
        df[f'precip_{station}'] = donnees_pluvios[station]

    #Krigeage
    resultats = {}
    for t in temps :
        ligne = df.loc[t]       #Info des pluviometres pour 1 pas de temps

        x_val = np.array([ligne[f"X_{st}"]     for st in stations])        #Coordonnees X des pluviometres
        y_val = np.array([ligne[f"Y_{st}"]     for st in stations])        #Coordonnees Y des pluviometres
        z_val = np.array([ligne[f"Z_{st}"]     for st in stations])        #Coordonnees Z des pluviometres
        precip = np.array([ligne[f"precip_{st}"] for st in stations])      #Precip aux pluviometres

        #DataFrame du pas de temps
        result_t = pd.DataFrame({"x":gx, "y":gy, "z":gz,"estimation":np.nan, "variance":np.nan},
                                index= radar_grid.index)

        if len(precip) > 0 and not np.all(np.isnan(precip)):

            uk = UniversalKriging(x_val, y_val, precip,
                variogram_model='spherical', nlags=8,

                drift_terms=['specified'],
                specified_drift = [z_val],

                verbose=False, enable_plotting=False, pseudo_inv=True)

            estim, var = uk.execute("points", gx, gy, specified_drift_arrays=[gz])

            result_t["estimation"] = estim
            result_t["variance"] = var
        
        resultats[t] = result_t
         
    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)
    
    return resultats, donnees_pluvios


def valid_krig_pluvios(data_krig, radar_grid, emplacements_pluvios, donnees_pluvios, date_debut, date_fin):
    """
    Parameters
    ----------
    data_krig : Chaine de caracteres
        Chemin vers le fichier .pkl contenant les resultats du krigeage
        Le dictionnaire doit etre structure par timestamps
    radar_grid : Chaine de caracteres
        Chemin vers le fichier .csv contenant la grille radar (coordonnees X, Y et altitude)
        (*radar_grid_xyz.csv)
    emplacements_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees 'x' et 'y', en metre, des pluviometres
        (*pluvio_xyz.csv)
    donnees_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les observations de précipitations 
        aux pluviometres (*precip_complete.csv)
    date_debut : chaine de caracteres
        Date du debut de la periode, avec un format comme cet exemple : "2025-05-17 13:30:00"
    date_fin : chaine de caracteres
        Date de la fin de la periode, avec un format comme cet exemple : "2025-05-17 13:30:00"

    Returns
    -------
    rmse_global : float
        RMSE global calcule sur toutes les stations et tous les pas de temps
    rmse_glob_station : pandas.DataFrame
        RMSE calcule individuellemenet pour chaque station
    mae_global : float
        MAE global sur l'ensemble des stations'
    mae_glob_station : pandas.DataFrame
        MAE calcule individuellemenet pour chaque station
    erreur_abs_serie : pandas.DataFrame
        Serie temporelle des erreurs absolues par station (en mm)
    erreur_rel_serie : pandas.DataFrame
        Serie temporelle des erreurs relatives (en %)
    flags_serie_relatif : pandas.DataFrame
        Tableau de classification des erreurs relatives :
            0 = bonne performance (≤ 20 %) ; 1 = erreur modérée (20–40 %)
            2 = erreur élevée (40–60 %) ; 3 = très mauvaise performance (> 60 %)
            4 = faible pluie (cas où l’interprétation relative est instable)
    """
    #Lire le dictionnaire des donnees calculees
    with open(data_krig, "rb") as f:
        data = pickle.load(f)

    #Periode
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)
    periode = [t for t in data.keys()
        if date_debut <= pd.Timestamp(t) <= date_fin]
    periode = sorted(periode)

    #Grille radar - centroides xyz (500x500m)
    radar_grid = pd.read_csv(radar_grid)
    radar_grid = radar_grid.set_index('id')
    radar_grid = radar_grid[['X','Y','ELEV_1']]
    radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6

    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])
    coords_grille = np.column_stack((gx, gy))

    #Emplacements des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric)
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    xpluvio = np.array(pluvio_xyz['X'])
    ypluvio = np.array(pluvio_xyz['Y'])
    coords_stations = np.column_stack((xpluvio, ypluvio))

    #Donnees des pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric)
    donnees_pluvios = donnees_pluvios[pluvio_xyz.index]

    #Distances
    dist = cdist(coords_stations, coords_grille)
    idx = np.argmin(dist, axis=1)

    #Dictionnaire
    erreurs_stations = {station: []
        for station in pluvio_xyz.index}

    erreur_abs_serie = []
    erreur_rel_serie = []
    flags_serie_relatif = []
    erreurs_glob = []

    #Boucle
    for t in periode :
        donnees_estim = data[t]

        estim_grille = donnees_estim['estimation'].values
        estim_station = estim_grille[idx]

        obs_station = (donnees_pluvios.loc[pd.Timestamp(t),
                    pluvio_xyz.index].values)

        ligne_temps = {'Temps': pd.Timestamp(t)}
        ligne_temps_rel = {'Temps': pd.Timestamp(t)}
        ligne_temps_flag = {'Temps': pd.Timestamp(t)}

        for i, station in enumerate(pluvio_xyz.index):
            obs = obs_station[i]
            estim = estim_station[i]

            if not np.isnan(obs) and not np.isnan(estim):
                erreur = obs - estim
                erreurs_stations[station].append(erreur)
                erreurs_glob.append(erreur)
                
                #Erreur absolue (mm)
                erreur_abs_inst = np.abs(erreur)
                ligne_temps[station] = erreur_abs_inst
                
                #Erreur relative (%)
                if obs > 0:
                    erreur_rel_inst = np.abs(erreur) / obs * 100
                else:
                    erreur_rel_inst = np.nan
                
                ligne_temps_rel[station] = erreur_rel_inst
                
                #Flags erreur relative
                if np.isnan(obs) or np.isnan(estim):
                    flag = np.nan
                elif obs == 0 :
                    flag = np.nan
                elif obs < 0.2: #Faible pluie
                    flag = 4
                elif np.isnan(erreur_rel_inst):
                    flag = np.nan
                elif erreur_rel_inst <= 20:
                    flag = 0
                elif erreur_rel_inst <= 40:
                    flag = 1
                elif erreur_rel_inst <= 60:
                    flag = 2
                else:
                    flag = 3
                
                ligne_temps_flag[station] = flag

            else:
                ligne_temps[station] = np.nan
                ligne_temps_rel[station] = np.nan
                ligne_temps_flag[station] = np.nan

        erreur_abs_serie.append(ligne_temps)
        erreur_rel_serie.append(ligne_temps_rel)
        flags_serie_relatif.append(ligne_temps_flag)

    # RMSE et MAE
    rmse_glob_station = {}
    mae_glob_station = {}

    for station, erreurs in erreurs_stations.items():
        erreurs = np.array(erreurs)

        if len(erreurs) == 0:
            rmse_glob_station[station] = np.nan
            mae_glob_station[station] = np.nan
        else:
            rmse_glob_station[station] = np.sqrt(np.mean(erreurs**2))
            mae_glob_station[station] = np.mean(np.abs(erreurs))

    rmse_glob_station = pd.DataFrame([rmse_glob_station])
    mae_glob_station = pd.DataFrame([mae_glob_station])

    erreurs_globales = np.array(erreurs_glob)
    
    rmse_global = np.sqrt(np.mean(erreurs_globales**2)) if len(erreurs_globales) > 0 else np.nan
    mae_global = np.mean(np.abs(erreurs_globales)) if len(erreurs_globales) > 0 else np.nan

    #Mettre en DataFrame
    erreur_abs_serie = pd.DataFrame(erreur_abs_serie)
    erreur_abs_serie = erreur_abs_serie.set_index('Temps')

    erreur_rel_serie = pd.DataFrame(erreur_rel_serie)
    erreur_rel_serie = erreur_rel_serie.set_index('Temps')

    flags_serie_relatif = pd.DataFrame(flags_serie_relatif)
    flags_serie_relatif = flags_serie_relatif.set_index('Temps')

    # FIGURE - boxplot
    plt.figure(figsize=(14,6))
    erreur_abs_serie.boxplot()
    plt.ylabel('Erreur absolue (mm)')
    plt.xlabel('Pluviomètre')
    plt.title('Distribution des erreurs absolues par pluviomètre')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return (rmse_global, rmse_glob_station, mae_global, mae_glob_station, erreur_abs_serie,
    erreur_rel_serie, flags_serie_relatif)


def figures_periode(data_calcul, radar_grid, emplacements_pluvios, donnees_pluvios, date_debut, date_fin, comparaison):
    """
    Parameters
    ----------
    data_calcul : Chaine de caracteres
        Chemin vers le fichier .pkl contenant le dictionnaire des resultats des calculs
        (interpolation IDW ou krigeage ordinaire)
    radar_grid : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees des cellules ou les precip sont krigees.
        C'est une grille 500 x 500 m (*radar_grid_xyz.csv)
    emplacements_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les coordonnees 'x' et 'y', en metre, des pluviometres
        (*pluvio_xyz.csv)
    donnees_pluvios : Chaine de caracteres
        Chemin vers le fichier .csv contenant les donnees de tous les pluviometres et pour tous
        les pas de temps de la periode (*precip_complete.csv)
    date_debut : chaine de caracteres
        Date du debut de la periode, avec un format comme cet exemple : "2025-05-17 13:30:00"
    date_fin : chaine de caracteres
        Date de la fin de la periode, avec un format comme cet exemple : "2025-05-17 13:30:00"
    comparaison : int
        Figure comparant la quantite de pluie observee et estimee
        0 : non ; 1 : oui
    
    Returns
    -------
    None
    """
    #Lire le dictionnaire des donnees calculees
    with open(data_calcul, "rb") as f:
        data = pickle.load(f)
    #donnees = data[pd.Timestamp(date_heure)]
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)
    
    periode = [t for t in data.keys()
        if date_debut <= pd.Timestamp(t) <= date_fin]
    periode = sorted(periode)
    
    #Grille radar - centroides xyz (500x500m)
    radar_grid = pd.read_csv(radar_grid)
    radar_grid = radar_grid.set_index('id')
    radar_grid = radar_grid[['X','Y','ELEV_1']]
    radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
    radar_grid = radar_grid.apply(pd.to_numeric)
    radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6
    
    gx = np.array(radar_grid['X'])
    gy = np.array(radar_grid['Y'])
    coords_grille = np.column_stack((gx, gy))
    
    #Emplacements des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6
    
    xpluvio = np.array(pluvio_xyz['X'])
    ypluvio = np.array(pluvio_xyz['Y'])
    coords_stations = np.column_stack((xpluvio, ypluvio))
    
    #Donnees des pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)
    donnees_pluvios = donnees_pluvios.apply(pd.to_numeric) 
    donnees_pluvios = donnees_pluvios[pluvio_xyz.index]
    
    for date_heure in periode:
        print(f"Traitement : {date_heure}")
        donnees = data[pd.Timestamp(date_heure)]

        pivot = donnees.pivot(index='y',columns='x',values='estimation')
        x, y = np.meshgrid(pivot.columns.values,pivot.index.values)
        precip_reshape = pivot.values
        
        data_pluvio = donnees_pluvios.loc[pd.Timestamp(date_heure)]
        data_pluvio = data_pluvio.sort_index()
        data_pluvio = data_pluvio.rename('precip')
    
        valeurs_pluvio = pluvio_xyz.copy()
        valeurs_pluvio['precip'] = data_pluvio   
    
        #Bornes des figures
        min_obs=min(valeurs_pluvio['precip'])
        max_obs=max(valeurs_pluvio['precip'])
        
        min_grille= np.nanmin(precip_reshape)
        max_grille= np.nanmax(precip_reshape)
                
        min_global=min([min_obs,min_grille])
        max_global=max([max_obs,max_grille])
        
        norm = plt.Normalize(min_global, max_global)
     
        #FIGURE : Carte radar - krigeage avec derive externe                
        fig, ax = plt.subplots(figsize=(8,6))
        plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
        plt.colorbar(label="Pluie (mm)")
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.title(f"Krigeage avec dérive externe - {date_heure}")
        
        plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), 
                    cmap='Blues', edgecolor='black',s=80, norm=norm)
        
        #FIGURE : Comparaison observation vs calculs
        if comparaison == 1 :
            dist = cdist(coords_stations, coords_grille) #Distance entre station et grille
            idx = np.argmin(dist, axis=1) #Indice du point grille le plus proche
            
            estim_result = donnees['estimation'].values #Valeurs interpolées aux stations
            estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')
            
            df_comparaison = pd.concat([data_pluvio.rename('obs'), estim_station],axis=1)
            df_comparaison['diff'] = np.abs(df_comparaison['obs'] - df_comparaison['estim'])
            
            df_comparaison.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
            plt.title("Comparaison des données observées et estimées pour tous les pluviomètres")
            plt.ylabel("Précipitation (mm)")
            plt.tight_layout()
            plt.show()
            
            #Les stations autour de la zone etudiee
            liste_stations = ['SPQ', 'CJC', 'ASC', 'SPH', 'JMJ']
            comparaison_cut = df_comparaison.loc[df_comparaison.index.intersection(liste_stations)]
            comparaison_cut.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
            plt.title("Comparaison des données observées et estimées pour les pluviomètres autour de la zone étudiée")
            plt.ylabel("Précipitation (mm)")
            plt.tight_layout()
            plt.show()
        
        elif comparaison == 0 :
            plt.show()
        else :
            raise ValueError("Le paramètre 'comparaison' doit être 0 ou 1")
    
    return


def format_pcswmm(index_garde, chemin_resultats, chemin_timeseries):
    """
    Parameters
    ----------
    index_garde : list
        Liste des ID des cases de la radar_grid que l'on veut garder
        *Facilement identifiable sur PCSWMM avec le shapefile de la grille
    chemin_resultats : chaine de caracteres
        Chemin vers le fichier .pkl, un dictionnaire contenant des dataframes :
        Un df par pas de temps, les ID des cases de la grille radar en index
        les colonnes sont "x", "y", "estimation" et "variance"
    chemin_timeseries : chaine de caracteres
        Chemin ou le fichier .csv sera enregistre. PCSWMM prend ce format de fichier.

    Returns
    -------
    timeseries_estim : DataFrame
        index : serie de pas de temps
        colonnes : les ID des cases couvrants la region etudiee
        valeurs : precipitation en mm
    """
    # Ouvrir le dictionnaire avec les dataframes de chaque pas de temps
    with open(chemin_resultats, "rb") as f:
        resultats = pickle.load(f)
    
    # Couper la grille pour garder seulement la region etudiee
    for key, df in resultats.items():
        idx = sorted(i for i in index_garde if i in df.index)
        resultats[key] = df.loc[idx]
    
    timeseries = pd.concat(resultats, names=["Temps", "points"])

    timeseries_estim = timeseries["estimation"].unstack("points")
    timeseries_estim = timeseries_estim.fillna(0)                      #Remplacer les nan par 0 pour PCSWMM
    timeseries_estim = timeseries_estim.clip(lower=0)                  #Mettre les valeurs negatives a zero
    timeseries_estim.index = pd.to_datetime(timeseries_estim.index)

    timeseries_estim.to_csv(chemin_timeseries)

    return timeseries_estim
    

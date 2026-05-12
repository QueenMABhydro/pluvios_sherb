#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des pluviometres de la
Ville de Sherbrooke.
- "ajoute_manquantes": Ajouter explicitement les donnees manquantes
- "filtre_precip_louche": Filtrer les valeurs qui semblent aberrantes

- "interpolation_IDW_grid" : Effectuer une interpolation par pondération inverse
    de la distance (IDW) des données des pluviometres sur une grille couvrant la 
    region etudiee avec "metpy.interpolate.inverse_distance_to_grid"
- "interpolation_IDW_point" : Effectuer une interpolation par pondération inverse
    de la distance (IDW) des données des pluviometres sur une grille couvrant la 
    region etudiee avec "metpy.interpolate.inverse_distance_to_points"
- "krig_ordinaire_pluvio" : Effectuer un krigeage ordinaire des donnees des 
    pluviometres sur une grille radar couvrant la region etudiee

- "format_pcswmm" : Formater les resulats pour les rendre compatibles avec PCSWMM

@author: Marie-Amelie Boucher, USherbrooke
"""

import pandas as pd
import numpy as np
from pykrige import OrdinaryKriging
import matplotlib.pyplot as plt
import pickle
from metpy.interpolate import inverse_distance_to_grid

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
    Notes
    ----------
    Le variogramme n'est pas bon (https://github.com/GeoStat-Framework/PyKrige/discussions/204)
    il faut revoir les parametres
    
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


def figures_pkl(data_calcul, radar_grid, emplacements_pluvios, donnees_pluvios, date_heure, methode_calcul):
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
    date_heure : chaine de caracteres
        Pas de temps observe, avec un format comme cet exemple : "2025-05-17 13:30:00"
    methode_calcul : Int
        1 : Si la methode de calcul est "interpolation IDW"
        2 : Si la methode de calcul est "krigeage ordinaire"

    Returns
    -------
    None
    """
    #Lire le dictionnaire des donnees calculees
    with open(data_calcul, "rb") as f:
        data = pickle.load(f)
    donnees = data[pd.Timestamp(date_heure)]
    
    if methode_calcul == 1 :
        pivot = donnees.pivot(index='Y', columns ='X', values='precip')   
        x, y = np.meshgrid(pivot.columns.values, pivot.index.values)
        precip_reshape = pivot.values
    elif methode_calcul == 2 :
        pivot = donnees.pivot(index='y', columns ='x', values='estimation')   
        x, y = np.meshgrid(pivot.columns.values, pivot.index.values)
        precip_reshape = pivot.values
    else :
        raise ValueError("Le paramètre 'methode_calcul' doit être 1 ou 2")

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
    
    #FIGURE : Interpolation IDW
    if methode_calcul == 1 :
        fig, ax = plt.subplots(figsize=(8,6))
        plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
        plt.colorbar(label="Pluie (mm)")
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.title(f"Interpolation IDW - {date_heure}")
        
        plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'],c=valeurs_pluvio['precip'].astype(float), 
                    cmap='Blues', edgecolor='black',s=80, norm=norm)
        
        plt.show()
    
    #FIGURE : Krigeage ordinaire
    elif methode_calcul == 2 :                
        fig, ax = plt.subplots(figsize=(8,6))
        plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
        plt.colorbar(label="Pluie (mm)")
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.title(f"Krigeage ordinaire - {date_heure}")
        
        plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), 
                    cmap='Blues', edgecolor='black',s=80, norm=norm)
    
    else :
        raise ValueError("Le paramètre 'figures' doit être 1 ou 2")
    
    
    #FIGURE : Comparaison observation vs calculs
    dist = cdist(coords_stations, coords_grille) #Distance entre station et grille
    idx = np.argmin(dist, axis=1) #Indice du point grille le plus proche
    
    if methode_calcul == 1 :
        estim_result = donnees['precip'].values #Valeurs interpolées aux stations
        estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')
    elif methode_calcul == 2 :
        estim_result = donnees['estimation'].values #Valeurs interpolées aux stations
        estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')
    
    comparaison = pd.concat([data_pluvio.rename('obs'), estim_station],axis=1)
    comparaison['diff'] = np.abs(comparaison['obs'] - comparaison['estim'])
    
    comparaison.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
    plt.title("Comparaison des donnees observees et interpolees pour tous les pluviometres")
    plt.ylabel("Précipitation (mm)")
    plt.tight_layout()
    plt.show()
    
    #Les stations autour de la zone etudiee
    liste_stations = ['SPQ', 'CJC', 'ASC', 'SPH', 'JMJ']
    comparaison_cut = comparaison.loc[comparaison.index.intersection(liste_stations)]
    comparaison_cut.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
    plt.title("Comparaison des donnees observees et interpolees pour les pluviometres autour de la zone etudiee")
    plt.ylabel("Précipitation (mm)")
    plt.tight_layout()
    plt.show()
        
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
    

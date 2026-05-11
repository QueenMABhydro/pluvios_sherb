#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des precipitometres de la
Ville de Sherbrooke.
- "ajoute_manquantes": Ajouter explicitement les donnees manquantes
- "filtre_precip_louche": filtrer les valeurs qui semblent aberrantes
    (avant de coder cette fonction, verifier si le code existe deja)
- "interpolation_IDW_pluvio" : Interpolation par pondération inverse à la distance (IDW) 
    des donnees des pluviometres sur une grille couvrant la region etudiee
- "visualiser_grille_IDW_pkl" : Figure illustrant les donnees interpolees (IDW) sous forme 
    de carte avec l'option d'une figure comparant les interpolations et les observations
- "krig_pluvio" : Krigeage ordinaire des donnees des pluviometres sur une grille "radar"
    couvrant la region etudiee
- "visualiser_grille_csv" : Figure illustrant les donnees krigees d'un pas de temps
    a partir d'un fichier csv
- "visualiser_grille_pkl" : Figure illustrant les donnees krigees d'un pas de temps
    a partir d'un fichier pkl
- "format_pcswmm" : Formatage des resulats pour les rendre compatible avec PCSWMM

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


def interpolation_IDW_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios, rayon, chemin_resultats):
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


def visualiser_grille_IDW_pkl(grille_IDW, radar_grid, donnees_pluvios, emplacements_pluvios, 
                              date_heure, comparaison):
    """
    Parameters
    ----------
    grille_IDW : chaine de caracteres
        Chemin vers le dictionnaire contenant les resultats de l'interpolation IDW 
        (dict_df.pkl)
    radar_grid : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des points de la grille radar 
        (radar_grid_xyz.csv) 
    donnees_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des donnees completes des precipitations 
        (precip_complete.csv)
    emplacements_pluvios : chaine de caracteres
        Chemin vers le fichier .csv des coordonnees des emplacements des pluviometres 
        (pluvio_xyz.csv)
    date_heure : chaine de caracteres
        Pas de temps observe, avec un format comme cet exemple : "2025-05-17 13:30:00"
    comparaison : chaine de caracteres
        Pour des figures comparant les observations aux pluviometres et les valeurs 
        d'interpolation au point le plus pres des pluviometres, inscrire : "oui"
        figure 1 : tous les pluviometres
        figure 2 : les pluviometres pres de la zone etudiee 

    Returns
    -------
    None
    """
    #Lire le dictionnaire des pas de temps
    with open(grille_IDW, "rb") as f:
        data = pickle.load(f)
    
    #temps = list(resultats.keys())[161] #162 il y a de la pluie
    donnees = data[pd.Timestamp(date_heure)]
    
    pivot = donnees.pivot(index='Y', columns ='X', values='precip')   
    x, y = np.meshgrid(pivot.columns.values, pivot.index.values)
    precip_reshape = pivot.values
    
    #Coordonnees xyz des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios)
    pluvio_xyz = pluvio_xyz.set_index('SONDEID')
    pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
    pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
    pluvio_xyz = pluvio_xyz.apply(pd.to_numeric) 
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6
    
    #Donnes des pluviometres    
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
    
    # Tracer la grille
    min_obs=min(valeurs_pluvio['precip'])
    max_obs=max(valeurs_pluvio['precip'])
    min_grille= np.nanmin(precip_reshape)
    max_grille= np.nanmax(precip_reshape)
    
    min_global=min([min_obs,min_grille])
    max_global=max([max_obs,max_grille])
    
    norm = plt.Normalize(min_global, max_global)
    
    plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
    plt.colorbar(label="Pluie (mm)")
    plt.xlabel("Longitude (m)")
    plt.ylabel("Latitude (m)")
    plt.title(date_heure)
    
    plt.scatter( valeurs_pluvio['X'], valeurs_pluvio['Y'],
        facecolors='none', edgecolors='black', s=25, linewidths=1)
    
    plt.show()

    if comparaison == "oui" :
        donnees_pluvios2 = donnees_pluvios.loc[pd.Timestamp(date_heure)]
        
        # Grille radar - centroides xyz (500x500m)
        radar_grid = pd.read_csv(radar_grid)
        radar_grid = radar_grid.set_index('id')
        radar_grid = radar_grid[['X','Y','ELEV_1']]
        radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
        radar_grid = radar_grid.apply(pd.to_numeric)
        radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6
        
        gx = np.array(radar_grid['X'])
        gy = np.array(radar_grid['Y'])
        coords_grille = np.column_stack((gx, gy))
        
        xpluvio = np.array(pluvio_xyz['X'])
        ypluvio = np.array(pluvio_xyz['Y'])
        coords_stations = np.column_stack((xpluvio, ypluvio))
        
        #Comparaison
        dist = cdist(coords_stations, coords_grille) #Distance entre station et grille
        idx = np.argmin(dist, axis=1) #Indice du point grille le plus proche
        interp_result = donnees['precip'].values #Valeurs interpolées aux stations
        interp_station = interp_result[idx]
        
        comparaison = pd.DataFrame({'obs': donnees_pluvios2.values, 'interp': interp_station}, index=donnees_pluvios2.index)
        comparaison['diff'] = np.abs(comparaison['obs'] - comparaison['interp'])
        comparaison.plot(y=['obs', 'interp'], kind='bar', figsize=(10,4))
        plt.title("Comparaison des donnees observees et interpolees pour tous les pluviometres")
        
        #Les stations autour de la zone etudiee
        liste_stations = ['SPQ', 'CJC', 'ASC', 'SPH', 'JMJ']
        comparaison_cut = comparaison.loc[liste_stations]
        comparaison_cut.plot(y=['obs', 'interp'], kind='bar', figsize=(10,4))
        plt.title("Comparaison des donnees observees et interpolees pour les pluviometres autour de la zone etudiee")
        
    else :
        print ("aucune figure de comparaison")
    
    return


def krig_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios, chemin_resultats):
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
    min_obs=min(valeurs_pluvio['precip'])
    max_obs=max(valeurs_pluvio['precip'])
    min_grille= np.nanmin(precip_reshape)
    max_grille= np.nanmax(precip_reshape)
    
    min_global=min([min_obs,min_grille])
    max_global=max([max_obs,max_grille])
    
    norm = plt.Normalize(min_global, max_global)
    
    plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
    plt.colorbar(label="Pluie (mm)")
    plt.xlabel("Longitude (m)")
    plt.ylabel("Latitude (m)")
    plt.title(date_heure)

    # Ajouter les pluviometres
    plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), 
                cmap='Blues', edgecolor='black',s=80, norm=norm)

    plt.show()


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


def visualiser_grilles_pkl(grille_krigee, radar_grid, donnees_pluvios, emplacements_pluvios, figures_dump):
    """
    Notes
    ----------
    Possibilite de modifier pour permettre de tracer toutes les figures entre 2 pas de temps
    
    Parameters
    ----------
    grille_krigee : Chaine de caracteres
        Chemin vers le fichier .pkl contenant le dictionnaire de donnees krigees.
        Chaque dataframme du dictionnaire contient 5 colonnes, soient une colonne d'index sans titre
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
    figure_dump : Chaine de caracteres
        Chemin ou on veut enregirstrer le dictionnaire des figures

    Returns
    -------
    figures : Dictionnaire
        Dictionnaire des figures de toute la periode

    """
    #Lire le dictionnaire des pas de temps
    with open(grille_krigee, "rb") as f:
        donnees = pickle.load(f)
    #Lire grille radar
    radar_grid = pd.read_csv(radar_grid)
    
    #Coordonnees pour tous les pas de temps
    lon = np.array(radar_grid['X'])
    lat = np.array(radar_grid['Y'])
    lon_tot=np.arange(lon[0],lon[len(lon)-1]+500, 500)
    lat_tot=lat[0:33]
    x, y = np.meshgrid(lon_tot, lat_tot)
    
    #Donnees des pluviometres
    donnees_pluvios = pd.read_csv(donnees_pluvios)
    donnees_pluvios = donnees_pluvios.set_index('Unnamed: 0')
    donnees_pluvios = donnees_pluvios.rename_axis('Temps')
    donnees_pluvios.index = pd.to_datetime(donnees_pluvios.index)

    emplacements_pluvios= pd.read_csv(emplacements_pluvios)
    valeurs_pluvio = emplacements_pluvios[['X', 'Y', 'SONDEID']].copy()
    valeurs_pluvio = valeurs_pluvio.sort_values(by='SONDEID').set_index('SONDEID')

    figures = {}     #Dictionnaire de figure

    import warnings #Pour reduire les messages qui sont normaux
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    for t, df in donnees.items():
        estim = df['estimation'].to_numpy()
        precip_reshape= np.reshape(estim, (len(lon_tot),len(lat_tot)) )
        precip_reshape=np.transpose(precip_reshape)
        
        if t in donnees_pluvios.index:
            data_pluvio = donnees_pluvios.loc[t].sort_index().rename("precip")
            valeurs_pluvio["precip"] = data_pluvio
        else:
            valeurs_pluvio["precip"] = np.nan

        # Tracer la grille
        min_obs=min(valeurs_pluvio['precip'])
        max_obs=max(valeurs_pluvio['precip'])
        min_grille= np.nanmin(precip_reshape)
        max_grille= np.nanmax(precip_reshape)
        
        min_global=min([min_obs,min_grille])
        max_global=max([max_obs,max_grille])
        norm = plt.Normalize(min_global, max_global)
        
        fig, ax = plt.subplots()
        plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
        plt.colorbar(label="Pluie (mm)")
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.title(t)
        # Ajouter les pluviometres
        plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), 
                    cmap='Blues', edgecolor='black',s=80, norm=norm)
    
        figures[t] = fig
        
        with open(figures_dump, "wb") as f:     #Enregistrer le dict en .pkl
            pickle.dump(figures, f)

    return figures

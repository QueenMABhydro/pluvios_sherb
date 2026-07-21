# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:43:00 2026

Fonctions pour traiter les donnees brutes des pluviometres de la
Ville de Sherbrooke - Vieille fonctions

- "krig_ordinaire" : Effectuer un krigeage ordinaire
- "krig_derive_altitude" : Effectuer un krigeage avec l'altitude comme derive externe

- "interpolation_IDW_grid" : Effectuer une interpolation par pondération inverse
    de la distance (IDW) des données des pluviometres sur une grille couvrant la 
    region etudiee avec "metpy.interpolate.inverse_distance_to_grid"
- "interpolation_IDW_point" : Effectuer une interpolation par pondération inverse
    de la distance (IDW) des données des pluviometres sur une grille couvrant la 
    region etudiee avec "metpy.interpolate.inverse_distance_to_points"
- "krig_derive_fixe_pluvio" : Effectuer un krigeage avec dérive externe et un variogramme FIXE

- "valid_krig" : Calcul de RMSE pour valider le krigeage

- "visualiser_grille_IDW_pkl" : Figure illustrant les donnees interpolees (IDW) sous forme 
    de carte avec l'option d'une figure comparant les interpolations et les observations
- "visualiser_grille_csv" : Figure illustrant les donnees krigees d'un pas de temps
    a partir d'un fichier csv
- "visualiser_grille_pkl" : Figure illustrant les donnees krigees d'un pas de temps
    a partir d'un fichier pkl
- "figures_pkl" : Pour tracer la carte radar et des graphiques de comparaison 
    selon la methode de calcul (interpolation IDW, krigeage)

@author: Justine Hamelin
"""
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from pykrige import OrdinaryKriging, UniversalKriging
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import contextily as ctx

def krig_ordinaire(grille_interp, emplacements_pluvios, donnees_pluvios, chemin_resultats):
    """
    Parameters
    ----------
    grille_interp : str
        Chemin vers le fichier CSV des coordonnees de la grille d'interpolation
        (*radar_grid_xyz.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    chemin_resultats : str
        Chemin du fichier PKL ou sera enregistre le dictionnaire de resultats

    Returns
    -------
    resultats : Dictionnaire
        Dictionnaire contenant un DataFrame par pas de temps.
        Chaque df a contient les colonnes x, y, estimation et variance
    """
    #Grille d'interpolation (500x500m)
    grille_interp = (pd.read_csv(grille_interp)
                     .set_index("id")[["X", "Y"]].apply(pd.to_numeric))
    grille_interp[['X','Y']] = np.floor(grille_interp[['X','Y']]*10**6)/10**6

    gx = np.array(grille_interp['X'])
    gy = np.array(grille_interp['Y'])

    #Coordonnees des pluviometres
    pluvio_xy = (pd.read_csv(emplacements_pluvios)
                 .set_index("SONDEID")[["X", "Y"]].apply(pd.to_numeric))
    pluvio_xy[['X','Y']] = np.floor(pluvio_xy[['X','Y']]*10**6)/10**6

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0, 
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    stations = donnees_pluvios.columns.values
    temps = donnees_pluvios.index

    #Krigeage
    resultats = {}
    for t in temps :
        precip = donnees_pluvios.loc[t, stations].values
        
        result_t = pd.DataFrame({"x":gx, "y":gy, "estimation":np.nan, "variance":np.nan},
                                index= grille_interp.index)

        masque = ~np.isnan(precip)
        
        if np.sum(masque) == 0: #Aucune observation dispo
            resultats[t] = result_t
            continue

        if np.all(precip[masque] == 0): #Toutes les stations 0 mm
            result_t["estimation"] = 0.0
            result_t["variance"] = 0.0
            resultats[t] = result_t
            continue

        if np.sum(masque) < 3: #Pas assez de station pour krigeage
            resultats[t] = result_t
            continue

        x_val = pluvio_xy.loc[stations[masque], "X"].values
        y_val = pluvio_xy.loc[stations[masque], "Y"].values

        
        ok = OrdinaryKriging(x_val,y_val,precip[masque],                  
                variogram_model='spherical', nlags=5)

        estim, var = ok.execute("points", gx, gy)

        estim = np.maximum(np.asarray(estim), 0.0) #Contrainte de poids
        var = np.asarray(var)

        #Resultats
        result_t["estimation"] = estim
        result_t["variance"] = var
        resultats[t] = result_t

    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return resultats


def krig_derive_altitude(grille_interp, emplacements_pluvios, donnees_pluvios,
                         chemin_resultats):
    """
    Parameters
    ----------
    grille_interp : str
        Chemin vers le fichier CSV des coordonnees de la grille d'interpolation
        (*radar_grid_xyz.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    chemin_resultats : str
        Chemin du fichier PKL ou sera enregistre le dictionnaire de resultats

    Returns
    -------
    resultats : Dictionnaire
        Dictionnaire contenant un DataFrame par pas de temps.
        Chaque df a contient les colonnes x, y, estimation et variance
    """
    #Grille d'interpolation (500x500m)
    grille_interp = (pd.read_csv(grille_interp).set_index("id")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    grille_interp[['X','Y','Z']] = np.floor(grille_interp[['X','Y','Z']]*10**6)/10**6

    gx = np.array(grille_interp['X'])
    gy = np.array(grille_interp['Y'])
    gz = np.array(grille_interp['Z'])

    #Coordonnees des pluviometres
    pluvio_xyz = (pd.read_csv(emplacements_pluvios).set_index("SONDEID")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0, 
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    stations = donnees_pluvios.columns.values
    temps = donnees_pluvios.index

    #Krigeage
    resultats = {}
    for t in temps :
        precip = donnees_pluvios.loc[t, stations].values
        
        result_t = pd.DataFrame({"x":gx, "y":gy, "z":gz, "estimation":np.nan, "variance":np.nan},
                                index= grille_interp.index)
        
        masque = ~np.isnan(precip)

        if np.sum(masque) == 0: #Aucune observation dispo
            resultats[t] = result_t
            continue

        if np.all(precip[masque] == 0): #Toutes les stations 0 mm
            result_t["estimation"] = 0.0
            result_t["variance"] = 0.0
            resultats[t] = result_t
            continue

        if np.sum(masque) < 3: #Pas assez de station pour krigeage
            resultats[t] = result_t
            continue

        x_val = pluvio_xyz.loc[stations[masque], "X"].values
        y_val = pluvio_xyz.loc[stations[masque], "Y"].values
        z_val = pluvio_xyz.loc[stations[masque], "Z"].values

        uk = UniversalKriging(x_val, y_val, precip[masque],
            variogram_model='spherical', nlags=5,
            drift_terms=['specified'], specified_drift = [z_val[masque]],
            pseudo_inv=True)

        estim, var = uk.execute("points", gx, gy, specified_drift_arrays=[gz])

        estim = np.maximum(np.asarray(estim), 0.0) #Contrainte de poids
        var = np.asarray(var)

        #Resultats
        result_t["estimation"] = estim
        result_t["variance"] = var
        resultats[t] = result_t

    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return resultats


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


def krig_derive_fixe_pluvio(radar_grid, emplacements_pluvios, donnees_pluvios, chemin_resultats):
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

    #Variogramme fixe
    coords = pluvio_xyz[["X","Y"]].values
    mask_time = (donnees_pluvios > 0).sum(axis=1) >= 3
    data_event = donnees_pluvios.loc[mask_time]

    h_vals = []
    gamma_vals = []

    for t in data_event.index:
        z = data_event.loc[t].values
        mask = ~np.isnan(z)
        z = z[mask]
        c = coords[mask]

        if len(z) < 3:
            continue

        for i, j in combinations(range(len(z)), 2):
            h = np.linalg.norm(c[i] - c[j])
            gamma = 0.5 * (z[i] - z[j])**2

            h_vals.append(h)
            gamma_vals.append(gamma)

    h_vals = np.array(h_vals)
    gamma_vals = np.array(gamma_vals)

    nlags = 5
    bins = np.linspace(0, np.max(h_vals), nlags + 1)

    bin_centers = []
    gamma_mean = []

    for k in range(nlags):
        mask = (h_vals >= bins[k]) & (h_vals < bins[k+1])

        if np.sum(mask) > 5:
            bin_centers.append(np.mean(h_vals[mask]))
            gamma_mean.append(np.mean(gamma_vals[mask]))

    bin_centers = np.array(bin_centers)
    gamma_mean = np.array(gamma_mean)

    # modèle sphérique INLINE
    popt, _ = curve_fit(lambda h, nugget, sill, rang: np.where(h < rang,
            nugget + sill * (1.5*(h/rang) - 0.5*(h/rang)**3),
            nugget + sill),bin_centers,gamma_mean, p0=[0.1, 1.0, 5000],maxfev=10000)
    nugget, sill, rang = popt

    #Krigeage
    resultats = {}
    for t in temps:
        ligne = donnees_pluvios.loc[t]

        x_val = np.array([pluvio_xyz.loc[st, 'X'] for st in stations])
        y_val = np.array([pluvio_xyz.loc[st, 'Y'] for st in stations])
        z_val = np.array([pluvio_xyz.loc[st, 'Z'] for st in stations])
        precip = np.array([ligne[st] for st in stations])

        result_t = pd.DataFrame({"x": gx,"y": gy,"z": gz,
            "estimation": np.nan,"variance": np.nan}, index=radar_grid.index)

        # CAS : pas de pluie
        if np.nansum(precip) == 0:
            result_t["estimation"] = 0
            result_t["variance"] = 0
            resultats[t] = result_t
            continue

        mask = ~np.isnan(precip)
        x_val = x_val[mask]
        y_val = y_val[mask]
        z_val = z_val[mask]
        precip = precip[mask]

        if len(precip) >= 3:
            uk = UniversalKriging(x_val, y_val, precip,variogram_model='spherical',
                variogram_parameters={"nugget": nugget,"sill": sill,"range": rang},
                drift_terms=['specified'],specified_drift=[z_val],verbose=False,
                enable_plotting=False,pseudo_inv=True)
            estim, var = uk.execute("points",gx, gy,specified_drift_arrays=[gz])

            result_t["estimation"] = estim
            result_t["variance"] = var

        resultats[t] = result_t

    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return resultats, donnees_pluvios


def valid_krig(data_krig, radar_grid, emplacements_pluvios, donnees_pluvios, date_debut, date_fin):
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

    #listes
    rmse_pasdetemps = []
    erreurs_glob = []
    #Distances
    dist = cdist(coords_stations, coords_grille)
    idx = np.argmin(dist, axis=1)

    #Boucle
    for t in periode :
        donnees_estim = data[t]

        estim_grille = donnees_estim['estimation'].values
        estim_station = estim_grille[idx]
        obs_station = (donnees_pluvios.loc[pd.Timestamp(t),
                    pluvio_xyz.index].values)

        mask = (~np.isnan(obs_station) & ~np.isnan(estim_station))
        obs_station = obs_station[mask]
        estim_station = estim_station[mask]

        if len(obs_station) == 0:
            rmse_t = np.nan
        else :
            erreurs = obs_station - estim_station
            erreurs_glob.extend(erreurs)
            rmse_t = np.sqrt(np.mean(erreurs**2))

        rmse_pasdetemps.append({'Temps': t,'RMSE': rmse_t})

    #RMSE de l'evenement
    erreurs_globales = np.array(erreurs_glob)
    rmse_global = np.sqrt(np.mean(erreurs_globales**2))
    rmse_serie = pd.DataFrame(rmse_pasdetemps)

    #Filtre RMSE
    seuil = 0.4
    rmse_mauvais = rmse_serie[rmse_serie['RMSE'] > seuil]

    return rmse_global, rmse_serie, rmse_mauvais


def visualiser_grille_IDW_pkl(grille_IDW, radar_grid, donnees_pluvios, emplacements_pluvios, date_heure, comparaison):
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
        3 : Si la methode de calcul est "krigeage avec derive externe"

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
    elif methode_calcul == 2 or 3 :
        pivot = donnees.pivot(index='y', columns ='x', values='estimation')   
        x, y = np.meshgrid(pivot.columns.values, pivot.index.values)
        precip_reshape = pivot.values
    else :
        raise ValueError("Le paramètre 'methode_calcul' doit être 1, 2 ou 3")

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
    
    #FIGURE : Krigeage ordinaire
    elif methode_calcul == 3 :                
        fig, ax = plt.subplots(figsize=(8,6))
        plt.pcolormesh(x, y, precip_reshape, shading='auto', cmap='Blues', norm=norm)
        plt.colorbar(label="Pluie (mm)")
        plt.xlabel("Longitude (m)")
        plt.ylabel("Latitude (m)")
        plt.title(f"Krigeage avec dérive externe - {date_heure}")
        
        plt.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'], c=valeurs_pluvio['precip'].astype(float), 
                    cmap='Blues', edgecolor='black',s=80, norm=norm)
    
    else :
        raise ValueError("Le paramètre 'figures' doit être 1, 2 ou 3")
    
    
    #FIGURE : Comparaison observation vs calculs
    dist = cdist(coords_stations, coords_grille) #Distance entre station et grille
    idx = np.argmin(dist, axis=1) #Indice du point grille le plus proche
    
    if methode_calcul == 1 :
        estim_result = donnees['precip'].values #Valeurs interpolées aux stations
        estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')
    elif methode_calcul == 2 or 3 :
        estim_result = donnees['estimation'].values #Valeurs interpolées aux stations
        estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')
    
    comparaison = pd.concat([data_pluvio.rename('obs'), estim_station],axis=1)
    comparaison['diff'] = np.abs(comparaison['obs'] - comparaison['estim'])
    
    comparaison.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
    plt.title("Comparaison des données observées et estimées pour tous les pluviomètres")
    plt.ylabel("Précipitation (mm)")
    plt.tight_layout()
    plt.show()
    
    #Les stations autour de la zone etudiee
    liste_stations = ['SPQ', 'CJC', 'ASC', 'SPH', 'JMJ']
    comparaison_cut = comparaison.loc[comparaison.index.intersection(liste_stations)]
    comparaison_cut.plot(y=['obs', 'estim'], kind='bar', figsize=(10,4))
    plt.title("Comparaison des données observées et estimées pour les pluviomètres autour de la zone étudiée")
    plt.ylabel("Précipitation (mm)")
    plt.tight_layout()
    plt.show()
       
    return

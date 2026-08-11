#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des pluviometres de la Ville de Sherbrooke

- "ajoute_manquantes": Ajouter explicitement les donnees manquantes afin d'obtenir
    une serie temporelle complete
- "single_raingauge_metadata" : Creer, pour chaque pluviometre, un fichier de precipitations
    completes ainsi que le fichier metadata utilise par le module RainfallQC

- "telecharger_gpm" : Telecharger les donnees de precipitation GPM IMERG demi-horaires
    (GPM_3IMERGHH) depuis EarthData, puis decouper la grille selon la zone d'etude
- "formater_gpm" : Reprojeter les coordonnees, ajuster le decalage horaire et
    selectionner les variables necessaires
- "krig_pluvio" : Effectuer un krigeage ordinaire, un krigeage avec l'altitude comme
    derive externe ou un krigeage avec des donnees satellitaires comme derive externe
- "tracer_variogramme" : Calculer et tracer les variogrammes experimentaux et les
    modeles theoriques ajustes pour chaque pas de temps

- "valid_krig_pluvio" : Valider le krigeage en calculant les erreurs absolues, 
    relatives, RMSE et MAE
- "valid_altitude" : Valider la correlation entre l'altitude et les precipitations

- "figures_periode" : Tracer une carte des precipitations interpolees pour 
    chaque pas de temps de la periode etudiee
    Option : Tracer des figures comparant les obsevations et les valeurs estimees
- "video_precip" : Creer une video (MP4 ou GIF) a partir d'une serie de figures

- "format_pcswmm" : Formater les resulats pour les rendre compatibles avec PCSWMM

@author: Marie-Amelie Boucher, USherbrooke
"""
import pickle
from pathlib import Path
from datetime import datetime
import contextily as ctx
import earthaccess
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from pykrige import OrdinaryKriging, UniversalKriging
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import linregress

def ajoute_manquantes(fichier_o, fichier_modif, date_debut, date_fin, pasdetemps):
    """
    Parameters
    ----------
    fichier_o : str
        Chemin vers le fichier .csv des donnees brutes des pluviometres
        (source : https://pluviometres.ville.sherbrooke.qc.ca/)
    fichier_modif : str
        Chemin du fichier CSV de sortie contenant la serie temporelle complete
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'
    pasdetemps : str
        Resolution temporelle des donnees : "5min" ou "1h"

    Returns
    -------
    donnees_pluvio_complet : pandas.DataFrame
        Serie temporelle complete sur la periode demandee
        Les dates manquantes sont ajoutees et leurs valeurs sont remplies avec NaN
    """
    fichier_o = Path(fichier_o) 
    fichier_modif = Path(fichier_modif)
    fichier_modif.parent.mkdir(parents=True, exist_ok=True) #Creer fichier si n'existe pas
    
    df = pd.read_csv(fichier_o, sep=';')

    if str(df.iloc[-1, 0]).strip().upper() == "TOTAL":
        df = df.iloc[:-1]

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

    if pasdetemps == "5min":
        df["Date"] = pd.to_datetime( 
            df["Date"].dt.strftime("%Y-%m-%d") + " " + 
            df["Période"].str.split(" à ").str[0] ) 
        df = df.drop(columns="Période")

    elif pasdetemps == "1h" :
        df['Heure'] = df['Période'].str.extract(r'(\d{1,2})h')[0].str.zfill(2) + ':00:00'
        df['Date'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Heure'])
        df = df.drop(columns=['Période', 'Heure'])

    else :
        raise ValueError(f"pasdetemps='{pasdetemps}' non supporté."
            "Utiliser '5min' ou '1h'.")

    df = df.set_index('Date')
    df = df.apply(pd.to_numeric, errors='coerce')

    serie_index = pd.date_range(start= date_debut, end= date_fin, freq= pasdetemps)

    donnees_pluvio_complet = df.reindex(serie_index)

    donnees_pluvio_complet.index.name = "Date"
    donnees_pluvio_complet.to_csv(fichier_modif, sep=',')

    return donnees_pluvio_complet


def single_raingauge_metadata(precip_complete, emplacements_pluvios, dossier_sortie):
    """
    Parameters
    ----------
    precip_complete : str
        Chemin vers le fichier CSV contenant les series temporelles completes
        de toutes les stations (precip_complete.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres
        (pluvio_xyz.csv)
    dossier_sortie : str
        Chemin du dossier ou seront enregistres les fichiers individuels
        et le fichier de metadata

    Returns
    -------
    metadata_df : pandas.DataFrame
        Tableau des metadonnees : station_id, latitude, longitude,
        start_datetime, end_datetime et path
    """
    #Donnees de precipitations
    donnees_precip_complet = pd.read_csv(precip_complete,sep=';',index_col=0,parse_dates=True)

    #Emplacement des pluviometres
    pluvio_xy = pd.read_csv(emplacements_pluvios)
    pluvio_xy = pluvio_xy.set_index('SONDEID')
    pluvio_xy = pluvio_xy[['Latitude','Longitude']]
    pluvio_xy = pluvio_xy.apply(pd.to_numeric)
    pluvio_xy.index = pluvio_xy.index.astype(str)

    #Creer le dossier de sortie
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    fichier_metadata = dossier_sortie / "precip_complete_metadata.csv"
    metadata_list = []

    for station in donnees_precip_complet.columns:
        #Fichier individuel
        df_station = donnees_precip_complet[[station]]
        chemin_fichier = dossier_sortie / f"precip_complete_{station}.csv"
        df_station.to_csv(chemin_fichier, sep=';')

        #Metadata
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


def telechargement_gpm(dossier_auth, dossier_download, date_debut, date_fin, bounding_box):
    """
    Parameters
    ----------
    dossier_auth : str 
        Chemin vers le dossier contenant les fichiers d'authentification EarthData
        (.edl_token, .urs_cookies et .dodsrc) Le dossier est cree s'il n'existe pas
    dossier_download : str
        Chemin vers le dossier ou les fichiers HDF5 temporaires et le fichier NetCDF
        final sont enregistres. Le dossier est cree s'il n'existe pas
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'
    bounding_box : tuple
        Limites spatiales de la zone d'etude :
            (longitude_min, latitude_min, longitude_max, latitude_max)
            #bounding_box = (-72.413, 45.022, -71.507, 45.697)

    Returns
    -------
    fichier_nc : str
        Chemin complet vers le fichier NetCDF cree

    Notes
    ------
    - Produit : GPM IMERG Half Hourly Final Run (GPM_3IMERGHH),
    - Version : 07
    - Resolution temporelle : 30 minutes
    """
    dossier_auth = Path(dossier_auth).resolve()
    dossier_auth.mkdir(parents=True, exist_ok=True)

    dossier_download = Path(dossier_download).resolve()
    dossier_download.mkdir(parents=True, exist_ok=True)

    datedebut = datetime.strptime(date_debut, "%Y-%m-%d").strftime("%Y%m%d")
    datefin = datetime.strptime(date_fin, "%Y-%m-%d").strftime("%Y%m%d")

    #Verif si le fichier netCDF existe
    fichier_nc = dossier_download / f"IMERG_{datedebut}_{datefin}.nc"

    if fichier_nc.exists():
        reponse = input (f"Le fichier existe deja : {fichier_nc}\n"
                         "Voulez-vous l'ecraser? (Oui/Non) : ")
        if reponse.lower() == "oui" :
            print("L'ancien fichier est supprime")
            fichier_nc.unlink()
        elif reponse.lower() == "non":
            print("Telecharement annule")
            return fichier_nc
        else :
            raise ValueError("Veuillez entrer Oui ou Non")

    token_file_path = dossier_auth / ".edl_token"
    urs_cookies_path = dossier_auth / ".urs_cookies"
    dodsrc_path = dossier_auth / ".dodsrc"

    #Authentification EarthData
    earthaccess.login(strategy="interactive", persist=True)

    token_info = earthaccess.get_edl_token()

    if token_info is None:
        raise RuntimeError("Impossible d'obtenir un jeton EarthData")

    token_file_path.write_text(token_info["access_token"])
    urs_cookies_path.touch()
    dodsrc_path.write_text(
        f"HTTP.COOKIEJAR={urs_cookies_path}\n"
        f"HTTP.NETRC={Path.home() / '.netrc'}")

    #Recherche des fichiers
    results = earthaccess.search_data(
        short_name="GPM_3IMERGHH",
        version="07",
        temporal=(date_debut, date_fin),
        bounding_box=(bounding_box))

    if len(results) == 0:
        raise ValueError("Aucun fichier GPM trouve pour cette periode")
    print(f"{len(results)} fichiers trouves")

    #Telechargement HDF5
    downloaded_files = earthaccess.download(results, local_path= dossier_download)

    xr.set_options(use_new_combine_kwarg_defaults=True)
    ds = xr.open_mfdataset(downloaded_files, group="Grid", combine="by_coords", data_vars="all")
    print(ds.time)
    #Decoupage spatial
    lon_min, lat_min, lon_max, lat_max = bounding_box
    ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    #Sauvegarder un seul NetCDF
    ds.to_netcdf(str(fichier_nc))
    ds.close()

    #Supprimer les fichiers HDF5 telecharges
    for fichier in downloaded_files:
        fichier = Path(fichier)
        if fichier.suffix == ".HDF5" and fichier.exists():
            fichier.unlink()

    print(f"Fichier cree : {fichier_nc}")

    return str(fichier_nc)


def formater_gpm(fichier_nc, chemin_resultats):
    """
    Parameters
    ----------
    fichier_nc : str
        Chemin vers le fichier NetCDF des donnees GPM
    chemin_resultats : str
        Chemin vers le fichier PKL de sortie

    Returns
    -------
    donnees_proj : dict
        Dictionnaire contenant les coordonnees geographiques et projetees,
        ainsi qu'un dictionnaire de grilles de precipitation par pas de temps.
    """
    chemin_resultats = Path(chemin_resultats)
    chemin_resultats.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(fichier_nc) as ds :
        #Decalage horaire - UTC vers heure locale Montreal
        temps = pd.to_datetime([t.strftime("%Y-%m-%d %H:%M:%S") for t in ds.time.values])
        temps = (temps.tz_localize("UTC").tz_convert("America/Montreal").tz_localize(None))

        #Coordonnees
        lon = ds["lon"].values
        lat = ds["lat"].values

        #Variables
        intensite = ds["precipitation"].transpose("time", "lat", "lon").values
        quality_index = ds["precipitationQualityIndex"].transpose("time", "lat", "lon").values

    #Conversion mm/h en mm - intervalle de 30 minutes
    precip = intensite * 30/60

    #Grille geographique
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2144", always_xy=True)
    x, y = transformer.transform(lon_grid, lat_grid)

    #Organiser les donnees
    donnees_gpm = {"longitude": lon_grid, "latitude": lat_grid,
                    "x": x, "y": y, "precip": {}, "quality_index": {}}

    for i, date in enumerate(temps):
        donnees_gpm["precip"][date] = precip[i, :, :]
        donnees_gpm["quality_index"][date] = quality_index[i, :, :]

    with open(chemin_resultats, "wb") as f:
        pickle.dump(donnees_gpm, f)

    return donnees_gpm


def krig_pluvio(grille_interp, emplacements_pluvios, donnees_pluvios,
                chemin_resultats, derive = None, donnees_derive = None):
    """
    Parameters
    ----------
    grille_interp : str
        Chemin vers le fichier CSV des coordonnees de la grille d'interpolation
        (*grille_interp.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    chemin_resultats : str
        Chemin du fichier PKL ou sera enregistre le dictionnaire de resultats
    derive : str (optionnel)
        - None : krigeage ordinaire
        - "altitude" : krigeage avec comme derive externe l'altitude
        - "gpm" : krigeage avec derive externe les precipitations de GPM
    donnees_derive : str (optionnel)
        Chemin vers le dictionnaire PKL contenant les donnees GPM reprojetees

    Returns
    -------
    resultats : dict
        Dictionnaire contenant un DataFrame par pas de temps.
        Chaque df contient les colonnes x, y, estimation et variance
    """
    if isinstance(derive, str):
        derive = derive.lower()
        if derive == "":
            derive = None
    if derive not in [None, "altitude", "gpm"]:
        raise ValueError("derive doit etre None, 'altitude' ou 'gpm'")

    #Grille d'interpolation (500 x 500 m)
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

    #Derive GPM
    if derive == "gpm":
        donnees_pluvios = donnees_pluvios.resample("30min").sum()
        #Donnees GPM
        with open(donnees_derive, "rb") as f:
            donnees_gpm = pickle.load(f)

        #Temps commun
        temps_gpm = pd.DatetimeIndex(donnees_gpm.keys())
        temps = donnees_pluvios.index.intersection(temps_gpm)

        if len(temps) == 0:
            raise ValueError("Aucun pas de temps commun entre les pluviometres et GPM")

        #Grille GPM
        df_gpm = donnees_gpm[temps_gpm[0]]
        gpm_xy = df_gpm[["x", "y"]].values
        tree_gpm = cKDTree(gpm_xy)

        #Associer GPM et stations
        pluvio_xy = np.column_stack((
            pluvio_xyz.loc[stations, "X"],
            pluvio_xyz.loc[stations, "Y"]))
        _, indices_gpm_pluvio = tree_gpm.query(pluvio_xy)

        #Associer GPM et grille radar
        radar_xy = np.column_stack((gx, gy))
        _, indices_gpm_radar = tree_gpm.query(radar_xy)

    else : #KO et KDE altitude gardent tous les 5 min
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

        if derive is None :
            #Krigeage ordinaire
            ok = OrdinaryKriging(x_val,y_val,precip[masque],
                    variogram_model='spherical', nlags=5)

            estim, var = ok.execute("points", gx, gy)

        elif derive == "altitude":
            #Krigeage avec derive externe l'altitude
            uk = UniversalKriging(x_val, y_val, precip[masque],
                variogram_model='spherical', nlags=5,
                drift_terms=['specified'], specified_drift = [z_val],
                pseudo_inv=True)

            estim, var = uk.execute("points", gx, gy, specified_drift_arrays=[gz])

        elif derive == "gpm" :
            #Krigeage avec derive externe gpm
            gpm_values = donnees_gpm[t]["estimation"].values
            gpm_station = gpm_values[indices_gpm_pluvio[masque]]
            gpm_radar = gpm_values[indices_gpm_radar]

            uk = UniversalKriging(x_val, y_val, precip[masque],
                variogram_model='spherical', nlags=5,
                drift_terms=['specified'], specified_drift = [gpm_station],
                pseudo_inv=True)

            estim, var = uk.execute("points", gx, gy, specified_drift_arrays=[gpm_radar])

        estim = np.maximum(np.asarray(estim), 0.0) #Contrainte de poids
        var = np.asarray(var)

        #Resultats
        result_t["estimation"] = estim
        result_t["variance"] = var
        resultats[t] = result_t

    with open(chemin_resultats, "wb") as f:
        pickle.dump(resultats, f)

    return resultats


def tracer_variogrammes(emplacements_pluvios, donnees_pluvios, date_debut, date_fin, chemin_figures=None):
    """
    Parameters
    ----------
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'
    chemin_figures : str (optionnel)
        Chemin vers le dossier ou les figures sont enregistrees en format png

    Returns
    -------
    resultats : pandas.DataFrame
        DataFrame contenant les parametres (plateau, portee, effet de pepite)
        des variogrammes et le R² pour chaque pas de temps
    """
    if chemin_figures is not None:
        chemin_figures = Path(chemin_figures)
        chemin_figures.mkdir(parents=True, exist_ok=True)
        chemin_parametres = (chemin_figures / "parametres_variogrammes.txt")
        fichier_parametres = open(chemin_parametres, "w", encoding="utf-8")
    else:
        fichier_parametres = None

    #Coordonnees des pluviometres
    pluvio_xyz = (pd.read_csv(emplacements_pluvios).set_index("SONDEID")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0,
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    #Periode d'interet
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)
    if date_fin < date_debut:
        raise ValueError("date_fin doit être après date_debut")
    donnees_pluvios = donnees_pluvios.loc[date_debut:date_fin]

    stations = donnees_pluvios.columns.values
    temps = donnees_pluvios.index

    #Variogrammes
    resultats = []
    for t in temps :
        print(f"Date : {t}")
        if fichier_parametres:
            fichier_parametres.write(f"\nDate : {t}\n")

        precip = donnees_pluvios.loc[t].values.astype(float)

        if np.all(np.isnan(precip)):
            print("Aucune précipitation enregistrée (toutes NaN)")
            if fichier_parametres:
                fichier_parametres.write("Aucune précipitation enregistrée (toutes NaN)\n")
            continue
        try:
            x = np.array([pluvio_xyz.loc[st, 'X'] for st in stations])
            y = np.array([pluvio_xyz.loc[st, 'Y'] for st in stations])
            z = np.array([pluvio_xyz.loc[st, 'Z'] for st in stations])

            uk = UniversalKriging(x, y, precip, variogram_model='spherical',
                nlags=5, drift_terms=['specified'], specified_drift=[z],
                verbose=False, enable_plotting=False, pseudo_inv=True)
            """
            ko = OrdinaryKriging(x, y, precip,variogram_model='spherical',
                                   nlags=5, enable_plotting=False, verbose=False,
                                   enable_statistics=False, coordinates_type='euclidean',
                                   pseudo_inv=True, weight=False)
            """
            # Experimental
            lags = np.asarray(uk.lags)
            semivariance = np.asarray(uk.semivariance)
            if len(lags) == 0:
                print(f"{t} : aucun lag calculé")
                continue

            params = uk.variogram_model_parameters
            partial_sill = params[0]
            vrange = params[1]
            nugget = params[2]
            sill_total = partial_sill + nugget

            if fichier_parametres:
                fichier_parametres.write(f"Nombre de lags : {len(uk.lags)}\n")
                fichier_parametres.write(f"lags : {uk.lags}\n")
                fichier_parametres.write(f"semivariance : {uk.semivariance}\n")

            # Theorique
            h = np.linspace(0, max(lags) * 1.2, 500)
            gamma = np.where(h <= vrange,
                nugget + partial_sill *(1.5 * h / vrange - 0.5 * (h / vrange) ** 3),
                sill_total)

            #R²
            gamma_interp = np.interp(lags, h, gamma)
            y_obs = semivariance
            y_pred = gamma_interp
            mask = ~np.isnan(y_obs) & ~np.isnan(y_pred)

            if np.sum(mask) > 1:
                ss_res = np.sum((y_obs[mask] - y_pred[mask])**2)
                ss_tot = np.sum((y_obs[mask] - np.mean(y_obs[mask]))**2)

                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            else:
                r2 = np.nan

            if fichier_parametres:
                fichier_parametres.write(f"R² : {r2}\n")

            resultats.append({"temps": t, "partial_sill": partial_sill,
                              "range": vrange, "nugget": nugget, "R²": r2})

            # Figures
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.scatter(lags, semivariance, s=60, label="Variogramme expérimental")
            ax.plot(h, gamma, linewidth=2, label="Modèle sphérique ajusté")

            ax.set_title(f"Variogramme - {t:%Y-%m-%d %H:%M}")

            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Semi-variance")

            ax.grid(True)

            texte = (
                f"Partial sill = {partial_sill:.3f}\n"
                f"Range = {vrange:.0f} m\n"
                f"Nugget = {nugget:.3f}\n"
                f"R² = {r2:.3f}")

            ax.text(0.98, 0.98, texte, transform=ax.transAxes,
                ha="right", va="top", bbox=dict(facecolor="white"))

            ax.legend()

            # Enregistrer les figures
            if chemin_figures is not None:
                nom_fichier = f"variogramme_{t:%Y%m%d_%H%M}.png"
                chemin_fichier = (chemin_figures / nom_fichier)
                plt.savefig(chemin_fichier, dpi=200, bbox_inches="tight")

            plt.show()
            plt.close(fig)

        except Exception as e:
            print(f"{t} : erreur : {e}")

    resultats = pd.DataFrame(resultats)
    resultats = resultats.set_index("temps")

    if fichier_parametres :
        fichier_parametres.close()

    return resultats


def valid_krig_pluvios(data_krig, grille_interp, emplacements_pluvios, 
                       donnees_pluvios, date_debut, date_fin):
    """
    Parameters
    ----------
    data_krig : str
        Chemin vers le fichier PKL contenant les donnees krigees    
        Dictionnaire avec un DataFrame par pas de temps.
        Chaque df contient les colonnes x, y, estimation et variance
    grille_interp : str
        Chemin vers le fichier CSV des coordonnees de la grille d'interpolation
        (*grille_interp.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'

    Returns
    -------
    rmse_global : float
        RMSE global calcule sur toutes les stations et tous les pas de temps
    rmse_glob_station : pandas.DataFrame
        RMSE calcule individuellemenet pour chaque station
    mae_global : float
        MAE global sur l'ensemble des stations
    mae_glob_station : pandas.DataFrame
        MAE calcule individuellemenet pour chaque station
    erreur_abs_serie : pandas.DataFrame
        Serie temporelle des erreurs absolues par station (mm)
    erreur_rel_serie : pandas.DataFrame
        Serie temporelle des erreurs relatives (%)
    flags_serie_relatif : pandas.DataFrame
        Tableau de classification des erreurs relatives :
            0 = bonne performance (≤ 20 %) ; 1 = erreur moderee (20–40 %)
            2 = erreur elevee (40–60 %) ; 3 = très mauvaise performance (> 60 %)
            4 = faible pluie (cas ou l’interpretation relative est instable)
    """
    #Lire le dictionnaire des donnees calculees
    with open(data_krig, "rb") as f:
        data = pickle.load(f)

    #Periode
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)
    periode = sorted(t for t in data
                     if date_debut <= pd.Timestamp(t) <= date_fin)

    #Grille d'interpolation (500 x 500 m)
    grille_interp = (pd.read_csv(grille_interp).set_index("id")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    grille_interp[['X','Y','Z']] = np.floor(grille_interp[['X','Y','Z']]*10**6)/10**6

    coords_grille = grille_interp[["X", "Y"]].values

    #Coordonnees des pluviometres
    pluvio_xyz = (pd.read_csv(emplacements_pluvios).set_index("SONDEID")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    coords_stations = pluvio_xyz[["X", "Y"]].values

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0,
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    #Distances
    dist = cdist(coords_stations, coords_grille)
    idx = np.argmin(dist, axis=1)

    #Dictionnaire
    erreurs_stations = {station: [] for station in pluvio_xyz.index}
    erreur_abs_serie = []
    erreur_rel_serie = []
    flags_serie_relatif = []
    erreurs_glob = []

    #Boucle
    for t in periode :
        donnees_estim = data[t]

        estim_grille = donnees_estim['estimation'].values   #Krigeage
        #estim_grille = donnees_estim['precip'].values       #IDW
        estim_station = estim_grille[idx]

        obs_station = (donnees_pluvios.loc[pd.Timestamp(t),
                    pluvio_xyz.index].values)

        ligne_temps = {'Temps': pd.Timestamp(t)}
        ligne_temps_rel = {'Temps': pd.Timestamp(t)}
        ligne_temps_flag = {'Temps': pd.Timestamp(t)}

        for i, station in enumerate(pluvio_xyz.index):
            obs = obs_station[i]
            estim = estim_station[i]

            if np.isnan(obs) or np.isnan(estim):
                ligne_temps[station] = np.nan
                ligne_temps_rel[station] = np.nan
                ligne_temps_flag[station] = np.nan
                continue

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
            if obs == 0 :
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

        erreur_abs_serie.append(ligne_temps)
        erreur_rel_serie.append(ligne_temps_rel)
        flags_serie_relatif.append(ligne_temps_flag)

    #RMSE et MAE par station
    rmse_glob_station = {}
    mae_glob_station = {}

    for station, erreurs in erreurs_stations.items():
        if erreurs:
            erreurs = np.array(erreurs)
            rmse_glob_station[station] = np.sqrt(np.mean(erreurs**2))
            mae_glob_station[station] = np.mean(np.abs(erreurs))
        else:
            rmse_glob_station[station] = np.nan
            mae_glob_station[station] = np.nan

    rmse_glob_station = pd.DataFrame([rmse_glob_station])
    mae_glob_station = pd.DataFrame([mae_glob_station])

    #RMSE et MAE global
    erreurs_globales = np.array(erreurs_glob)

    if len(erreurs_globales) > 0:
        rmse_global = np.sqrt(np.mean(erreurs_globales**2))
        mae_global = np.mean(np.abs(erreurs_globales))
    else:
        rmse_global = np.nan
        mae_global = np.nan

    #Mettre en DataFrame
    erreur_abs_serie = pd.DataFrame(erreur_abs_serie).set_index("Temps")
    erreur_rel_serie = pd.DataFrame(erreur_rel_serie).set_index("Temps")
    flags_serie_relatif = pd.DataFrame(flags_serie_relatif).set_index("Temps")

    #FIGURE - boxplot
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


def valid_altitude(emplacements_pluvios, donnees_pluvios, date_debut, date_fin):
    """
    Parameters
    ----------
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'

    Returns
    -------
    resume : dict
        Resume statistique de la relation pluie-altitude
    resultats : pandas.DataFrame
        Resultat de la regression lineaire pour chaque pas de temps
    """
    #Coordonnees des pluviometres
    pluvio_xyz = (pd.read_csv(emplacements_pluvios).set_index("SONDEID")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0,
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    #Periode
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)
    periode = donnees_pluvios.loc[date_debut:date_fin].index

    resultats = []

    for t in periode:
        obs = donnees_pluvios.loc[t]

        masque = obs.notna() #Garde les stations avec des observations
        pluie = obs[masque].values
        altitude = pluvio_xyz.loc[masque, 'Z'].values

        if len(pluie) >= 3: #minimum 3 stations pour effectuer une regression
            regression = linregress(altitude, pluie)

            resultats.append({
                'Temps': t,
                'Nb_stations': len(pluie),
                'Pente': regression.slope,
                'Intercept': regression.intercept,
                'R': regression.rvalue,
                'R2': regression.rvalue**2,
                'P_value': regression.pvalue,
                'Erreur_type': regression.stderr})
        else:
            resultats.append({
                'Temps': t,
                'Nb_stations': len(pluie),
                'Pente': np.nan,
                'Intercept': np.nan,
                'R': np.nan,
                'R2': np.nan,
                'P_value': np.nan,
                'Erreur_type': np.nan})

    resultats = pd.DataFrame(resultats)
    resultats = resultats.set_index('Temps')

    resume = {
        'R2_moyen': resultats['R2'].mean(),
        'R2_median': resultats['R2'].median(),
        'R_moyen': resultats['R'].mean(),
        'P_value_moyenne': resultats['P_value'].mean(),
        'Pourcentage_R2_sup_05': (resultats['R2'] >= 0.5).mean() * 100,
        'Pourcentage_significatif': (resultats['P_value'] < 0.05).mean() * 100}

    #Figure : Evolution du R²
    plt.figure(figsize=(12,5))
    plt.plot(resultats.index, resultats['R2'])
    plt.ylabel('R²')
    plt.xlabel('Temps')
    plt.title("Évolution temporelle de la corrélation pluie-altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Figure : Distribution des R²
    plt.figure(figsize=(6,4))
    plt.hist(resultats['R2'].dropna(), bins=15)
    plt.xlabel('R²')
    plt.ylabel('Nombre de pas de temps')
    plt.title('Distribution des R²')
    plt.tight_layout()
    plt.show()

    return resume, resultats


def figures_periode(data_krig, grille_interp, emplacements_pluvios, donnees_pluvios, 
                    date_debut, date_fin, chemin_figures=None, comparaison=0):
    """
    Parameters
    ----------
    data_krig : str
        Chemin vers le fichier PKL contenant les donnees krigees    
        Dictionnaire avec un DataFrame par pas de temps.
        Chaque df contient les colonnes x, y, estimation et variance
    grille_interp : str
        Chemin vers le fichier CSV des coordonnees de la grille d'interpolation
        (*grille_interp.csv)
    emplacements_pluvios : str
        Chemin vers le fichier CSV des coordonnees des pluviometres en metre
        (*pluvio_xyz.csv)
    donnees_pluvios : str
        Chemin vers le fichier CSV des precipitations de toutes les stations
        (*precip_complete.csv)
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'
    chemin_figures : str (optionnel)
        Chemin vers le dossier ou les figures sont enregistrees en format png
    comparaison : int (0 par default)
        Figure comparant la quantite de pluie observee et estimee
        0 : non ; 1 : oui

    Returns
    -------
    None
    """
    chemin_figures = Path(chemin_figures) if chemin_figures is not None else None
    if chemin_figures is not None:
        chemin_figures.mkdir(parents=True, exist_ok=True)

    #Lire le dictionnaire des donnees calculees
    with open(data_krig, "rb") as f:
        data = pickle.load(f)
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)

    periode = [pd.Timestamp(t) for t in data.keys()
           if date_debut <= pd.Timestamp(t) <= date_fin]
    periode.sort()

    #Grille d'interpolation (500 x 500 m)
    grille_interp = (pd.read_csv(grille_interp).set_index("id")[["X", "Y", "ELEV_1"]]
        .rename(columns={"ELEV_1": "Z"}).apply(pd.to_numeric))
    grille_interp[['X','Y','Z']] = np.floor(grille_interp[['X','Y','Z']]*10**6)/10**6

    coords_grille = grille_interp[["X", "Y"]].values

    #Emplacements des pluviometres
    pluvio_xyz = pd.read_csv(emplacements_pluvios, sep=",")
    pluvio_xyz = pluvio_xyz.set_index('Code')
    colonnes = ["X", "Y", "Z", "X_0", "Y_0", "Z_0"]
    pluvio_xyz[colonnes] = pluvio_xyz[colonnes].apply(pd.to_numeric)
    pluvio_xyz[colonnes] = np.floor(pluvio_xyz[colonnes] * 1e6) / 1e6

    dates_changement = {
    "DEA": pd.Timestamp("2020-05-22"),
    "LEN": pd.Timestamp("2022-07-20"),
    "SPH": pd.Timestamp("2023-02-01"),
    "SPQ": pd.Timestamp("2023-03-22")}

    for station, date in dates_changement.items():
        if station in pluvio_xyz.index and date_debut < date:
            pluvio_xyz.loc[station, ["X", "Y"]] = \
                pluvio_xyz.loc[station, ["X_0", "Y_0"]].values

    pluvio_xyz = pluvio_xyz[["X", "Y"]]
    xpluvio = pluvio_xyz["X"].to_numpy()
    ypluvio = pluvio_xyz["Y"].to_numpy()
    coords_stations = np.column_stack((xpluvio, ypluvio))

    #Distance entre les stations et les cellules de la grille
    dist = cdist(coords_stations, coords_grille)
    idx = np.argmin(dist, axis=1)

    #Donnees de precipitations
    donnees_pluvios = pd.read_csv(donnees_pluvios, index_col=0,
                                  parse_dates=True).apply(pd.to_numeric)
    donnees_pluvios.index.name = "Temps"

    #Echelle de couleur
    colors = ["#addd8e","#31a354","#ffff38","#f74d50","#7b3294"]
    cmap = LinearSegmentedColormap.from_list("green_purple", colors)

    vmax_event = max(np.nanmax(data[t]['estimation'].values)
                     for t in periode
                     if np.any(np.isfinite(data[t]['estimation'].values)))

    #Shapefile des bassins versants
    #Ajuster les chemins avant d'utiliser la fonction
    #contour_nick = gpd.read_file("chemin vers shapefile bassin nick/Bassin_final_modif.shp")
    #contour_wilson = gpd.read_file("chemin vers shapefile bassin wilson/Bassin_Ruisseau_Wilson.SHP")

    for date_heure in periode:
        print(f"Traitement : {date_heure}")
        date_str = date_heure.strftime("%Y%m%d_%H%M")
        donnees = data[date_heure]

        pivot = donnees.pivot(index='y',columns='x',values='estimation')
        precip_reshape = pivot.values
        x_vals = pivot.columns.values
        y_vals = pivot.index.values
        if not np.all(np.isfinite(x_vals)) or not np.all(np.isfinite(y_vals)):
            raise ValueError(f"Axes invalides (NaN/Inf) à {date_heure}")
        x, y = np.meshgrid(x_vals, y_vals)

        masked_precip = np.ma.masked_where(precip_reshape < 0.1, precip_reshape)
        norm = PowerNorm(gamma= 0.5, vmin=0.1, vmax=vmax_event)

        data_pluvio = donnees_pluvios.loc[pd.Timestamp(date_heure)]
        data_pluvio = data_pluvio.sort_index()
        data_pluvio = data_pluvio.rename('precip')

        valeurs_pluvio = pluvio_xyz.copy()
        valeurs_pluvio['precip'] = data_pluvio

        #FIGURE : Carte de precip - krigeage avec derive externe
        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(x, y, masked_precip, shading='auto', cmap=cmap, norm=norm, alpha=0.7)
            #alpha (apres norm) c'est pour la transparence (0.7)
        ax.set_xlabel("Longitude (m)")
        ax.set_ylabel("Latitude (m)")
        ax.set_title(f"Krigeage avec dérive externe - {date_heure}")

        station_vals = valeurs_pluvio['precip'].astype(float).values
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        station_couleurs = sm.to_rgba(station_vals)
        station_couleurs[station_vals < 0.1] = (1, 1, 1, 1)

        ax.scatter(valeurs_pluvio['X'], valeurs_pluvio['Y'],
                    color=station_couleurs, edgecolor='black',s=80)
        
        #contour_nick.boundary.plot(ax=ax, color="black", linewidth=1)
        #contour_wilson.boundary.plot(ax=ax, color="black", linewidth=1)
        
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Pluie (mm)")

        #Pour ajouter une basemap :
        #ctx.add_basemap(ax,crs="EPSG:32187",source=ctx.providers.OpenStreetMap.Mapnik)
        ctx.add_basemap(ax,crs="EPSG:32187",source=ctx.providers.CartoDB.Voyager)
        #Pour ajouter l'ortho photo :
        #ctx.add_basemap(ax,crs="EPSG:32187",source=ctx.providers.Esri.WorldImagery)
        plt.show()

        #Enregistrer les figures
        if chemin_figures is not None:
            nom_fichier = f"carte_{date_str}.png"
            fig.savefig(chemin_figures / nom_fichier, dpi=300)
        plt.close(fig)

        #FIGURE : Comparaison observation vs calculs
        if comparaison not in [0, 1]:
            raise ValueError("Le paramètre 'comparaison' doit être 0 ou 1")

        if comparaison == 1 :
            estim_result = donnees['estimation'].values
            estim_station = pd.Series(estim_result[idx], index=pluvio_xyz.index, name='estim')

            df_comparaison = pd.concat([data_pluvio.rename('obs'), estim_station],axis=1)
            df_comparaison['diff'] = np.abs(df_comparaison['obs'] - df_comparaison['estim'])

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            df_comparaison.plot(y=['obs', 'estim'], kind='bar', ax=ax2)
            ax2.set_title("Comparaison des données observées et estimées")
            ax2.set_ylabel("Précipitation (mm)")
            plt.tight_layout()

            if chemin_figures:
                nom_fichier = f"comparaison_{date_str}.png"
                fig2.savefig(chemin_figures / nom_fichier, dpi=300)
            plt.close(fig2)

    return


def video_precip(chemin_figures, date_debut, date_fin, format_video, prefixe="carte_", fps=3):
    """
    Parameters
    ----------
    chemin_figures : str
        Chemin vers le dossier ou les figures sont enregistrees
    date_debut : str
        Date de debut de la periode d'interet au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin de la periode d'interet au format 'YYYY-MM-DD'
    prefixe : str (optionnel)
        Par default le prefixe est "carte_"
    fps : int (optionnel)
        Vitesse de defilement. Par default 3 images par seconde
    format_video : chaine de caractere
        Format de la video : "gif" et "mp4"

    Returns
    -------
    None
    """
    chemin_figures = Path(chemin_figures)
    
    date_debut = pd.Timestamp(date_debut)
    date_fin = pd.Timestamp(date_fin)

    if format_video not in ["mp4", "gif"]:
        raise ValueError("format_video doit être 'mp4' ou 'gif'")

    nom_video = (f"video_"
                 f"{date_debut.strftime('%Y%m%d_%H%M')}_"
                 f"{date_fin.strftime('%Y%m%d_%H%M')}."
                 f"{format_video}")

    fichiers = []
    for fichier in chemin_figures.glob(f"{prefixe}*.png"):
        try:
            date_fichier = pd.to_datetime(fichier.replace(prefixe, "")
                            .replace(".png", ""),format="%Y%m%d_%H%M")
        except Exception:
            continue
        if date_debut <= date_fichier <= date_fin:
            fichiers.append((date_fichier, fichier))
    fichiers.sort()

    if len(fichiers) == 0:
        raise ValueError("Aucune image trouvee dans la periode demandee")

    chemin_video = chemin_figures / nom_video

    if format_video == "mp4" :
        with imageio.get_writer(chemin_video, fps=fps, codec="libx264", macro_block_size=None) as writer:
            for _, fichier in fichiers:
                image = imageio.imread(fichier)
                writer.append_data(image)
    else : #GIF
        images = []
        for _, fichier in fichiers:
            image = imageio.imread(fichier)
            images.append(image)
        imageio.mimsave(chemin_video, images, fps=fps)

    print(f"Video creee : {chemin_video}")

    return


index_wilson = (list(range(644, 648)) + list(range(681, 686)) +
                list(range(717, 723)) + list(range(755, 761)) +
                list(range(793, 798)))

index_nick = (list(range(547, 550)) + list(range(584, 588)) +
              list(range(622, 627)) + list(range(659, 666)) +
              list(range(698, 703)) + list(range(735, 740)) +
              list(range(773, 777)) + list(range(810, 813)) +
              list(range(847, 849)))

def format_pcswmm(index_garde, chemin_krig, chemin_timeseries):
    """
    Parameters
    ----------
    index_garde : list
        Liste des ID des cases de la grille d'interpolation que l'on veut garder
        Facilement identifiables sur PCSWMM avec le shapefile de la grille
    chemin_krig : str
        Chemin vers le fichier PKL contenant un dictionnaire de Dataframes:
        un df par pas de temps, les ID des cases de la grille en index et
        les colonnes sont "x", "y", "estimation" et "variance"
    chemin_timeseries : str
        Chemin ou le fichier CSV sera enregistre. 
        PCSWMM prend ce format de fichier.

    Returns
    -------
    timeseries_estim : pandas.DataFrame
        Index : serie de pas de temps
        Colonnes : ID des cases couvrants la region etudiee
        Valeurs : precipitation en mm
    """
    with open(chemin_krig, "rb") as f:
        resultats = pickle.load(f)

    #Couper la grille pour garder seulement la region etudiee
    resultats = {temps: df.loc[df.index.intersection(index_garde)]
                 for temps, df in resultats.items()}

    timeseries = pd.concat(resultats, names=["Temps", "points"])

    timeseries_estim = timeseries["estimation"].unstack("points")
    timeseries_estim = timeseries_estim.fillna(0) #Remplacer les nan par 0 pour PCSWMM

    timeseries_estim.index = pd.to_datetime(timeseries_estim.index)
    timeseries_estim.index = timeseries_estim.index.strftime("%m/%d/%Y %H:%M")

    timeseries_estim.to_csv(chemin_timeseries)

    return timeseries_estim
    

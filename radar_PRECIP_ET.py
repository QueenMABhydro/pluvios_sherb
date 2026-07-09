# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:07:11 2026

Traiter les donnees radar d'Environnement et Changement climatique Canada
(PRECIP-ET)
- ajoute_extension : Ajouter l'extension ".txt" aux fichiers originaux
- formater_PRECIPET : Rassembler les donnees dans un seul dictionnaire et convertir les intensites
    de precipitations (mm/h) en precipitations cumulees sur 6 minutes (mm)
- reprojection_PRECIPET : Reprojection des coordonnes du produit PRECIP-ET vers EPSG:32187
- figures_PRECIPET : Tracer les cartes du produit radar PRECIP-ET sur la ville de Sherbrooke

@author: Justine Hamelin
"""
import os
from pathlib import Path
import shutil
import pickle
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer

def ajoute_extension(dossier_source, dossier_destination=None):
    """
    Parameters
    ----------
    dossier_source : chaine de caracteres
        Chemin vers le dossier contenant les fichiers sans extention
    dossier_destination : chaine de caracteres, optional
        Chemin vers le dossier ou les fichiers sont enregistres en format txt
    """
    dossier_source = Path(dossier_source)
    
    if dossier_destination is None :
        dossier_destination = dossier_source
    else :
        dossier_destination = Path(dossier_destination)
        dossier_destination.mkdir(parents=True, exist_ok=True)
    
    for fichier in dossier_source.iterdir():
        if fichier.is_file() and fichier.suffix == "":
            destination = dossier_destination / f"{fichier.name}.txt"
            shutil.copyfile(fichier, destination)

    
def formater_PRECIPET(dossier, fichier_pkl=None, fichier_coord=None, resume=False):
    """
    Parameters
    ----------
    dossier : Chaine de caracteres
        Chemin vers le dossier contenant les fichiers txt du produit PRECIP-ET
    fichier_pkl : Chaine de caracteres (optionnel)
        Chemin vers le fichier PKL dans lequel est enregistre le dictionnaire contenant
        les donnees de PRECIP-ET. Les colonnes sont latitude, longitude, intensite et precip
    fichier_coord : Chaine de caracteres
        Chemin vers le fichier CSV dans lequel sont enregistrees les coordonnees communes a chaque grille
    resume : True/False (optionnel)
        Si True, affiche un tableau resumant les caracteristiques de chaque grille spatiale
        Par defaut, False
    
    Returns
    -------
    donnees : Dictionnaire
        Dictionnaire dont les cles correspondent aux pas de temps des fichiers et
        il est constitue de DataFrame contenant les coordonnees (latitude, longitude),
        l'intensite de precipitation (mm/h) et la precipitation cumulee sur 6 minutes (mm)
    coords : DataFrame (ou None)
        Coordonnees communes a toutes les grilles si fichier_coord est specifie, sinon None
    resume_grilles : DataFrame (ou None)
        Tableau resumant les caracteristiques des grilles si True, sinon None
    """
    dossier = Path(dossier)
    
    donnees = {}
    
    coords_communes = [] if fichier_coord is not None else None
    resume_grilles = [] if (fichier_coord is not None or resume) else None
    
    for fichier in sorted(dossier.glob("*.txt")):
        #print(f"Lecture : {fichier.name}")
        
        date = pd.to_datetime("_".join(fichier.stem.split("_")[:2]),format="%Y%m%d%H_%M")
    
        with open(fichier, "r", encoding="utf-8") as f :
            for _ in range(22):
                next(f, None)
            lignes = f.readlines()
        
        ligne = lignes[0].strip()
        ligne = ligne.removeprefix("Data").strip()
    
        valeurs = ligne.split(",")
        
        #Sachant que les donnes sont dans le meme ordre :
        df = pd.DataFrame({"latitude": valeurs[0::3], "longitude": valeurs[1::3],
            "intensite": valeurs[2::3]}).astype(float)
    
        #Remplacer les donnees manquantes par des NaN
        df["intensite"] = df["intensite"].replace(-0.0099, np.nan)
    
        #Conversion mm/h en mm sachant que la resolution est de 6 minutes
        df["precip"] = df["intensite"] * (6 / 60)
        
        donnees[date] = df[["latitude", "longitude", "intensite", "precip"]]
        
        if fichier_coord is not None:
            coords = set(zip(df.latitude, df.longitude))
            coords_communes.append(coords)
        
        if resume_grilles is not None:
            resume_grilles.append({
                "fichier": fichier.name,
                "nb_points": len(df),
                "lat_min": df["latitude"].min(),
                "lat_max": df["latitude"].max(),
                "lon_min": df["longitude"].min(),
                "lon_max": df["longitude"].max(),
                "nb_lat": df["latitude"].nunique(),
                "nb_lon": df["longitude"].nunique(),
                "dlat": np.min(np.diff(np.sort(df["latitude"].unique()))),
                "dlon": np.min(np.diff(np.sort(df["longitude"].unique())))})
    
    coords = None
    if fichier_coord is not None:
        coords_communes = set.intersection(*coords_communes)
        coords = pd.DataFrame(sorted(coords_communes),
                              columns=["latitude", "longitude"])
        coords.to_csv(fichier_coord, index=False)
    
    if resume_grilles is not None:
        resume_grilles = pd.DataFrame(resume_grilles)
        if resume:
            print(resume_grilles)
    
    # Enregistrement du dictionnaire
    if fichier_pkl is not None:
        with open(fichier_pkl, "wb") as f:
            pickle.dump(donnees, f) 
    
    return donnees, coords, resume_grilles


def reprojection_PRECIPET(donnees_radar, fichier_coords, chemin_resultats):
    """
    Parameters
    ----------
    donnees_radar : Chaine de caracteres
        Chemin vers le fichier PKL contenant un dictionnaire dont les cles correspondent aux pas de temps
        Chaque valeur est un DataFrame contenant es coordonnees (latitude, longitude),
        l'intensite de precipitation (mm/h) et la precipitation cumulee sur 6 minutes (mm)
    fichier_coords : Chaine de caracteres
        Chemin vers le fichier CSV contenant les coordonnes de la grille spatiale de la zone d'etude
    chemin_resultats : Chaine de caracteres
        Chemin vers le fichier PKL ou le dictionnaire des donnees reprojectees
        (de EPSG: 4326 vers EPSG:32187) est enregistre

    Returns
    -------
    donnees_proj : Dictionnaire
        Dictionnaire contenant les coordonnees de la grille en latitude, longitude, 
        x et y, ainsi qu'un sous-dictionnaire "dates" dont les cles correspondent 
        aux pas de temps et les valeurs sont les matrices de precipitations
    """
    dossier = os.path.dirname(chemin_resultats)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    with open(donnees_radar, "rb") as f:
        donnees = pickle.load(f)

    #Grille fixe
    coords = pd.read_csv(fichier_coords)
    lat_unique = np.sort(coords["latitude"].unique())
    lon_unique = np.sort(coords["longitude"].unique())
    lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

    #Reprojection
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32187", always_xy=True)
    x, y = transformer.transform(lon_grid, lat_grid)

    #Dictionnaire
    donnees_proj = {"longitude": lon_grid, "latitude": lat_grid, "x": x, "y": y, "dates": {}}

    for date, df in donnees.items():

        grille_precip = coords.merge(df[["latitude", "longitude", "precip"]],
            on=["latitude", "longitude"], how="left")

        pivot = grille_precip.pivot(index="latitude", columns="longitude", values="precip")
        pivot = pivot.reindex(index=lat_unique, columns=lon_unique)

        donnees_proj["dates"][date] = pivot.values

    with open(chemin_resultats, "wb") as f:
        pickle.dump(donnees_proj, f)

    return donnees_proj


def figures_PRECIPET(donnees_proj, chemin_figures=None):
    """
    Parameters
    ----------
    fichier_pkl : Chaine de caracteres
        Chemin vers le dictionnaire PKL contenant les coordonnes (latitude et longitude),
        l'intensite (mm/h) et les precipitations cumulees sur 6 minutes (mm) du produit PRECIP-ET
    chemin_figures : Chaine de caracteres (optionnel)
        Chemin vers le dossier ou les figures sont enregistrees en format PNG dans l'heure locale
    """
    save_figures = chemin_figures is not None
    if save_figures:
        os.makedirs(chemin_figures, exist_ok=True)

    with open(donnees_proj, "rb") as f:
        donnees = pickle.load(f)

    #Grille fixe
    x = donnees["x"]
    y = donnees["y"]

    #Constantes pour tracer les figures
    colors = ["#addd8e","#31a354","#ffff38","#f74d50","#7b3294"]
    cmap = LinearSegmentedColormap.from_list("green_purple", colors)
    vmax = max(np.nanmax(precip) for precip in donnees["dates"].values())
    norm = PowerNorm(gamma=0.5, vmin=0.1, vmax=vmax)

    #Pas de temps a garder pour comparer avec la resolution de 5 minutes
    dates_filtrees = [d for d in sorted(donnees["dates"])
                      if pd.to_datetime(d, format="%Y%m%d%H_%M").minute in [0, 30]]

    for date in dates_filtrees:
        #Decalage horaire
        date_utc = pd.to_datetime(date, format="%Y%m%d%H_%M", utc=True)
        date_local = date_utc.tz_convert("America/Montreal")
        print(f"Traitement : {date_local}")

        precip = donnees["dates"][date]

        masked_precip = np.ma.masked_where(np.isnan(precip) | (precip < 0.1), precip)

        fig, ax = plt.subplots(figsize=(8, 6))

        pcm = ax.pcolormesh(x, y, masked_precip, shading="auto", cmap=cmap, norm=norm, alpha=0.7)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Pluie (mm)")

        ax.set_title(f"PRECIP-ET - {date_local.strftime('%Y-%m-%d %H:%M')}")
        ax.set_xlabel("Longitude (m)")
        ax.set_ylabel("Latitude (m)")

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        #Pour ajouter une basemap :
        ctx.add_basemap(ax, crs="EPSG:32187", source=ctx.providers.OpenStreetMap.Mapnik)

        #Enregistrer les figures
        if save_figures :
            nom_fichier = f"carte_{date_local.strftime('%Y%m%d_%H%M')}.png"
            fig.savefig(os.path.join(chemin_figures, nom_fichier), dpi=300)

        plt.show()
        plt.close(fig)
        

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:07:11 2026

Traiter les donnees radar d'Environnement et Changement climatique Canada
(PRECIP-ET)
- ajoute_extension : Ajouter l'extension ".txt" aux fichiers originaux
- formater_PRECIPET : Rassembler les donnees dans un seul dictionnaire et 
    convertir les intensites de precipitations (mm/h) en precipitations 
    cumulees sur 6 minutes (mm)
- reprojection_PRECIPET : Reprojeter des coordonnes du produit PRECIP-ET 
    vers le systeme de coordonnees EPSG:32187
- figures_PRECIPET : Tracer les cartes de precipitation du produit radar 
    PRECIP-ET sur la ville de Sherbrooke
- comparaison_produits : Comparer les precipitations du produit PRECIP-ET 
    avec les grilles issues du krigeage (ordinaire ou avec derive externe)

@author: Justine Hamelin
"""
import pickle
import shutil
from pathlib import Path
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

def ajoute_extension(dossier_source, dossier_destination):
    """
    Parameters
    ----------
    dossier_source : str
        Chemin vers le dossier contenant les fichiers sans extension
    dossier_destination : str (optionnel)
        Chemin vers le dossier ou les fichiers sont enregistres en format TXTest
    """
    dossier_source = Path(dossier_source)
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
    dossier : str
        Chemin vers le dossier contenant les fichiers TXT du produit PRECIP-ET
    fichier_pkl : str (optionnel)
        Chemin vers le fichier PKL dans lequel est enregistre le dictionnaire contenant
        les donnees de PRECIP-ET. Les colonnes sont latitude, longitude, intensite et precip
    fichier_coord : str
        Chemin vers le fichier CSV dans lequel sont enregistrees les coordonnees 
        communes a chaque grille
    resume : True/False (optionnel)
        Si True, affiche un tableau resumant les caracteristiques de chaque 
        grille spatiale

    Returns
    -------
    donnees : dict
        Dictionnaire dont les cles correspondent aux pas de temps des fichiers
        Chaque valeur est un DataFrame contenant les coordonnees (latitude, longitude),
        l'intensite de precipitation (mm/h) et la precipitation cumulee sur 6 minutes (mm)
    coords : pandas.DataFrame (ou None)
        Coordonnees communes a toutes les grilles si fichier_coord est specifie, sinon None
    resume_grilles : pandas.DataFrame (ou None)
        Tableau resumant les caracteristiques des grilles si True, sinon None
    """
    dossier = Path(dossier)
    
    if fichier_pkl is not None:
        fichier_pkl = Path(fichier_pkl)
    if fichier_coord is not None:
        fichier_coord = Path(fichier_coord)

    donnees = {}

    coords_communes = [] if fichier_coord is not None else None
    resume_grilles = [] if (fichier_coord is not None or resume) else None

    for fichier in sorted(dossier.glob("*.txt")):
        date = pd.to_datetime("_".join(fichier.stem.split("_")[:2]),format="%Y%m%d%H_%M")

        with fichier.open("r", encoding="utf-8") as f:
            for _ in range(22):
                next(f, None)
            lignes = f.readlines()

        ligne = lignes[0].strip()
        ligne = ligne.removeprefix("Data").strip()

        valeurs = ligne.split(",")

        #Les donnes sont dans le meme ordre : latitude, longitude, intensite
        df = pd.DataFrame({"latitude": valeurs[0::3], "longitude": valeurs[1::3],
            "intensite": valeurs[2::3]}).astype(float)

        #Remplacer les donnees manquantes par des NaN
        df["intensite"] = df["intensite"].replace(-0.0099, np.nan)

        #Conversion mm/h en mm pour un intervalle de 6 minutes
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

    #Enregistrement du dictionnaire
    if fichier_pkl is not None:
        with fichier_pkl.open("wb") as f:
            pickle.dump(donnees, f)

    return donnees, coords, resume_grilles


def reprojection_PRECIPET(donnees_radar, fichier_coords, chemin_resultats):
    """
    Parameters
    ----------
    donnees_radar : str
        Chemin vers le fichier PKL contenant un dictionnaire dont les cles 
        correspondent aux pas de temps. Chaque valeur est un DataFrame contenant 
        les coordonnees (latitude, longitude), l'intensite de precipitation (mm/h) 
        et la precipitation cumulee sur 6 minutes (mm)
    fichier_coords : str
        Chemin vers le fichier CSV contenant les coordonnes de la grille spatiale de la zone d'etude
    chemin_resultats : str
        Chemin vers le fichier PKL ou le dictionnaire des donnees reprojectees
        (de EPSG: 4326 vers EPSG:32187) est enregistre

    Returns
    -------
    donnees_proj : dict
        Dictionnaire contenant les coordonnees de la grille en latitude, longitude,
        x et y, ainsi qu'un sous-dictionnaire "dates" dont les cles correspondent
        aux pas de temps et les valeurs sont les matrices de precipitations
    """
    donnees_radar = Path(donnees_radar)
    fichier_coords = Path(fichier_coords)
    chemin_resultats = Path(chemin_resultats)

    chemin_resultats.parent.mkdir(parents=True, exist_ok=True)

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
    fichier_pkl : str
        Chemin vers le dictionnaire PKL contenant les coordonnes (latitude et longitude),
        l'intensite (mm/h) et les precipitations cumulees sur 6 minutes (mm) du produit PRECIP-ET
    chemin_figures : str (optionnel)
        Chemin vers le dossier ou les figures sont enregistrees en format PNG dans l'heure locale
        Si None, les figures ne sont pas enregistrees
    """
    donnees_proj = Path(donnees_proj)

    if chemin_figures is not None:
        chemin_figures = Path(chemin_figures)
        chemin_figures.mkdir(parents=True, exist_ok=True)

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
        if chemin_figures :
            nom_fichier = f"carte_{date_local.strftime('%Y%m%d_%H%M')}.png"
            fig.savefig(chemin_figures / nom_fichier, dpi=300)

        plt.show()
        plt.close(fig)


def comparaison_produits(donnees_proj, data_krig):
    """
    Parameters
    ----------
    donnees_proj : str
        Chemin vers le dictionnaire PKL contenant les coordonnes (latitude/longitude
        et x/y projetees), l'intensite (mm/h) et les precipitations cumulees sur 
        6 minutes (mm) du produit PRECIP-ET
    data_krig : str
        Chemin vers le fichier PKL contenant les resultats du krigeage,
        structure par timestamps avec les coordonnees (x, y) et les precipitations estimees

    Returns
    -------
    comparaison : pandas.DataFrame
        Tableau resumant la comparaison des deux produits pour les dates communes
    """
    donnees_proj = Path(donnees_proj)
    data_krig = Path(data_krig)
    
    #Donnees PRECIP-ET
    with open(donnees_proj, "rb") as f:
        donnees = pickle.load(f)

    #Donnees krigees
    with open(data_krig, "rb") as f:
        data = pickle.load(f)

    #Coordonnees PRECIP-ET
    x_radar = donnees["x"].ravel()
    y_radar = donnees["y"].ravel()
    radar_xy = np.column_stack((x_radar, y_radar))

    #Coordonnees krigeage (avec derive externe)
    unedate = next(iter(data))
    x_krig = data[unedate]["x"].to_numpy()
    y_krig = data[unedate]["y"].to_numpy()
    krig_xy = np.column_stack((x_krig, y_krig))

    #Voisins
    arbre = cKDTree(radar_xy)
    distances, indices = arbre.query(krig_xy)

    #Dates communes
    dates_communes = sorted(set(donnees["dates"]) & set(data))

    resultats = []

    for date in dates_communes:
        pluie_radar = donnees["dates"][date].ravel()
        pluie_radar = pluie_radar[indices]

        pluie_krig = data[date]["estimation"].to_numpy()

        masque = (~np.isnan(pluie_radar)) & (~np.isnan(pluie_krig))

        if np.sum(masque) == 0:
            resultats.append({
                "date": date,
                "biais": np.nan,
                "erreur_abs_moy": np.nan,
                "rmse": np.nan,
                "ecart_type": np.nan,
                "R": np.nan,
                "R2": np.nan,
                "nb_points": 0,
                "calcul_R" : "pas de donnees"})
            continue

        diff = pluie_krig[masque] - pluie_radar[masque]

        #Correlation (1e-10 etant une tolerence car considere constant sinon)
        if (np.std(pluie_radar[masque]) > 1e-10 and np.std(pluie_krig[masque]) > 1e-10):
            R, _ = pearsonr(pluie_radar[masque], pluie_krig[masque])
            R2 = R**2
            calcul_R = "calcule"
        else:
            R = np.nan
            R2 = np.nan
            calcul_R = "valeurs constantes"

        resultats.append({
        "date": date,
        "biais": np.mean(diff),
        "erreur_abs_moy": np.mean(np.abs(diff)),
        "rmse": np.sqrt(np.mean(diff**2)),
        "ecart_type": np.std(diff),
        "R": R,
        "R2": R2,
        "nb_points": np.sum(masque),
        "calcul_R": calcul_R})

    comparaison = pd.DataFrame(resultats)

    return comparaison
    

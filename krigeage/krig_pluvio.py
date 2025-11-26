# -*- coding: utf-8 -*-
"""
12 novembre 2025

Krigeage des donnees des pluviometres de la ville de Sherbrooke
sur une grille 500x500m.

Justine Hamelin
"""

# %% Libraries
import pandas as pd
import numpy as np
from pykrige import OrdinaryKriging3D

# %% Read files
#main_dir = os.path.realpath(os.path.dirname(__file__))
main_dir = "C:/Users/hamj2113/Desktop/Collecteur_de_lhospice_original/code"

# Grilles de points xyz (500x500m) 
grille_xyz = pd.read_csv(main_dir+'/grilles_elev/grille_xyz.csv')
grille_xyz = grille_xyz[['X','Y','ELEV_1']]
grille_xyz = grille_xyz.rename(columns={'ELEV_1': 'Z'})
grille_xyz[['X','Y','Z']] = np.floor(grille_xyz[['X','Y','Z']]*10**6)/10**6

gx = np.array(grille_xyz['X'])
gy = np.array(grille_xyz['Y'])
gz = np.array(grille_xyz['Z'])

# Coordonnees xyz des pluviometres
pluvio_xyz = pd.read_csv(main_dir+'/precipitations/Emplacement des pluviomètres/pluvio_xyz.csv')
pluvio_xyz = pluvio_xyz.set_index('SONDEID')
pluvio_xyz = pluvio_xyz[['X','Y','ELEV_1']]
pluvio_xyz = pluvio_xyz.rename(columns={'ELEV_1': 'Z'})
pluvio_xyz[['X','Y','Z']] = np.floor(pluvio_xyz[['X','Y','Z']]*10**6)/10**6

# Donnees pluviometres
serie_precip = pd.read_csv(main_dir+'/precip_data/precip_complete.csv')
serie_precip = serie_precip.set_index('Unnamed: 0')
serie_precip = serie_precip.rename_axis('Temps')
serie_precip.index = pd.to_datetime(serie_precip.index)

# %% Rassembler les donnees des pluviometres aux pluviometres
stations = serie_precip.columns.values
temps = serie_precip.index

df = pd.DataFrame(index=temps)

for station in pluvio_xyz.index :
    df[f'X_{station}'] = pluvio_xyz.loc[station, 'X']
    df[f'Y_{station}'] = pluvio_xyz.loc[station, 'Y']
    df[f'Z_{station}'] = pluvio_xyz.loc[station, 'Z']
    df[f'precip_{station}'] = serie_precip[station]

# %% Krigeage
resultats = {}      

for t in temps :
    ligne = df.loc[t]       #Info des pluviometres pour 1 pas de temps
    
    x_val = np.array([ligne[f"X_{st}"]     for st in stations])        #Coordonnees X des pluviometres
    y_val = np.array([ligne[f"Y_{st}"]     for st in stations])        #Coordonnees Y des pluviometres
    z_val = np.array([ligne[f"Z_{st}"]     for st in stations])        #Coordonnees Z des pluviometres
    precip = np.array([ligne[f"precip_{st}"] for st in stations])      #Precip aux pluviometres
    
    #DataFrame du pas de temps
    result_t = pd.DataFrame({"x":gx, "y":gy, "z":gz, "estimation":np.nan, "variance":np.nan})
    
    if len(precip) > 0 and not np.all(np.isnan(precip)):
        krig = OrdinaryKriging3D(x_val,y_val,z_val,precip,variogram_model='spherical')
    
        estim, var = krig.execute("points", gx, gy, gz)
    
        result_t["estimation"] = estim
        result_t["variance"] = var
        
    else : pass
    
    resultats[t] = result_t
    

    



















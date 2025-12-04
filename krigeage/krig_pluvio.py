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

# Grille radar - centroides xyz (500x500m) 
radar_grid = pd.read_csv(main_dir+'/radar_grid/radar_grid_xyz.csv')
radar_grid = radar_grid.set_index('id')
radar_grid = radar_grid[['X','Y','ELEV_1']]
radar_grid = radar_grid.rename(columns={'ELEV_1': 'Z'})
radar_grid[['X','Y','Z']] = np.floor(radar_grid[['X','Y','Z']]*10**6)/10**6

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
    #z_val = np.array([ligne[f"Z_{st}"]     for st in stations])       #Coordonnees Z des pluviometres
    precip = np.array([ligne[f"precip_{st}"] for st in stations])      #Precip aux pluviometres
    
    #DataFrame du pas de temps
    result_t = pd.DataFrame({"x":gx, "y":gy, "estimation":np.nan, "variance":np.nan}, index= radar_grid.index)
    
    if len(precip) > 0 and not np.all(np.isnan(precip)):
        krig = OrdinaryKriging(x_val,y_val,precip,variogram_model='spherical',
                               nlags=2, enable_plotting=False, verbose=False,
                               enable_statistics=False, coordinates_type='euclidean', 
                               pseudo_inv=True, weight=False)
    
        estim, var = krig.execute("points", gx, gy, gz)
    
        result_t["estimation"] = estim
        result_t["variance"] = var
        
    else : pass
    
    resultats[t] = result_t
    
# %% Rainfall time series
timeseries = pd.concat(resultats, names=["Temps", "points"])

timeseries_estim = timeseries["estimation"].unstack("points")
timeseries_estim2 = timeseries_estim.fillna(0)                      #Remplacer les nan par 0 pour PCSWMM
timeseries_estim2.index = pd.to_datetime(timeseries_estim2.index)

#timeseries_var = timeseries["variance"].unstack("points")

#timeseries_estim2.to_csv(main_dir + "/precip_data/20250517_20250519/timeseries.csv")
    





















# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:14:22 2023

@author: reds2401
"""

# %% Libraries
import pandas as pd
import numpy as np
from scipy import stats

from tqdm import tqdm
from pykrige import OrdinaryKriging

# Read original grid files
gr_df = pd.read_csv('DEM_sherbrooke.csv') # /Data/DEM/MNE_Qc_10km.csv
gr_df['Z']=gr_df['Z'].replace(gr_df['Z'].max(),np.nan)      # Replace extreme values with NaN

# %% Read station coordinates and station records files and unify them
st_info=pd.read_csv('/Data/Stations_meteo.csv',sep=';',skiprows=3)        
st_info.set_index('NO_STATION', inplace=True)                     
st_rec=pd.read_csv('/Data/Donnees_meteo.csv',sep=';')                     
st_rec['DATE'] = pd.to_datetime(st_rec['DATE'], format='%Y-%m-%d')
st_rec.set_index('NO_STATION', inplace=True)   


    
def krig_melcc(var, year, stcoo, strec, xyz):
    """
    Kriging interpolation of meteorological data with the MELCC method.

    Parameters
    ----------
    var : String
        Variable name from the station records
    year : Integer
        Interpolation year
    stcoo : Array
        Station X/Y coordinates.
    strec : Dataframe
        Station records from a csv file.
    xyz : DataFrame
        DataFrame containing X, Y and Z coordinates from a nc file

    Returns
    -------
    kr_df : Dataframe
        Interpolated variable values to the defined grid.
    Kr_vr : Dataframe
        Variance grid of interpolated values.
    """
    #  Build year dataframe for specific variable
    grx = np.array(xyz['X'])
    gry = np.array(xyz['Y'])
    grz = np.array(xyz['Z'])
    vr_rec = strec[['DATE',var]].reset_index().set_index('DATE')    # Reset index to date
    vr_rec = vr_rec[vr_rec.index.year==year]                        # Filter by year
    st_names = vr_rec['NO_STATION'].unique()
    dates = pd.date_range(start=min(vr_rec.index),
                          end=max(vr_rec.index)).strftime('%Y-%m-%d')
    vr_df = pd.DataFrame(index=dates, columns=st_names)             # Empty DataFrame for variable
    print('Building '+var+' DataFrame')
    for st in tqdm(st_names):
        vr_df[st] = vr_rec[var][vr_rec['NO_STATION']==st]           # Fill DataFrame with station records
    vr_df=vr_df.dropna(axis=1,how='all')
    vr_df.index.name = var

    #  Temperature normalization at 0 masl
    if (var == 'TMAX' or var == 'TMIN'):
        # By every 100 m of altitude, substract gradient temperature of -0.5 C
        t_norm = list(stcoo.loc[vr_df.columns]['Z']/100*0.5)
        vr_df = vr_df+t_norm
        
    #  Kriging interpolation
    idx = pd.RangeIndex(len(grx))
    kr_df = pd.DataFrame(index=idx,columns=dates)   # Empty DataFrame for kriging interpolation
    kr_vr = kr_df.copy()                            # Empty DataFrame for kriging variance
    print('Starting Kriging interpolations for '+var)
    for d in tqdm(dates):
        kr_val = vr_df.loc[d].dropna()
        kr_val = kr_val[(np.abs(stats.zscore(kr_val)) < 3)] # Remove outliers with z-score>=3
        kr_dat = pd.concat([stcoo['X'].loc[vr_df.columns],
                            stcoo['Y'].loc[vr_df.columns],
                            kr_val],axis=1)   # DataFrame for kriging interpolation
        kr_dat = kr_dat.dropna(axis=0, how='any')   # Remove NaN values
        x = np.array(kr_dat['X'])
        y = np.array(kr_dat['Y'])
        v = np.array(kr_dat[d])
        if np.all(v==0):                            # Skip interpolation if all values are zero (assign zero to all grid)     
            kr_df[d] = 0
            kr_vr[d] = 0
        else:
            ok_interp = OrdinaryKriging(x, y, v, variogram_model='spherical', nlags=2,
                                       enable_plotting=False, verbose=False,
                                       enable_statistics=False,
                                       coordinates_type='euclidean', pseudo_inv=True, weight=False)
            kr_res, kr_var = ok_interp.execute('points',grx,gry)
            kr_df[d] = kr_res
            kr_vr[d] = kr_var
    print('Ended Kriging interpolations')
    
    #  Variable corrections
    # By every 100 m of altitude, temperatures are augmented in 0.5 C
    if (var == 'TMAX' or var == 'TMIN'):
        t_corr = grz / 100 * 0.5
        print('Correcting temperature for elevation')
        kr_df -= t_corr[:,np.newaxis]

    # Assigning 0 to all negative precipitations
    elif (kr_df < 0).any().any():
        kr_df = kr_df.clip(lower=0)
        p_corr = grz*0
        kr_df += p_corr[:,np.newaxis]   # Artificial correction to remove points outside the grid
        
    #  Assign coordinates as indexes
    kr_df.set_index(xyz.set_index(['X', 'Y']).index, inplace = True)
    kr_vr.set_index(xyz.set_index(['X', 'Y']).index, inplace = True)
    kr_df.columns = pd.to_datetime(kr_df.columns)
    kr_vr.columns = kr_df.columns
    return kr_df, kr_vr

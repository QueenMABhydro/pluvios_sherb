#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:46 2025

Fonctions pour traiter les donnees brutes des precipitometres de la 
Ville de Sherbrooke.
- "ajoute_manquantes": Ajouter explicitement les donnees manquantes
- "filtre_precip_louche": filtrer les valeurs qui semblent aberrantes
(avant de coder cette fonction, verifier si le code existe deja)

@author: Marie-Amelie Boucher, USherbrooke
"""

import pandas as pd
import numpy as np


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

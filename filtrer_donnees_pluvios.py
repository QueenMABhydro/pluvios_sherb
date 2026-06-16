# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:58:08 2026

Filtrer les donnees de precipitation - horaire

@author: Justine Hamelin
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rainfallqc import gauge_checks, timeseries_checks, comparison_checks, neighbourhood_checks

chemin_pluvios = "x/precip_complete.csv"
dossier_single_raingauge = "x"
chemin_metadata = "x/rain_gauge_metadata.csv"
chemin_donnees_filtre = "x/precip_filtre.csv"

QC_resume = pl.DataFrame({
    "QC" : ["QC3", "QC4", "QC6", "QC8", "QC9", "QC10", "QC11", "QC13", "QC21", "QC22", "QC23"],
    "Description": ["Weekday bias","Hourly bias","Breakpoint", "r99p","prcptot","world record","rx1day",
                    "daily accumulation", "timing offset", "affinity index","correlation"]})

metadata = pl.read_csv(chemin_metadata, separator=';', truncate_ragged_lines=True)

# Precip complete avec toutes les stations
donnees_pluvios = pl.read_csv(chemin_pluvios, separator=";", try_parse_dates=True)
donnees_pluvios = donnees_pluvios.rename({'': "time"})
for col in donnees_pluvios.columns:
    if donnees_pluvios.schema[col] == pl.String:
        donnees_pluvios = donnees_pluvios.with_columns(
            pl.col(col).str.strip_chars().cast(pl.Float64, strict=False))

for target_gauge_id in metadata["station_id"]:
    print(target_gauge_id)

    # Single gauge
    data = pl.read_csv((dossier_single_raingauge + f'precip_complete_{target_gauge_id}.csv'),
                       separator=';',try_parse_dates=True)
    data = data.rename({data.columns[0]: "time", data.columns[1]: "rain_mm"})
    # Metadata
    target_metadata = metadata.filter(pl.col("station_id") == target_gauge_id)
    # Coordonnees de la station
    gauge_lat = target_metadata["latitude"][0]
    gauge_lon = target_metadata["longitude"][0]
    # Trouver les stations proches
    metadata_dist = (metadata.filter(pl.col("station_id") != target_gauge_id)
        .with_columns(((pl.col("latitude") - gauge_lat)**2 + (pl.col("longitude") - gauge_lon)**2)
        .alias("distance")).sort("distance"))
    liste_stations_proches = (metadata_dist.head(12).get_column("station_id").to_list())
    station_voisine = metadata_dist.select("station_id").row(0)[0]
    # Liste des annees
    annees = sorted(data["time"].dt.year().unique().to_list())

    # %% GAUGE CHECKS
    # QC3 - Days of the week bias
    days_bias = gauge_checks.check_temporal_bias(data, "rain_mm", "weekday")

    # QC4 - Hours of day bias
    hours_bias = gauge_checks.check_temporal_bias(data, "rain_mm", "hour")
    """
    # QC5 - Intermittency
    intermittency = gauge_checks.check_intermittency(data, "rain_mm", 2, 5)
    intermittency_resultat = ("Aucun flag"
        if len(intermittency) == 0
        else f"{len(intermittency)} year(s): {', '.join(map(str, intermittency))}")
    """
    # QC6 - Breakpoints
    breakpoints = gauge_checks.check_breakpoints(data, "rain_mm")

    # %% COMPARISON CHECKS
    # QC8 - Annual exceedance of maximum R99p
    etccdi_r99p = comparison_checks.check_annual_exceedance_etccdi_r99p(data,
                    "rain_mm", gauge_lat, gauge_lon)
    r99p_annees_flag = [annees for annees, flag in zip(annees, etccdi_r99p) if flag > 0]
    r99p_resultat = ("Aucun flag"
        if sum(etccdi_r99p) == 0
        else f"{len(r99p_annees_flag)} year(s): {', '.join(map(str, r99p_annees_flag))}")

    # QC9 - PRCPTOT
    prcptot = comparison_checks.check_annual_exceedance_etccdi_prcptot(data, "rain_mm", gauge_lat, gauge_lon)
    prcptot_annees_flag = [annees for annees, flag in zip(annees, prcptot) if flag > 0]
    prcptot_resultat = ("Aucun flag"
        if sum(prcptot) == 0
        else f"{len(prcptot_annees_flag)} year(s): {', '.join(map(str, prcptot_annees_flag))}")

    # QC10 - World Record
    world_record = comparison_checks.check_exceedance_of_rainfall_world_record(data,
                        "rain_mm", "1h")
    world_record_flag = world_record.filter(
        pl.col("world_record_check").is_not_nan() &
        (pl.col("world_record_check") > 0))
    world_record_flag = world_record_flag.join(
        data.select(["time", "rain_mm"]), on="time",how="left")

    world_record_resultat = ("Aucun flag"
        if world_record_flag.is_empty()
        else "; ".join(
            f"{t:%Y-%m-%d %H:%M} ({p} mm)"
            for t, p in zip(
                world_record_flag["time"].to_list(),
                world_record_flag["rain_mm"].to_list())))

    # QC11 - Rx1day
    rx1day = comparison_checks.check_hourly_exceedance_etccdi_rx1day(data,
                "rain_mm", gauge_lat, gauge_lon)
    rx1day_flag = rx1day.filter(pl.col("rx1day_check").is_not_nan() &
        (pl.col("rx1day_check") > 0))
    rx1day_flag = rx1day_flag.join(
        data.select(["time", "rain_mm"]), on="time",how="left")

    rx1day_resultat = ("Aucun flag"
        if rx1day_flag.is_empty()
        else "; ".join(
            f"{t:%Y-%m-%d %H:%M} ({p} mm)"
            for t, p in zip(
                rx1day_flag["time"].to_list(),
                rx1day_flag["rain_mm"].to_list())))

    # %% TIMESERIES CHECKS
    # QC13 - Daily accumulation
    daily_accumul = timeseries_checks.check_daily_accumulations(data, "rain_mm", gauge_lat, gauge_lon)
    daily_accumul_flag = daily_accumul.filter(pl.col("daily_accumulation").is_not_nan() &
        (pl.col("daily_accumulation") > 0))

    daily_accumul_resultat = (
        "Aucun flag"
        if daily_accumul_flag.is_empty()
        else "; ".join(
            f"{t:%Y-%m-%d} ({v})"
            for t, v in zip(
                daily_accumul_flag["time"].to_list(),
                daily_accumul_flag["daily_accumulation"].to_list())))

    # QC15 - Streaks
    streaks = timeseries_checks.check_streaks(data, "rain_mm", gauge_lat, gauge_lon, 0.1)
    flag_cols = [c for c in streaks.columns if "streak_flag" in c]
    streaks_flag = streaks.filter(pl.any_horizontal([pl.col(c).fill_null(0) > 0
            for c in flag_cols]))

    streaks_resultat = ("Aucun flag"
        if streaks_flag.is_empty()
        else "; ".join(
            f"{t:%Y-%m-%d %H:%M} (" + ", ".join(f"{c}={v}"
                for c, v in zip(flag_cols, row)
                if v > 0) + ")"
            for t, row in zip(
                streaks_flag["time"].to_list(),
                streaks_flag.select(flag_cols).to_numpy().tolist())))

    # %% NEIGHBOURHOOD CHECKS
    # QC21 - Timings offset
    timing = neighbourhood_checks.check_timing_offset(donnees_pluvios, target_gauge_id,
        station_voisine, "1h")
    if timing == -1 :
        timing_resultat = "-1 day offset"
    elif timing == 0 :
        timing_resultat = "No offset"
    elif timing == 1 :
        timing_resultat = "+1 day offset"
    else:
        timing_resultat = f"Unknown flag ({timing})"

    # QC22 Pre-QC Affinity Index
    affinity = neighbourhood_checks.check_neighbour_affinity_index(donnees_pluvios,
        target_gauge_id, station_voisine)

    # QC23 Pre-QC Pearson correlation
    correlation = neighbourhood_checks.check_neighbour_correlation(donnees_pluvios,
        target_gauge_id, station_voisine)

    # %% RESUME
    resultats_station = [days_bias,hours_bias,breakpoints,r99p_resultat,
                         prcptot_resultat,world_record_resultat,rx1day_resultat,
                         daily_accumul_resultat,timing_resultat,
                         affinity,str(correlation)]

    QC_resume = QC_resume.with_columns(pl.Series(target_gauge_id,resultats_station,strict=False))

QC_resume.write_csv(chemin_donnees_filtre, separator=";")

# %% QC12 - CDD
qc12_resume = []
qc12_flags = pl.DataFrame({"time": donnees_pluvios["time"]})

for target_gauge_id in metadata["station_id"]:
   # Single gauge
   data = pl.read_csv((dossier_single_raingauge + f'precip_complete_{target_gauge_id}.csv'),
                      separator=';',try_parse_dates=True)
   data = data.rename({data.columns[0]: "time", data.columns[1]: "rain_mm"})
   # Metadata
   target_metadata = metadata.filter(pl.col("station_id") == target_gauge_id)
   # Coordonnees de la station
   gauge_lat = target_metadata["latitude"][0]
   gauge_lon = target_metadata["longitude"][0]

   cdd = timeseries_checks.check_dry_period_cdd(data, "rain_mm", "hourly", gauge_lat, gauge_lon)

   qc12_flags = qc12_flags.join(cdd.select(["time", "dry_spell_flag"])
                    .rename({"dry_spell_flag": target_gauge_id}),on="time",how="left")
   flags = cdd["dry_spell_flag"]

   qc12_resume.append({
        "station": target_gauge_id,
        "flag_1": int((flags == 1).sum()),
        "flag_2": int((flags == 2).sum()),
        "flag_3": int((flags == 3).sum()),
        "flag_4": int((flags == 4).sum())})

qc12_resume = pl.DataFrame(qc12_resume)

# %% QC15- Streaks
qc15_resume = []

for target_gauge_id in metadata["station_id"]:
   # Single gauge
   data = pl.read_csv((dossier_single_raingauge + f'precip_complete_{target_gauge_id}.csv'),
                      separator=';',try_parse_dates=True)
   data = data.rename({data.columns[0]: "time", data.columns[1]: "rain_mm"})
   # Metadata
   target_metadata = metadata.filter(pl.col("station_id") == target_gauge_id)
   # Coordonnees de la station
   gauge_lat = target_metadata["latitude"][0]
   gauge_lon = target_metadata["longitude"][0]

   streaks = timeseries_checks.check_streaks(data, "rain_mm", gauge_lat, gauge_lon, 0.1)
   flag_cols = [c for c in streaks.columns if "streak_flag" in c]

   flags_mat = streaks.select(flag_cols).fill_null(0).cast(pl.Int8)
   nb_events = (flags_mat > 0).to_numpy().sum()

   qc15_resume.append({
    "station": target_gauge_id,
    "nb_events": int(nb_events),
    **{c: int((flags_mat[c] > 0).sum()) for c in flag_cols}})

qc15_resume = pl.DataFrame(qc15_resume)

# %% QC16 - Daily neighbours (wet)
qc16_resume = []
qc16_flags = pl.DataFrame({"time": donnees_pluvios["time"]})

for target_gauge_id in metadata["station_id"]:

    target_metadata = metadata.filter(pl.col("station_id") == target_gauge_id)

    gauge_lat = target_metadata["latitude"][0]
    gauge_lon = target_metadata["longitude"][0]

    metadata_dist = (metadata.filter(pl.col("station_id") != target_gauge_id)
        .with_columns(((pl.col("latitude") - gauge_lat)**2 + (pl.col("longitude") - gauge_lon)**2)
        .alias("distance")).sort("distance"))

    liste_stations_proches = (metadata_dist.head(7).get_column("station_id").to_list())

    # QC16
    wet_neighbours = neighbourhood_checks.check_wet_neighbours(donnees_pluvios, target_gauge_id,
        liste_stations_proches, "1h", 0.1, 3)

    qc16_flags = qc16_flags.with_columns(
        wet_neighbours["wet_spell_flag_1h"].alias(target_gauge_id))

    wet_flag = wet_neighbours.filter(
        pl.col("wet_spell_flag_1h").is_not_nan() &
        (pl.col("wet_spell_flag_1h") > 0))

    flags = wet_flag["wet_spell_flag_1h"]
    n1 = (flags == 1).sum()
    n2 = (flags == 2).sum()
    n3 = (flags == 3).sum()

    qc16_resume.append({"station": target_gauge_id,"flag_1": n1,"flag_2": n2,"flag_3": n3})

qc16_resume = pl.DataFrame(qc16_resume)

qc16_total = pl.DataFrame({
    "station": ["TOTAL"],
    "flag_1": [qc16_resume["flag_1"].sum()],
    "flag_2": [qc16_resume["flag_2"].sum()],
    "flag_3": [qc16_resume["flag_3"].sum()]})

#qc16_resume = pl.concat([qc16_resume, qc16_total])

# FIGURE
stations = qc16_resume["station"].to_list()

x = np.arange(len(stations))
width = 0.25
plt.figure(figsize=(12,6))

plt.bar(x - width, qc16_resume["flag_1"], width, label="Flag 1")
plt.bar(x,         qc16_resume["flag_2"], width, label="Flag 2")
plt.bar(x + width, qc16_resume["flag_3"], width, label="Flag 3")

plt.xticks(x, stations, rotation=45)
plt.ylabel("Nombre d'occurrences")
plt.xlabel("Station")
plt.title("QC16 - Wet neighbours")
plt.legend()
plt.tight_layout()
plt.show()

# %% QC18 - Daily neighbours (dry)
qc18_resume = []
for target_gauge_id in metadata["station_id"]:

    target_metadata = metadata.filter(pl.col("station_id") == target_gauge_id)

    gauge_lat = target_metadata["latitude"][0]
    gauge_lon = target_metadata["longitude"][0]

    metadata_dist = (metadata.filter(pl.col("station_id") != target_gauge_id)
        .with_columns(((pl.col("latitude") - gauge_lat)**2 + (pl.col("longitude") - gauge_lon)**2)
        .alias("distance")).sort("distance"))

    liste_stations_proches = (metadata_dist.head(7).get_column("station_id").to_list())

    #QC18
    dry_neighbours = neighbourhood_checks.check_dry_neighbours(donnees_pluvios, target_gauge_id,
        liste_stations_proches, "1h", 3)
    dry_flag = dry_neighbours.filter(
        pl.col("dry_spell_flag_1h").is_not_nan() &
        (pl.col("dry_spell_flag_1h") > 0))

    flags = dry_flag["dry_spell_flag_1h"]
    n1 = (flags == 1).sum()
    n2 = (flags == 2).sum()
    n3 = (flags == 3).sum()

    qc18_resume.append({"station": target_gauge_id,"flag_1": n1,"flag_2": n2,"flag_3": n3})

qc18_resume = pl.DataFrame(qc18_resume)

qc18_total = pl.DataFrame({
    "station": ["TOTAL"],
    "flag_1": [qc18_resume["flag_1"].sum()],
    "flag_2": [qc18_resume["flag_2"].sum()],
    "flag_3": [qc18_resume["flag_3"].sum()]})

#qc18_resume = pl.concat([qc18_resume, qc18_total])

# FIGURE
stations = qc18_resume["station"].to_list()

x = np.arange(len(stations))
width = 0.25
plt.figure(figsize=(12,6))

plt.bar(x - width, qc18_resume["flag_1"], width, label="Flag 1")
plt.bar(x,         qc18_resume["flag_2"], width, label="Flag 2")
plt.bar(x + width, qc18_resume["flag_3"], width, label="Flag 3")

plt.xticks(x, stations, rotation=45)
plt.ylabel("Nombre d'occurrences")
plt.xlabel("Station")
plt.title("QC18 - Dry neighbours")
plt.legend()
plt.tight_layout()
plt.show()

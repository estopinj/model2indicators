#!/usr/bin/python
# -*- coding: utf-8 -*-

# ==============================================================================
# author          :Ghislain Vieilledent, Joaquim Estopinan (coding on Ghislain's work)
# email           :ghislain.vieilledent@cirad.fr, ghislainv@gmail.com, estopinan.joaquim@gmail.com
# web             :https://ghislainv.github.io
# python_version  :>=3.0
# license         :GPLv3
# ==============================================================================

# Imports
import subprocess
from util.tif2cog import tif2cog
from pathlib import Path
import glob
from datetime import datetime



def maps_merge_all(indices_params=None, deletes_src_tifs=False):
    L = []
    for status_lvl in ["iucn", "comp"]:
        for type in ["cats", "shannon", "B", "S"]:
            l = maps_merge_one(status_lvl, type, indices_params=indices_params, deletes_src_tifs=deletes_src_tifs)
            L.append(l)
    return L


def maps_merge_one(status_lvl, type, indices_params=None, deletes_src_tifs=False):
    # Retrieves argts
    levels = indices_params["levels"]
    num_threads = indices_params["num_threads"]
    gdal_cachemax = indices_params["gdal_cachemax"]
    statuses = indices_params["Lstats"]

    # Date for folder name
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%Hh%M")

    # Paths construction
    type_folder = "shannon" if type=="shannon" else '_'.join([status_lvl+'-'+s for s in statuses]) if type=="S" else status_lvl+'-'+type
    tifs_folder = indices_params["maps_path"] / type_folder / indices_params["grid_name"]


    # Output name
    out_p = Path("../../out/maps/merged/") / type_folder / indices_params["grid_name"]
    out_name = "merged_" + dt_string
    # Creates destination folder if does not already exist
    out_p.mkdir(parents=True, exist_ok=True)


    # STEP 1: DECOMPRESS LZ4 FILES WHEN NECESSARY
    target_lz4 = tifs_folder / "*.lz4"
    print("*** target_lz4 *** :", target_lz4)
    # List all .lz4 from target_path
    L_lz4 = glob.glob(str(target_lz4))
    print("len(L_lz4):", len(L_lz4))
    if len(L_lz4)>0:
        cmd = ["lz4", "-d", "-m", str(target_lz4)]
        subprocess.call(" ".join(cmd), shell=True)


    # STEP 2: LIST ALL TIFS FILES in tifs_folder
    # targeted tifs
    target_tifs = tifs_folder / "*.tif"
    print("*** target_tifs *** :", target_tifs)
    # List all .tif from target_path
    L_tifs = glob.glob(str(target_tifs))
    print("len(L_tifs):", len(L_tifs))

    # STEP 3: LAUNCH CSV MERGE
    # List .txt and .tif files
    list_txt = out_name + '.txt'
    out_tif = out_name + '.tif'

    # Creates list .txt file
    with open(tifs_folder / list_txt, 'w') as f:
        for line in L_tifs:
            f.write(f"{line}\n")

    # --- Creates COG .tif file ---
    check = tif2cog(input_file_list=str(tifs_folder / list_txt),
                    output_file=str(out_p / out_tif),
                    levels=levels,
                    num_threads=num_threads,
                    cachemax=gdal_cachemax)
    print("\n\t\t\t*** ***\n", check, "\n\t\t\t*** ***\n")


    if deletes_src_tifs:
        # STEP 4: DELETES DECOMPRESSED TIF FILES
        cmd = ["rm", str(target_tifs)]
        subprocess.call(" ".join(cmd), shell=True)

    return out_p

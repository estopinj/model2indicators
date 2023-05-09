import os
import os.path
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import time

from rasterio import features
from rasterio.enums import Resampling
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from pathlib import Path

import rasterio
from rasterio.merge import merge
import shutil
import ast


def loads_grid_ref(occurrences):
    grid = pd.read_csv(occurrences, sep=";", header='infer', low_memory=False)
    # Convert columns types to best suited types
    grid = grid.convert_dtypes()
    grid = grid.drop_duplicates()

    print("Grid reference shape\t:", grid.shape, grid.columns)
    print("Grid reference .head(10)\t:", grid.head(10))
    return grid


def out2df(output, grid):
    # Converts list to array
    arr = np.asarray(output) #, dtype=np.float64) & NOW Try like this!    ## TRY LIKE THIS, dtype=np.float32) ####
    print("arr.shape\t:", arr.shape)
    # print(arr)
    # Converts to df
    df = pd.DataFrame(arr, columns=["gbifid", "classes", "probas"])
    print("df.shape\t:", df.shape)
    # Aggregates by gbifid
    print("Unique gbifid groups sizes", set(df.groupby("gbifid").size()))
    df = df.groupby("gbifid").aggregate(list)
    df.reset_index(inplace=True)
    print("After group_by and index reset:")
    print("df.shape, df.columns\t:", df.shape, df.columns, "\ndf.head()\t:", df.head())
    # Converts lists back to arrays to allow selection by list
    df['classes'] = df['classes'].apply(lambda x: np.array(x))
    df['probas'] = df['probas'].apply(lambda x: np.array(x))

    # TYPES conversion
    df = df.convert_dtypes()
    grid = grid.convert_dtypes()
    print("df.shape\t\t:", df.shape)
    print("grid.shape\t\t:", grid.shape)
    # MERGE BETWEEN OUTPUT & REF
    df = df.merge(grid, how="left", on="gbifid")

    # Deletes rows with ANY NA value
    # df.dropna(inplace=True)

    return df


def appliesT(grid, params):
    grid["probas"]  = grid.apply(lambda r: [p for p in r.probas if p >= params['T']], axis=1)
    grid["classes"] = grid.apply(lambda r: r.classes[:len(r["probas"])], axis=1)
    return grid


def loads_DZ(path):
    with open(path, 'r') as f:
        DZ = json.load(f)

    # Converts DZ keys back to int
    keys_int = list(map(int, list(DZ.keys())))
    DZ = dict(zip(keys_int, DZ.values()))

    # Converts each lvl keys back to int
    for lvl in [1, 2, 3]:
        keys_int = list(map(int, list(DZ[lvl].keys())))
        DZ[lvl] = dict(zip(keys_int, DZ[lvl].values()))

    return DZ


def df2gdf(df, params):
    # Converts to gdf
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.decimallongitude, df.decimallatitude), crs=4326)

    if params["removes_oor"] and params["grid_name"] != "W2_05_grid":
        # Spatial join with lvl1_gdf
        df = df.sjoin(params["lvl1_gdf"], how='left')
        df.drop(columns=["index_right", "LEVEL1_NAM"], inplace=True)
        # Attributes nearest lvl1 codes to points outside base geometries - quicker in two steps like that
        df.loc[pd.isna(df.LEVEL1_COD), "LEVEL1_COD"] = df[pd.isna(df.LEVEL1_COD)].sjoin_nearest(params["lvl1_gdf"]).LEVEL1_COD_right
        # NEWLINE TO DELETE SJOIN_NEAREST DUPLICATES
        df.drop_duplicates(subset="gbifid", inplace=True)

    if params["removes_oor"] and params["grid_name"] == "W2_05_grid":
        df['LEVEL1_COD'] = df['LEVEL1_COD'].apply(lambda x: ast.literal_eval(x)[0])

    df = df.convert_dtypes()
    return df


def removes_oor_species(row, DZ, lvl=1):
    selector       = [row["LEVEL"+str(lvl)+"_COD"] in DZ[lvl][spK] for spK in row['classes']]
    row['classes'] = np.array(row['classes'])[selector]
    row['probas']  = np.array(row['probas'])[selector]
    return row


def applies_oor_filter(df, params):
    if params['removes_oor']:
        df = df.apply(lambda row: removes_oor_species(row, params['DZ']), axis=1)
    return df


def normAbT(df, params):
    if params["norm_AbTprobas"]:
        df['probas'] = df.apply(lambda row: row['probas'] / np.sum(row['probas']), axis=1)
    return df


# ***


def loads_GT(path):
    GT = pd.read_csv(path, sep=',')
    # Convert columns types to best suited types
    GT = GT.convert_dtypes()
    GT.drop("Unnamed: 0", axis=1, inplace=True)
    GT.rename(columns={'labels': 'GT'}, inplace=True)

    # GT binary
    def threat_binary(row):
        return True if row["GT"] in ["VU", "EN", "CR"] else False

    GT['GTb'] = GT.apply(lambda row: threat_binary(row), axis=1)

    return GT


def loads_predsBS(params):

    # Binary preds
    pred_B = pd.read_csv(params["iucn_path"]+"predsIUCNN_binary.csv", sep=',')
    # Convert columns types to best suited types
    pred_B = pred_B.convert_dtypes()
    pred_B.drop("Unnamed: 0", axis=1, inplace=True)
    pred_B.rename(columns={'preds': 'pred'}, inplace=True)
    def pred_threat(row):
        return True if row["pred"] == "Threatened" else False
    pred_B['pred_B'] = pred_B.apply(lambda row: pred_threat(row), axis=1)

    # Status preds
    pred_S = pd.read_csv(params["iucn_path"]+"predsIUCNN_statuses.csv", sep=',')
    # Convert columns types to best suited types
    pred_S = pred_S.convert_dtypes()
    pred_S.drop("Unnamed: 0", axis=1, inplace=True)
    pred_S.rename(columns={'preds': 'pred_S'}, inplace=True)

    return pred_B, pred_S


def computes_species_ref(ref_path):
    # Reads data
    ref_df = pd.read_csv(ref_path, sep=';')
    # Convert columns types to best suited types
    ref_df = ref_df.convert_dtypes()

    df = pd.DataFrame(ref_df.groupby('speciesKey').first()['canonical_name']).reset_index()
    return df.rename(columns={"canonical_name": "species"})




def status_table(params):
    # Loads data
    df_sp          = computes_species_ref(params["ref_path"])
    GT             = loads_GT(params["iucn_path"]+"IUCN_GroundTruth.csv")
    pred_B, pred_S = loads_predsBS(params)

    # Merge into table
    df = df_sp.merge(GT, how="left", on="species")
    df = df.merge(pred_B, how="left", on="species")
    df = df.merge(pred_S, how="left", on="species")
    df = df.convert_dtypes()

    # Helper functions
    def pred_B_only(row):
        return row["pred_B"] if pd.isna(row["GT"]) else pd.NA
    def pred_S_only(row):
        return row["pred_S"] if pd.isna(row["GT"]) else pd.NA
    def pred_B_onGT(row):
        return pd.NA if pd.isna(row["GT"]) else row["pred_B"]
    def pred_S_onGT(row):
        return pd.NA if pd.isna(row["GT"]) else row["pred_S"]
    def GTorPred_B(row):
        return row["pred_B"] if pd.isna(row["GT"]) else row["GTb"]
    def GTorPred_S(row):
        return row["pred_S"] if pd.isna(row["GT"]) else row["GT"]

    df['pred_B_only'] = df.apply(lambda row: pred_B_only(row), axis=1)
    df['pred_S_only'] = df.apply(lambda row: pred_S_only(row), axis=1)
    df['pred_B_onGT'] = df.apply(lambda row: pred_B_onGT(row), axis=1)
    df['pred_S_onGT'] = df.apply(lambda row: pred_S_onGT(row), axis=1)
    df['GT_o_pred_B'] = df.apply(lambda row: GTorPred_B(row), axis=1)
    df['GT_o_pred_S'] = df.apply(lambda row: GTorPred_S(row), axis=1)
    df = df.convert_dtypes()

    return df


# ***


def threat_dict(df, level="B", source="comp"):                     # level = "B" or "S / source = "comp" or "iucn"
    assert level == "B" or level == "S"
    assert source == "comp" or source == "iucn"

    if source == "comp":
        var = "GT_o_pred_"+level
    if source == "iucn":
        var = "GT" if level == "S" else "GTb"

    keys = list(df.speciesKey)
    return dict(zip(keys, list(df[var])))


def classes_selector(row, D, classes="classes", status=None):
    if status is None:
        return np.array([D[spK] if not pd.isna(D[spK]) else False for spK in row[classes]])
    else:
        return np.array([True if (not pd.isna(D[spK]) and D[spK] == status) else False for spK in row[classes]])


def matching_probas(row, classes_name, probas_name="probas"):
    selected_classes = np.array(row[classes_name])
    selected_probas = np.array(row[probas_name])
    return selected_probas[selected_classes.astype(bool)]


def sumap(grid, sp_ref, source="comp", level="B", status=None):

    D = threat_dict(sp_ref, level=level, source=source)

    grid['selected_classes'] = grid.apply(lambda row: classes_selector(row, D, status=status, classes="classes"), axis=1)
    grid['selected_probas']  = grid.apply(lambda row: matching_probas(row, classes_name="selected_classes", probas_name="probas"), axis=1)

    name = "-".join([source, status]) if status is not None else "-".join([source, level])
    grid[name] = grid.apply(lambda row: np.sum(row['selected_probas']), axis=1)
    return grid, name


# ***


def shannon(grid, params):
    def shannon(ps, norm=False):
        if norm and len(ps) != 0:
            return -1 / len(ps) * np.sum(ps * np.log(ps))
        else:
            return -np.sum(ps * np.log(ps))

    shannon_name = 'shannon'
    grid[shannon_name] = grid.apply(lambda row: shannon(row["probas"], norm=params["norm_shannon"]), axis=1)
    return grid, shannon_name



def computes_empty_preds(grid, params):
    name = params['empty_name']
    # No preds above T
    grid[name] = grid.apply(lambda r: len(r["probas"]) == 0, axis=1)
    grid[name] = grid[name].astype(bool)
    return grid



def worst_status_cat(df, params, source="comp"):
    # Name
    cats_name = source + "-cats"

    # Initializes new categorical variable
    df[cats_name] = params["empty_value"]
    df[cats_name] = df[cats_name].astype(np.uint8)

    for i, s in enumerate(params["Lstats"]):
        # Each status variable name
        stat_name = "-".join([source, s])
        # Each status nonzero indices
        stat_idxs = df[stat_name].to_numpy().nonzero()[0]           # np.flatnonzero(x)
        # Assigns int to matching nonzero points
        df.loc[stat_idxs, cats_name] = i + 1

    return df, cats_name


# ***


# rasterize function
def rasterize_dict(secs=30, fill=-1, dtype=None, **kwargs):
    params = dict(kwargs)
    params["secs"]    = secs
    params["touched"] = False
    params["fill"]    = fill
    params["dtype"]   = dtype
    params["transform"] = Affine(params["secs"]/3600, 0, -180, 0, -params["secs"]/3600, 90)
    params["out_shape"] = (180*3600//params["secs"], 360*3600//params["secs"])
    return params


def rasterization(grid, var, raster_params, indices_params):
    # Applies empty values mask to var
    grid[var] = grid.apply(lambda row: indices_params["empty_value"] if row[indices_params["empty_name"]] else row[var], axis=1)
    # Geoms preparation
    geoms = grid[['geometry', var]].values.tolist()

    # Rasterizes vector using the shape and transform of the raster
    arr = features.rasterize(geoms,
                             out_shape=raster_params["out_shape"],
                             transform=raster_params["transform"],
                             all_touched=raster_params["touched"],
                             fill=raster_params["fill"],  # background value
                             dtype=raster_params["dtype"])

    return arr



def stacking(Dz):
    name_stack = "_".join(list(Dz.keys()))
    data_stack = np.stack(list(Dz.values()))
    return name_stack, data_stack



def defines_profile(stack_shape, data_dict, blocksize=512, crs=4326, tiled=False, compress="deflate"):
    profile = DefaultGTiffProfile(count=stack_shape[0])

    profile.update(
        height=stack_shape[1],
        width=stack_shape[2],
        blockxsize=blocksize,
        blockysize=blocksize,
        transform=data_dict["transform"],
        crs=crs,
        nodata=data_dict["fill"],
        dtype=data_dict["dtype"],
        tiled=tiled,
        compress=compress
    )
    return profile


def saves_buffer_tif_disk(stack, stack_name, profile, params, buff_count=0):
    # Destination
    folder = os.path.join(params["maps_path"], stack_name, params["grid_name"])
    Path(folder).mkdir(parents=True, exist_ok=True)
    path_name = os.path.join(folder, str(buff_count) + ".tif")

    # Register GDAL format drivers and configuration options with a context manager.
    with rasterio.Env():
        with rasterio.open(path_name, 'w', **profile) as dst:
            dst.write(stack, list(np.arange(1, stack.shape[0] + 1)))
    # Context manager exits and all drivers are de-registered.
    return path_name


def saves_buffer_tif(stack, stack_name, profile, params, factors=None, buff_count=0):
    start_tif = time.time()

    # Tiled tif, possibly a geotif if factors are provided
    if params["tiled"]:
        profile.update(tiled=params["tiled"])

        # Raster + Overviews creation
        with MemoryFile() as mf, mf.open(**profile) as dst:
            dst.write(stack, list(np.arange(1, stack.shape[0] + 1)))
            dst.build_overviews(factors, Resampling.average) if factors is not None else ""

            # Destination
            folder = os.path.join(params["maps_path"], stack_name, params["grid_name"])
            Path(folder).mkdir(parents=True, exist_ok=True)
            path_name = os.path.join(folder, str(buff_count) + ".tif")
            # tif creation
            copy(dst,
                 path_name,
                 copy_src_overviews=factors is not None,
                 **profile)

    # Not a Cloud Optimized Geotiff: Writing is much faster directly on disk
    else:
        path_name = saves_buffer_tif_disk(stack, stack_name, profile, params, buff_count=buff_count)

    print(
        '**TIME** ' + stack_name + '.tif: \n\t\t\tTime\t: ' + str(np.round(time.time() - start_tif)) + "s.")
    return path_name


# ***
def grid2tif(grid, params, iucn_table=None, src="comp", var_type="B", exp_count=0):
    assert var_type in ["B", "S", "shannon", "cat"]

    # Prepares dict
    if var_type=="cat":
        map_dict = rasterize_dict(secs=params["seconds"], fill=params["fill_cat"], dtype=params["dtype_cat"])
    else:
        # Prepares map dict
        map_dict = rasterize_dict(secs=params["seconds"], fill=params["fill_sumap"], dtype=params["dtype_sumap"])

    # Initializes Dz
    Dz = {}

    # Binary Sumap
    if var_type=="B":
        grid, var_name = sumap(grid, iucn_table, source=src, level=var_type)
        Dz[var_name]   = rasterization(grid, var_name, map_dict, params)
    # Status Sumaps
    elif var_type=="S":
        for s in params["Lstats"]:
            grid, var_name = sumap(grid, iucn_table, source=src, level=var_type, status=s)
            Dz[var_name]   = rasterization(grid, var_name, map_dict, params)
    # Shannon
    elif var_type=="shannon":
        grid, var_name = shannon(grid, params)
        Dz[var_name]   = rasterization(grid, var_name, map_dict, params)
    # Worst cat
    elif var_type=="cat":
        grid, var_name = worst_status_cat(grid, params, source=src)
        Dz[var_name] = rasterization(grid, var_name, map_dict, params)
    else:
        print("Unknown var_type")


    # Stacking result
    name_stack, stack = stacking(Dz)
    print("Nb Non-empty cells:", len(stack[np.where(stack != map_dict["fill"])]))
    # Defines profile
    profile = defines_profile(stack.shape, map_dict, compress=params["compress"])
    # Saves tif
    buff_str = saves_buffer_tif(stack, name_stack, profile, params, buff_count=exp_count)
    return buff_str


def incremental_tif_merge(tif, tif2=None, dest_p=None, copy=True, copy_name="last_merge.tif", block_size=512,
                          verbose=False):
    '''
    If tif2 is provided, merges tif with tif2.
        Otherwise: Checks in tif_folder/merged/ if last_merge.tif exists:
        If it does, merge tif with it.
            Otherwise: does not merge tif with anything but still creates dest & copy files filled with tif only for coherence.
    '''
    print("Tif merge begins...:", tif.split('/')[-1])
    start_merge = time.time()

    L2mosaic = [tif]

    # Retrieves tif path and name
    tif_p, tif_n = os.path.dirname(tif), os.path.basename(tif)[:-4]
    # Creates merged folder if does not already exist
    merged_folder = os.path.join(tif_p, "merged")
    print("merged_folder:", merged_folder)
    Path(merged_folder).mkdir(parents=True, exist_ok=True)

    # Checks if tif2 is provided:
    if tif2 is not None:
        L2mosaic.append(tif2)

        # Output names
        tif2_n = os.path.basename(tif2)[:-4]
        dest_p = os.path.join(merged_folder, tif_n + '_' + tif2_n + "_merged.tif") if dest_p is None else dest_p
        copy_p = os.path.join(merged_folder, "last_tif2merge.tif") if copy else None

    else:
        # Only tif is provided: Test if last_merge.tif is in merged_folder
        if copy_name in os.listdir(merged_folder):
            # Merge tif with last_merged
            L2mosaic.append(os.path.join(merged_folder, copy_name))
        # else :
        # tif is not merged with any other .tif file,
        # but tif_merged.tif & last_merged.tif are created for coherence with future merges.

        # Output names
        dest_p = os.path.join(merged_folder, tif_n + "_merged.tif") if dest_p is None else dest_p
        copy_p = os.path.join(merged_folder, copy_name) if copy else None

    # List of the source files opened in read mode
    src_files = [rasterio.open(fp) for fp in L2mosaic]
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files)

    # Copy the metadata
    out_meta = rasterio.open(tif).meta.copy()
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     'blockysize': block_size,
                     'blockxsize': block_size,
                     'tiled': False,
                     'compress': 'deflate',
                     'interleave': 'band'
                     }
                    )
    print(out_meta) if verbose else None

    # Write the mosaic raster to disk
    with rasterio.open(dest_p, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Copy to last_merged
    shutil.copy(dest_p, copy_p) if copy else None

    # Log
    print("\ndest_p, copy_p\t:", dest_p, copy_p)
    print(
        '**TIME** Tif merge: \n\t\tTime\t: ' + str(np.round(time.time() - start_merge)) + "s.")
    return dest_p, copy_p

import rasterio
from indices import loads_DZ
from loaders.dataset.providers.raster_provider import Normalize
import geopandas as gpd
from sklearn.model_selection import train_test_split
from indices import loads_grid_ref


model_params = {
    "model_name": "model_69epoch",
    "model_path": "../../data/model/",
    'n_input': 102,
    'n_labels': 14129,
    'normalise_last_layer': True,
    'nb_gpus': 1,
}

raster_list = ['Railways_WGS84', 'Roads_WGS84',
               'NavWater1994_WGS84', 'NavWater2009_WGS84',
               'Pasture1993_WGS84', 'Pasture2009_WGS84',
               'croplands1992_WGS84', 'croplands2005_WGS84',
               'Lights1994_WGS84', 'Lights2009_WGS84',
               'Popdensity1990_WGS84', 'Popdensity2010_WGS84',
               'Built1994_WGS84', 'Built2009_WGS84',
               'ecoregions001',
               'bdod_0-5cm_mean_1000_WGS84', 'cec_0-5cm_mean_1000_WGS84', 'cfvo_0-5cm_mean_1000_WGS84',
               'clay_0-5cm_mean_1000_WGS84',
               'nitrogen_0-5cm_mean_1000_WGS84', 'ocd_0-5cm_mean_1000_WGS84', 'ocs_0-30cm_mean_1000_WGS84',
               'phh2o_0-5cm_mean_1000_WGS84',
               'sand_0-5cm_mean_1000_WGS84', 'silt_0-5cm_mean_1000_WGS84', 'soc_0-5cm_mean_1000_WGS84',
               'wc2.1_30s_bio_1', 'wc2.1_30s_bio_2', 'wc2.1_30s_bio_3', 'wc2.1_30s_bio_4', 'wc2.1_30s_bio_5',
               'wc2.1_30s_bio_6', 'wc2.1_30s_bio_7', 'wc2.1_30s_bio_8', 'wc2.1_30s_bio_9', 'wc2.1_30s_bio_10',
               'wc2.1_30s_bio_11', 'wc2.1_30s_bio_12', 'wc2.1_30s_bio_13', 'wc2.1_30s_bio_14', 'wc2.1_30s_bio_15',
               'wc2.1_30s_bio_16', 'wc2.1_30s_bio_17', 'wc2.1_30s_bio_18', 'wc2.1_30s_bio_19']

data_params = {
    "res": 1000,
    "size": 64,
    "use_rasters": True,
    "rasters": "../../data/predictors/",
    "raster_list": raster_list,
    "raster_transform": Normalize,
    "patch_transform": ["Normalize"],
    "use_geo": True,
    "geo_mode": "constant",

    "splitter": train_test_split,
    "strat_cls": 'lvl2_code',
    "cell_cls": "cell_index",
    "validation_size": 0.0,
    "test_size": 1.0,
    "SingleOccSpInTrain": True,

    "sep": ";",
    "latitude": 'decimallatitude',
    "longitude": 'decimallongitude',
    "id_name": 'gbifid',
    "label_name": None
}


export_params = {
    'bs_test' : 512,                # TO LOWER IF MEMORY OVERFLOW
    'size' :550,
    'bin_export': True,
    'buffer_size': 100,
    "with_labels": False,

    "index_path": "../../data/model/index.json",
    "save_index": False,
    "load_index": True
}


indices_params = {
    'T': 8.75e-5,
    'DZ': loads_DZ("../../data/occurrences/species_continents.json"),
    "ref_path": "../../data/occurrences/DeepOrchidSeries.csv",
    'lvl1_gdf': gpd.read_file("../../data/level1/level1.shp"),
    'iucn_path': "../../data/iucn_status/",
    "maps_path": "../../out/maps",
    "bin_path": "../../out/bin",
    "Lstats": ['LC', 'NT', 'VU', 'EN', 'CR'],
    "norm_shannon": False,
    "fill_sumap": 10,
    "dtype_sumap": rasterio.float32,
    "fill_cat": 10,
    "dtype_cat": rasterio.uint8,
    "tiled": False,
    'empty_name': "empty_preds",
    "empty_value": 255,
    "removes_oor": True,
    "norm_AbTprobas": True,
    "compress": None,
    "seconds": 30,
    "nb_workers": 4,
    # COG Params
    "levels": 6,
    "num_threads": "ALL_CPUS",
    "gdal_cachemax": 4096
}

def update_dicts(occs_file, indices_params, export_params):
    file_name = occs_file.split('/')[-1][:-4]
    indices_params["grid_name"] = file_name,
    indices_params["grid_ref"]  = loads_grid_ref(occs_file)
    export_params['name']       = file_name
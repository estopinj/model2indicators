from .modular_dataset import ModularDataset
from providers import GeoProvider, RasterProvider


class ModDatasetS2(ModularDataset):
    def __init__(self, labels, dataset, ids, data_params=None):

        # Retrieves data_params elements
        rasters, raster_list = data_params["rasters"], data_params["raster_list"]
        size, res = data_params["size"], data_params["res"]
        use_geo, use_rasters = data_params["use_geo"], data_params["use_rasters"]
        geo_mode = data_params["geo_mode"]
        raster_transform, patch_transform = data_params["raster_transform"], data_params["patch_transform"]

        # Potential rasters
        provider_geo     = GeoProvider(geo_mode=geo_mode, size=size, res=res, patch_transform=patch_transform) if use_geo else None
        provider_rasters = RasterProvider(rasters, raster_list=raster_list, size=size, res=res, patch_transform=patch_transform, raster_transform=raster_transform) if use_rasters else None

        providers = tuple(filter(None, (provider_geo, provider_rasters)))
        super().__init__(labels, dataset, ids, providers)
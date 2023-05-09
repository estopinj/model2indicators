import numpy as np

class GeoProvider(object):
    def __init__(self, geo_mode="constant", size=64, res=10, degrad_decimal_round=2, patch_transform=None):
        self.geo_mode = geo_mode
        self.size = size
        self.res  = res
        self.degrad_decimal_round = degrad_decimal_round
        self.patch_transform = patch_transform
        self.min_lat = -90
        self.max_lat = 90
        self.min_lon = -180
        self.max_lon = 180
        self.avg_lat = 29.50334
        self.std_lat = 35.58280
        self.avg_lon = 24.09806
        self.std_lon = 67.70266



    def Normalize(self, patch): # in between [-1,1]
        patch[0,:,:] = 2*(patch[0,:,:] - self.min_lat) / (self.max_lat - self.min_lat) -1
        patch[1,:,:] = 2*(patch[1,:,:] - self.min_lon) / (self.max_lon - self.min_lon) -1
        return patch

    def Standardize(self, patch): # in between [-1,1]
        patch[0,:,:] = (patch[0,:,:] - self.avg_lat) / self.std_lat
        patch[1,:,:] = (patch[1,:,:] - self.avg_lon) / self.std_lon
        return patch


    def __getitem__(self, item):            # item = (id_, latitude, longitude)
        gbifid = item[0]
        latitude = item[1]
        longitude = item[2]

        # Calculating geo_features
        try:
            geo_tensor = self.geo_features(latitude, longitude)
            # print("decreasing lat:", geo_tensor[0,:,0])
            # print("increasing lon:", geo_tensor[1,0,:])
        except Exception:
            geo_tensor = None
            print('geo features calculation from occurrence '+str(gbifid))

        if self.patch_transform is not None:
            if "Normalize" in self.patch_transform:
                geo_tensor = self.Normalize(geo_tensor)
            elif "Standardize" in self.patch_transform:
                geo_tensor = self.Standardize(geo_tensor)
        return geo_tensor


    def extent(self, dlat, dlon):
        """
        outputs (dlat_min, dlat_max, dlon_min, dlon_max) for a square of radius d around (dlat,dlon)
        """
        d = self.size*self.res//2/1000

        delta_lat = d / 111
        delta_lon = d / (111 * np.cos(int(dlat) * np.pi / 180))

        # Security if (dlat+-delta_lat) >90 or <-90
        dlat_min = (dlat - delta_lat) if (dlat - delta_lat) >= -90 else (dlat - delta_lat) % 90
        dlat_max = (dlat + delta_lat) if (dlat + delta_lat) <= 90 else (dlat + delta_lat) % -90

        # Security if (dlon+-delta_lon) >180 or <-180
        dlon_min = (dlon - delta_lon) if (dlon - delta_lon) >= -180 else (dlon - delta_lon) % 180
        dlon_max = (dlon + delta_lon) if (dlon + delta_lon) <= 180 else (dlon + delta_lon) % -180

        return dlat_min, dlat_max, dlon_min, dlon_max


    def geo_features(self, dlat, dlon):
        """
        Returns latitude and longitude patches concatenated with dimensions (2,size,size).
        lat patch is [0,:,:], lon patch is [1,:,:].
        if constant=True, dlat and dlon are juste repeated along each patch respectfully.
        Otherwise, a sliding effect is applied along the patch.
        """

        if self.geo_mode == "constant":
            # constant lat & lon across patches
            lat_patch = np.broadcast_to(dlat, (self.size, self.size))
            lon_patch = np.broadcast_to(dlon, (self.size, self.size))

        elif self.geo_mode == "degraded":
            # constant degraded lat & lon across patches
            lat_patch = np.broadcast_to(np.round(dlat, self.degrad_decimal_round), (self.size, self.size))
            lon_patch = np.broadcast_to(np.round(dlon, self.degrad_decimal_round), (self.size, self.size))

        elif self.geo_mode == "sliding":
            # Sliding lat & lon across patches
            e = self.extent(dlat, dlon)
            # lat patch
            lat_vec = np.linspace(e[1], e[0], num=self.size)
            lat_patch = np.broadcast_to(lat_vec, (lat_vec.shape[0], lat_vec.shape[0])).T

            # lon patch
            lon_vec = np.linspace(e[2], e[3], num=self.size)
            lon_patch = np.broadcast_to(lon_vec, (lon_vec.shape[0], lon_vec.shape[0]))
        else:
            print('Unknown geo_features mode provided:' + self.geo_mode)
            lat_patch, lon_patch = None, None

        return np.stack((lat_patch, lon_patch))

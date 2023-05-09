from default_params import model_params, export_params, indices_params, data_params, update_dicts
from loaders.occurrence_loader import occurrence_loader
from loaders.dataset import ModDatasetS2
from loaders.model_loader import load_model
from inception_env import InceptionEnv
from export import export_bigdata
from maps_merge import maps_merge_all


# *** Points to predict on ***
occurrences = "../../data/occurrences/grid_sample.csv"
# ****************************
update_dicts(occurrences, indices_params, export_params)


# Loads occurrences and patches
train, validation, test = occurrence_loader(ModDatasetS2, occurrences, data_params=data_params)

# Loads model
model = load_model(model_class=InceptionEnv, model_params=model_params)

# Exports tiffs by buffer
export_bigdata(model, test, export_params=export_params, indices_params=indices_params)

# Merge buffered tiffs in COG
maps_merge_all(indices_params=None, deletes_src_tifs=False)

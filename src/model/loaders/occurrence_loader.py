import pandas as pd
from util.split import perform_split


def occurrence_loader(dataset_class, occurrences=None, data_params=None, **kwargs):
    """
    :return: train, val and test set, pytorch ready
    """
    # Retrieves data_params elements
    sep = data_params["set"]
    id_name = data_params["id_name"]
    latitude, longitude = data_params["latitude"], data_params["longitude"]
    validation_size, test_size = data_params["validation_size"], data_params["test_size"]
    splitter = data_params["splitter"]


    # Loads occurrences
    df          = pd.read_csv(occurrences, sep=sep, header='infer', low_memory=False)
    df['label'] = -1 # No labels
    ids         = df[id_name].to_numpy()
    labels      = df['label'].to_numpy()
    dataset     = df[[latitude, longitude]].to_numpy()
    columns     = (labels, dataset, ids)

    # splitting train_temp test
    train_temp, test = perform_split(columns, test_size, splitter)
    # splitting train validation
    train, val = perform_split(train_temp, validation_size, splitter)


    # train set
    train = dataset_class(*train, data_params=data_params, **kwargs)
    # test set
    test = dataset_class(*test, data_params=data_params, **kwargs)
    # validation set
    validation = dataset_class(*val, data_params=data_params, **kwargs)
    return train, validation, test

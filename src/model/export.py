import torch
import pandas as pd
import time
import numpy as np
import os
import os.path
from loaders.util.index import get_index
from indices import out2df, appliesT, df2gdf, applies_oor_filter, normAbT, computes_empty_preds, status_table, grid2tif


def _export_bigdata(fc, fp, results, test_ids, indexed_labels, size, exp_count, bin_export=False, indices_params=None, start=None):
    """
    Exporting a batch of results
    :param offset:
    :param size:
    :param f:
    :param results:
    :param test:
    :param indexed_labels:
    :return:
    """

    results = np.concatenate(results)
    print("\n\t\t *** NEW export ***")
    print("results.shape\t:", results.shape)

    order = np.argsort(-results, axis=1)[:, :size]
    print("order.shape\t:", order.shape)
    output = []
    L_classes = []
    L_probas  = []

    for i, elmt in enumerate(order):
        _id = int(test_ids[exp_count + i])
        # print("i, _id:", i, _id)
        L_classes.append(_id)
        for j in elmt:
            proba = results[i][j]
            if indexed_labels is None:
                class_id = j
            else:
                if j in indexed_labels:
                    class_id = indexed_labels[j]
                else:
                    print(str(j) + " is not in indexed_labels.keys()")
                    class_id = None

            L_classes.append(class_id)
            L_probas.append(proba)
            output.append([_id, class_id, proba])                  # j+1 is not the rank but only the species position in the output file!

    if bin_export:
        np.asarray(L_classes, dtype=np.uint32).tofile(fc) #### np.float64 and not np.float32
        np.asarray(L_probas, dtype=np.single).tofile(fp)  #### np.float64 and not np.float32

    else:
        df = pd.DataFrame(np.array(output))
        df.to_csv(fc, header=False, index=False)

    # EXPORTED COUNT UPDATE
    exp_count += order.shape[0]

    print('\nPredictions saved, total exported: '+str(exp_count))
    if start is not None:
        print('**TIME** Output computation + Bin export: \n\t\tTime\t: ' + str(np.round(time.time() - start)) + "s.")

    # INDICES COMPUTATION
    start_indice = time.time()

    grid = out2df(output, indices_params["grid_ref"])
    print("out2df done.")
    print("Grid shape\t:", grid.shape, grid.columns, "\ngrid.head():\n", grid.head())

    # Applies T
    grid = appliesT(grid, indices_params)
    print("\tT applied.")

    # Converts to geodataframe
    grid = df2gdf(grid, indices_params)
    print("\t\tdf2gdf done.")
    print('**TIME** out2gdf only: \n\t\tTime\t: ' + str(np.round(time.time() - start_indice)) + "s.")

    # Filters out ooc species
    start_oor = time.time()
    grid = applies_oor_filter(grid, indices_params)
    print("\t\t\toor filter applied.")
    print('**TIME** oor filtering only: \n\t\tTime\t: ' + str(np.round(time.time() - start_oor)) + "s.")

    # Norm. species probas above T
    start_norm = time.time()
    grid = normAbT(grid, indices_params)
    print("\t\t\t\tNomalization done.")
    # Prepares empty prediction boolean mask
    grid = computes_empty_preds(grid, indices_params)
    print("\t\t\t\tEmpty preds boolean mask computed.")
    # Computes status table
    iucn_table = status_table(indices_params)
    print("\t\t\t\t\tIucn table done.")
    print('**TIME** Norm + Empty preds mask + IUCN table: \n\t\tTime\t: ' + str(np.round(time.time() - start_norm)) + "s.")

    # Verifies grid state
    print("grid shape and columns:\n", grid.shape, grid.columns)
    print("grid.head() before tifs:\n", grid.head())

    # Computes Maps
    print("\n\n\t\t*** GRID2TIF ***\n\n")
    start_tifs = time.time()
    for source in ['iucn', 'comp']:
        for var_type in ["B", "S", "cat"]:
            print(source, var_type, " begins...")
            tif_p = grid2tif(grid, indices_params, iucn_table=iucn_table, src=source, var_type=var_type, exp_count=exp_count)
            print("\t\t... ", source, var_type, " saved at:\n\t\t", tif_p)
            # tif merge
            # incremental_tif_merge(tif_p)

    # Shannon
    print("Shannon index begins...")
    tif_p = grid2tif(grid, indices_params, var_type="shannon", exp_count=exp_count)
    print("\t\t... Shannon index saved at:\n\t\t", tif_p)
    print(
        '**TIME** Global .tifs exports + merges: \n\t\tTime\t: ' + str(np.round(time.time() - start_tifs)) + "s.")

    return exp_count



def export_bigdata(model, test, export_params=None, indices_params=None):
    # Retrrieves arguments
    batch_size = export_params['bs_test']
    buffer_size= export_params["buffer_size"]
    size       = export_params["size"]          # Nb of predictions kept per point
    name       = export_params["name"]
    bin_export = export_params["bin_export"],

    # Global time start
    glob_start = time.time()
    # Workers
    num_workers = indices_params["nb_workers"]
    # Test data loader
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    # Test ids to reference predictions
    test_ids = list(test.ids)
    print("len(test_ids), test_ids[:10]\t:", len(test_ids), test_ids[:10])

    # Output file
    out_name    = 'preds_size'+str(size)+"_buff"+str(buffer_size)
    out_name   += "_" + name if name is not None else ""
    out_name_c, out_name_p = out_name+"_classes", out_name + "_probas"
    out_name_c += '.bin' if bin_export else '.csv'
    out_name_p += '.bin' if bin_export else '.csv'
    export_path_c, export_path_p = os.path.join(indices_params["bin_path"], out_name_c), os.path.join(indices_params["bin_path"], out_name_p)
    # Open mode
    open_mode   = 'ab' if bin_export else 'w'

    # check if labels have been indexed
    index_path = export_params["index_path"]
    indexed_labels = get_index(index_path)

    # Exported count Initialization
    exported_count = 0
    # *****************************

    with torch.no_grad():
        model.eval()
        results = []

        with open(export_path_c, open_mode) as fc, open(export_path_p, open_mode) as fp:
            print('Exporting predictions at ' + export_path_c + " AND " + export_path_p)
            fc.write('id,class_id,proba\n') if not bin_export else None  # header

            for idx, data in enumerate(test_loader):

                if len(results)==0:
                    start_pred = time.time()

                # get the inputs
                inputs, _ = data

                outputs = model(inputs)

                # Results list appending
                # print("outputs:", type(outputs)) # <class 'torch.Tensor'>
                results.append(outputs.detach().cpu().numpy())

                # Buffer export
                if len(results)*batch_size >= buffer_size:
                    print("\n\n\n\n\n\t*** Export idx =", idx, "***, Export #", str((idx+1)//98))
                    print('**TIME** Prediction time, idx: ' + str(idx) + "\n\tTime\t: " + str(np.round(time.time() - start_pred)) + "s.")

                    start_bin = time.time()
                    exported_count = _export_bigdata(fc, fp, results, test_ids, indexed_labels, size, exported_count, bin_export=bin_export, indices_params=indices_params, start=start_bin)

                    results = []
                    print(
                        "**TIME** Total buffer .bin exports + Indices comput. + .tif exports, \n\tTime\t: " + str(np.round(time.time() - start_bin)) + "s.")
                    print("\n---> END EXPORT, exported count:" + str(exported_count) + ",\n\t" + str(
                        100 * exported_count / len(indices_params["grid_ref"])) + " %")


            # Final batches export
            if len(results) >= 0:
                print("\t*** Last export ***")
                exported_count = _export_bigdata(fc, fp, results, test_ids, indexed_labels, size, exported_count, bin_export=bin_export, indices_params=indices_params)

                print("\n---> FINAL END EXPORT, exported count:" + str(exported_count) + ",\n\t" + str(
                    100 * exported_count / len(indices_params["grid_ref"])) + " %")

            print("**TIME** Global buffer time\n\tTime\t: " + str(np.round(time.time() - glob_start)) + "s.\n\n")

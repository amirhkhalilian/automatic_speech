import numpy as np

from matplotlib import pyplot as plt

from pyhypnos.utils import areas_rename

def region_analysis(corrcoef, filter_lags, regions):
    indices = np.where(~np.isnan(corrcoef))[0]
    corrcoef = corrcoef[indices]
    filter_lags = filter_lags[indices]
    regions = [regions[i] for i in indices]

    rename_map = {"STG": ["cSTG", "mSTG"],
                  "frontal": ["parsopercularis",
                              "parsorbitalis",
                              "parstriangularis",
                              "rostralmiddlefrontal",
                              "caudalmiddlefrontal"]}
    regions, counts = areas_rename(regions,
                                   rename_map=rename_map)

    rois = ['STG', 'frontal', 'precentral', 'postcentral']

    results = dict()
    for r, roi in enumerate(rois):
        inds = [i for i, val in enumerate(regions) if val==roi]
        results[roi+'_corr'] = corrcoef[inds]
        results[roi+'_time'] = filter_lags[inds]

    return results


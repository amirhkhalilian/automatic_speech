import os
import argparse
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from pyhypnos.loadingtools import filter_electrodes
from pyhypnos.loadingtools import remove_indices_from_coords
from pyhypnos.loadingtools import get_multiple_task_badelecs

from pyhypnos.data import EcogTaskPreProcessing

def dataloader_structured(subj, tasks,
                          flag_response_lock = False,
                          epoch_start = -384,
                          epoch_end = 384,
                          number_log_dist_bands = 1,
                          test_ratio = 0.0):
    if isinstance(epoch_start, int):
        epoch_start = [epoch_start]*len(tasks)
    if isinstance(epoch_end, int):
        epoch_end = [epoch_end]*len(tasks)
    # processing tools
    subj_dir = os.path.join(f'EcogData/{subj}')
    gdat_all = []
    zoom_all = []
    for ti, task in enumerate(tasks):
        ET = EcogTaskPreProcessing(subj, task,
                                   subj_dir = subj_dir,
                                   flag_response_lock = flag_response_lock,
                                   epoch_start = epoch_start[ti],
                                   epoch_end = epoch_end[ti],
                                   gdat_mode = 'gdat_CAR',
                                   use_jax = True,
                                   normalization_mode = 'MultiBandHG_Jax',
                                   number_log_dist_bands = number_log_dist_bands)
        # load ecog data -- trial x time x elec
        gdat_ep, coords = ET.get_ecogdata()
        inds2rm = filter_electrodes(coords)
        bads = get_multiple_task_badelecs(subj, subj_dir=subj_dir,
                                          tasks=['VisRead',
                                                 'SpeechOutput_C',
                                                 'SpeechOutput_D',
                                                 'SpeechOutput_M'])
        inds2rm = np.unique(np.concatenate([bads, inds2rm]))
        coords = remove_indices_from_coords(coords, inds2rm)
        inds2use = np.setdiff1d(np.arange(gdat_ep.shape[0]), inds2rm)
        gdat_ep = gdat_ep[inds2use,...]
        gdat_ep = np.transpose(gdat_ep, axes=[1,2,0])
        gdat_all.append(gdat_ep.copy())
        # load zoom data -- trial x time x 1 (mean over freq features)
        zoom_ep = ET.get_zoomdata()
        zoom_ep = zoom_ep.mean(axis=0, keepdims=True)
        zoom_ep = np.transpose(zoom_ep, axes=[1,2,0])
        zoom_all.append(zoom_ep.copy())
    # concatenate trials
    gdat_ep = np.concatenate(gdat_all, axis=0)
    zoom_ep = np.concatenate(zoom_all, axis=0)
    # convert data to list over trials
    if test_ratio == 0.:
        zoom_train = [zoom_ep[i,:,:].squeeze() for i in range(zoom_ep.shape[0])]
        zoom_test = []
        gdat_train = [gdat_ep[i,:,:].squeeze() for i in range(gdat_ep.shape[0])]
        gdat_test = []
    else:
        ntrials = gdat_ep.shape[0]
        indices = np.arange(ntrials)
        np.random.shuffle(indices)
        test_indices = indices[:int(ntrials*test_ratio)]
        train_indices = indices[int(ntrials*test_ratio):]
        zoom_train = [zoom_ep[i,:,:].squeeze() for i in train_indices]
        zoom_test = [zoom_ep[i,:,:].squeeze() for i in test_indices]
        gdat_train = [gdat_ep[i,:,:].squeeze() for i in train_indices]
        gdat_test = [gdat_ep[i,:,:].squeeze() for i in test_indices]
    return zoom_train, gdat_train, zoom_test, gdat_test, coords

def single_subj_loader(subj, tasks,
                       flag_response_lock=False,
                       epoch_start = -384,
                       epoch_end = 384,
                       number_log_dist_bands = 1,
                       test_ratio = 0.0):

    # loading structured tasks only
    Xr, Yr, Xs, Ys, coords = dataloader_structured(subj, tasks,\
                               flag_response_lock = flag_response_lock,\
                               epoch_start = epoch_start,\
                               epoch_end = epoch_end,\
                               number_log_dist_bands = number_log_dist_bands,
                               test_ratio = test_ratio)

    return Xr, Yr, Xs, Ys, coords


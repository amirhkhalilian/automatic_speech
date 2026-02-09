import os
import argparse

import numpy as np

from scipy.io import savemat

import pyvista as pv

from loaders import single_subj_loader

from encoding_model import training
from encoding_model import testing
from encoding_model import pearsonr_list
from encoding_model import model_filter_timing
from encoding_model import significant_electrodes

from plotting import plotting


def get_subj_task_from_index(idx):
    subjs = ['NY749', 'NY758', 'NY765',  'NY798', 'NY829', 'NY857', 'NY869']
    tasks = ['VisRead', ['SpeechOutput_C', 'SpeechOutput_D', 'SpeechOutput_M']]
    l1_ratio = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]

    hash_map = {}
    i = 1
    for subj, l1 in zip(subjs, l1_ratio):
        for task in tasks:
            if isinstance(task, list):
                if task[0].startswith('SpeechOutput'):
                    flag_response_lock = False
                else:
                    flag_response_lock = True
            else:
                if task.startswith('SpeechOutput'):
                    flag_response_lock = False
                else:
                    flag_response_lock = True
            hash_map[i] = (subj, task, l1, flag_response_lock)
            i += 1

    if idx not in hash_map:
        raise ValueError(f"Invalid index {idx}. Must be between 1 and {max(hash_map.keys())}.")
    return hash_map[idx]

def get_task_name(task):
    if isinstance(task, list):
        if len(task)>1:
            return task[0]+task[-1][-1]
        else:
            return task[0]
    else:
        return task

def single_subj_analysis(subj, tasks,
                         flag_response_lock = False,
                         epoch_start = -384,
                         epoch_end = 384,
                         params=dict(),
                         flag_plot=False,
                         save_dir="results_test"):
    default_params = {'tmin':-0.4,
                      'tmax':0.4,
                      'sfreq':512.,
                      'alpha':1.0,
                      'l1_ratio':1e-4}
    for key, value in default_params.items():
        if key not in params:
            params[key] = value
    # load data
    Xr, Yr, Xs, Ys, coords = single_subj_loader(subj, tasks,
                                flag_response_lock=flag_response_lock,
                                epoch_start = epoch_start,
                                epoch_end = epoch_end,
                                test_ratio=0.2)
    # train model
    model = training(Xr, Yr, params)
    corrcoef = testing(Xs, Ys, model)
    filter_lags = model_filter_timing(model)
    mni_coords, brain = get_projected_coords(coords)

    name = get_task_name(task)
    save_dir = os.path.join(save_dir, subj, name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subj}_{name}_encoding.npz")
    np.savez(save_path,
             subj=subj,
             task=task,
             corrcoef=corrcoef,
             filter_lags=filter_lags,
             mni_coords=mni_coords,
             coords=coords)

    # perform permutation
    obs_corr, null_corr, pvals, sig = significant_electrodes(Xr, Yr,
                                                             params,
                                                             Xs = Xs,
                                                             Ys = Ys,
                                                             n_perm=1000,
                                                             alpha=0.01)
    save_path = os.path.join(save_dir, f"{subj}_{name}_perm.npz")
    np.savez(save_path,
             subj=subj,
             task=task,
             corrcoef=corrcoef,
             filter_lags=filter_lags,
             mni_coords=mni_coords,
             coords=coords,
             obs_corr=obs_corr,
             null_corr=null_corr,
             pvals=pvals,
             sig=sig)

    return corrcoef, filter_lags, mni_coords, coords['elec_region']

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Lookup subject-task pair by index.")
    p.add_argument("--idx", type=int, required=True, help="Index for subject-task pair")
    args = p.parse_args()

    subj, task, l1_ratio, flag_response_lock = get_subj_task_from_index(args.idx)

    if isinstance(task, str):
        task = [task]

    print(f"working on:")
    print("subj: ", subj, "task: ", task, "flag: ", flag_response_lock)
    epoch_start = -384
    epoch_end = 384
    params = {'tmin':-0.4,
              'tmax':0.4,
              'sfreq':512.,
              'alpha':1.0,
              'l1_ratio':l1_ratio}
    single_subj_analysis(subj, task,
                         flag_response_lock=flag_response_lock,
                         epoch_start = epoch_start,
                         epoch_end = epoch_end,
                         params=params,
                         flag_plot=False)


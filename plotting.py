import os
import numpy as np

import matplotlib.pyplot as plt

import pyvista as pv

from mithra.visualtools import visualtools
from mithra.misc import project_elecs_pial

def get_mni_brain():
    brain = os.path.join('Mithra/SampleData/MNI-FS',
                         'FSL_MNI152_lh_pial.mat')
    annot = os.path.join('Mithra/SampleData/MNI-FS',
                         'FSL_MNI152.lh.aparc.split_STG_MTG.annot')
    return brain, annot

def get_projected_coords(coords):
    brain, annot = get_mni_brain()
    elec_locs_proj, _ = project_elecs_pial(coords['elec_MNI'],
                                           coords['elec_region'],
                                           brain, annot)
    return elec_locs_proj, brain

def plotting(corrcoef, filter_lags, mni_coords, brain, rm_nan=None):
    if rm_nan is None:
        indices = np.arange(len(corrcoef))
    else:
        indices = np.where(~np.isnan(corrcoef))[0]
    VT = visualtools(Subj='MNI', HS='lh', flag_UseAnnot=False, BrainFile=brain, BrainColor=[0.5,0.5,0.5,1.0])
    pl = pv.Plotter(shape=(1,2))
    pl.subplot(0,0)
    VT.PlotElecOnBrain(mni_coords[indices, :],
                       ElecColor=corrcoef[indices],
                       radius = 1.5,
                       cmap = 'Greens',
                       clim = [0,0.8],
                       BrainPlotter = pl,
                       scalar_bar_args = {'title':'r-value'},
                       show = False)
    pl.subplot(0,1)
    VT.PlotElecOnBrain(mni_coords[indices,:],
                       ElecColor = filter_lags[indices],
                       radius=1.5,
                       BrainPlotter = pl,
                       cmap = 'bwr',
                       clim = [-0.4,0.4],
                       scalar_bar_args = {'title':'time'},
                       show = False)
    pl.link_views()
    pv.set_plot_theme('paraview')
    pl.show()


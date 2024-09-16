# Script to perform FC analysis based on seed or parcellation to voxel correlations
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
import bct
from datetime import datetime
import glob
import gzip
import h5py
import importlib
import itertools
import joblib
from joblib import Parallel, delayed
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import nilearn
from nilearn import datasets
from nilearn.image import load_img, new_img_like, resample_to_img, binarize_img, iter_img, math_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_img, plot_roi, view_img
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
import nltools
import numpy as np
import os
import pickle
import pandas as pd
import pdb
import scipy
from scipy.io import loadmat
from scipy import ndimage
import seaborn as sbn
import shutil
import statsmodels
from statsmodels.stats import multitest
import sys
import time
from time import time
import platform
import warnings
warnings.filterwarnings('once')

from OCD_baseline.behavioral import ybocs_analysis

# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

# general paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCDbaseline'
lukeH_proj_dir = working_dir+'lab_lucac/lukeH/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
lukeH_deriv_dir = os.path.join(lukeH_proj_dir, 'data/derivatives')
atlas_dir = os.path.join(proj_dir, 'utils')

# This section should be replaced with propper packaging one day
sys.path.insert(0, os.path.join(code_dir, 'old'))
sys.path.insert(0, os.path.join(code_dir, 'utils'))
sys.path.insert(0, os.path.join(code_dir, 'structural'))
import qsiprep_analysis
import atlaser
from voxelwise_diffusion_analysis import cohen_d

# there you go:
#from ..old import qsiprep_analysis
#from ..utils import atlaser
#from ..structural.voxelwise_diffusion_analysis import cohen_d

from atlaser import Atlaser

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)

# Harrison 2009 seed locations:
seed_loc = {'AccL':[-9,9,-8], 'AccR':[9,9,-8], \
        'dPutL':[-28,1,3], 'dPutR':[28,1,3]}#, \
        #'vPutL':[-20,12,-3] , 'vPutR':[20,12,-3]} #, \
        #'dCaudL':[-13,15,9], 'dCaudR':[13,15,9]} #, \
        #'vCaudSupL':[-10,15,0], 'vCaudSupR':[10,15,0], \
        #'drPutL':[-25,8,6], 'drPutR':[25,8,6]}

# FC file suffix
seed_suffix = { 'Harrison2009': 'ns_sphere_seed_to_voxel',
                'TianS4':'seed_to_voxel'}

cohorts = ['controls', 'patients']

# pathwa masks induced from Shephard et al. 2021
pathway_mask = {'Acc':['OFC', 'PFClv', 'PFCv'],
                'dCaud':['PFCd_', 'PFCmp', 'PFCld_'],
                'dPut':['Ins', 'SomMotB_S2'], #'PFCld_''PFCl_',
                'vPut':['PFCl_', 'PFCm']} #'PFCd'

# cluster location for ortho-view visualization
cut_coords = {'Acc':[25,57,-6],
              'dCaud':None,
              'dPut':[50,11,19],
              'vPut':[-25,56,35]}

# frontal striatal pathway (single hemiphere, left only provided]) from Jaspers et al.
jaspers_files = {'Acc': 'ProbMap_R_N3_ventromed.nii',
                 'dPut': 'ProbMap_R_N3_putamen.nii',
                 'vPut': 'ProbMap_R_N3_putamen.nii',
                 'Caud': 'ProbMap_R_N3_caudate.nii'}

# fronto-striatal pathway based on Di Martino seeds, based on HCP
hcp_files = {'Acc': 'Acc_fcmap_fwhm6_HCP_REST1_avg.nii',#'nac_mask_fcmap_avg.nii.gz',
                 'dPut': 'dPut_fcmap_fwhm6_HCP_REST1_avg.nii', #'putamen_mask_fcmap_avg.nii.gz',
                 'vPut': 'vPut_fcmap_fwhm6_HCP_REST1_avg.nii', #'putamen_mask_fcmap_avg.nii.gz',
                 'dCaud': 'dCaud_fcmap_fwhm6_HCP_REST1_avg.nii'} #'caudate_mask_fcmap_avg.nii.gz'}

# results summary files for visualization. Note threshold for p-values is different when FWE-corrected (0.05) that uncorrected (0.005)
result_files = {'HCP': {
                    'Acc': {    'corrp': 'Acc_outputs_n5000_c35_group_by_hemi_Ftest_grp111_100hcpThr100SD_GM_23092022_clustere_corrp_tstat1.nii.gz',
                                'tstat': 'Acc_outputs_n5000_c35_group_by_hemi_Ftest_grp111_100hcpThr100SD_GM_23092022_tstat1.nii.gz',
                                'thr': 0.95 },
                    'dPut': {   'corrp': 'dPut_outputs_n5000_c30_group_by_hemi_Ftest_grp111_100hcpThr275SD_GM_22092022_clustere_corrp_tstat1.nii.gz',
                                'tstat': 'dPut_outputs_n5000_c30_group_by_hemi_Ftest_grp111_100hcpThr275SD_GM_22092022_tstat1.nii.gz',
                                'thr': 0.95 } },
                'Shephard': {
                    'Acc': {    'corrp': 'Acc_outputs_n5000_TFCE_group_by_hemi_Ttest_grp111_15022023_tfce_corrp_tstat2.nii.gz',#'Acc_outputs_n5000_TFCE_OCD_minus_HC_13062022_tfce_corrp_tstat1.nii.gz', #
                                'tstat': 'Acc_outputs_n5000_TFCE_OCD_minus_HC_13062022_tstat1.nii.gz',
                                'thr': 0.95 },
                    'dPut': {   'corrp': 'dPut_outputs_n5000_TFCE_group_by_hemi_Ttest_grp111_15022023_tfce_corrp_tstat1.nii.gz',#'dPut_outputs_n5000_TFCE_HC_minus_OCD_13062022_tfce_corrp_tstat1.nii.gz', #
                                'tstat': 'dPut_outputs_n5000_TFCE_HC_minus_OCD_13062022_tstat1.nii.gz',
                                'thr': 0.95 },
                    'vPut': {   'corrp': 'vPut_outputs_n5000_TFCE_HC_minus_OCD_13062022_tfce_p_tstat1.nii.gz',
                                'tstat': 'vPut_outputs_n5000_TFCE_HC_minus_OCD_13062022_tstat1.nii.gz',
                                'thr': 0.995 } } 
                }
                

# whether to use element-wise association between seeds and ROIs (zip) or all combinations (product)
combi = {'zip':zip, 'product':itertools.product}

voi_to_seed = {'OFC':'Acc', 'PFC':'dPut'}
seed_to_voi = {'Acc':'OFC', 'dPut':'PFC'}

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def get_cohort(subj):
    """ return cohort (control vs patient) of the subject """
    if 'control' in subj:
        return 'controls'
    elif 'patient' in subj:
        return 'patients'
    else:
        return 'none'


def get_x_sym(img):
    """ create a sagitally symetric image from the input img (assumed left hemisphere valued)
        ---
        used to create bilateral mask from unilateral maps given by Jaspers et al.
    """
    affine = img.affine

    def x_sym(inds):
        r_x = inds + affine[0,-1]/affine[0,0] - 1
        l_x = -r_x - affine[0,-1]/affine[0,0] - 1
        return l_x.astype(int)

    data = img.get_fdata().copy()
    inds_r = np.array(np.where(data))
    inds_l = np.apply_along_axis(x_sym, 0, inds_r)
    data[tuple(inds_l)] = data[tuple(inds_r)]
    nimg = new_img_like(img, data, copy_header=True)
    return nimg

def create_design_matrix(subjs, args):
    """ Create a more complex design matrix with group by hemisphere interactions """
    if args.group_by_hemi:
      n_con = np.sum(['control' in s for s in subjs])
      n_pat = np.sum(['patient' in s for s in subjs])
      if args.paired_design:
        design_mat = np.zeros((2*(n_con+n_pat),1+n_con+n_pat), dtype=int)
        design_mat[:2*n_con,0] = 1
        design_mat[-2*n_pat,0] = -1
        design_mat[:n_con,1:n_con+1] = np.eye(n_con)
        design_mat[n_con:2*n_con,1:n_con+1] = np.eye(n_con)
        design_mat[-n_pat:,n_con+1:] = np.eye(n_pat)
        design_mat[-2*n_pat:-n_pat,n_con+1:] = np.eye(n_pat)

        design_matrix = pd.DataFrame()
        design_matrix['group'] = design_mat[:,0]
        for i,subj in enumerate(subjs):
          design_matrix[subj] = design_mat[:,i+1]

      elif args.repeated2wayANOVA:
        design_mat = np.zeros((2*n_con+2*n_pat, n_con+n_pat+2))
        # subject means :
        design_mat[:n_con,:n_con] = np.eye(n_con)
        design_mat[n_con:2*n_con,:n_con] = np.eye(n_con)
        design_mat[2*n_con:2*n_con+n_pat,n_con:n_con+n_pat] = np.eye(n_pat)
        design_mat[2*n_con+n_pat:,n_con:n_con+n_pat] = np.eye(n_pat)
        # session main effect:
        design_mat[:n_con,-2] = 1
        design_mat[n_con:2*n_con,-2] = -1
        design_mat[-2*n_pat:-n_pat,-2] = 1
        design_mat[-n_pat:,-2] = -1
        # group by session interaction
        design_mat[:n_con,-1] = 1
        design_mat[n_con:2*n_con,-1] = -1
        design_mat[-2*n_pat:-n_pat,-1] = -1
        design_mat[-n_pat:,-1] = 1

        design_matrix = pd.DataFrame(design_mat)
      else:
        design_mat = np.zeros((2*(n_con+n_pat),4), dtype=int)

        design_mat[:n_con, 0] = 1 # HC_L
        design_mat[n_con:2*n_con, 1] = 1 # HC_R
        design_mat[-2*n_pat:-n_pat, 2] = 1 # OCD_L
        design_mat[-n_pat:, 3] = 1 # OCD_L

        design_matrix = pd.DataFrame()
        design_matrix['HC_L'] = design_mat[:,0]
        design_matrix['HC_R'] = design_mat[:,1]
        design_matrix['OCD_L'] = design_mat[:,2]
        design_matrix['OCD_R'] = design_mat[:,3]

    else:  # Create a simple group difference design matrix
        n_con = np.sum(['control' in s for s in subjs])
        n_pat = np.sum(['patient' in s for s in subjs])

        design_mat = np.zeros((n_con+n_pat,2), dtype=int)
        design_mat[:n_con,0] = 1
        design_mat[-n_pat:, 1] = 1

        design_matrix = pd.DataFrame()
        design_matrix['con'] = design_mat[:,0]
        design_matrix['pat'] = design_mat[:,1]

    return design_matrix

def get_voi_mask(voi, args):
    """ create VOI mask from corrected p-value contrast from randomise """
    if voi in voi_to_seed.keys():
        seed = voi_to_seed[voi]
        if args.unilateral_seed:
            print(f"Warning: randomise hasn't been run with unilateral seeds {seed}, using bilateral {seed} instead")    
            fname = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/', 'smoothed_but_sphere_seed_based', args.metrics[0],
                                'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm)), seed, 'randomise', result_files[args.deriv_mask][seed]['corrp'])
        else:
            fname = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/', 'smoothed_but_sphere_seed_based', args.metrics[0],
                                'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm)), seed, 'randomise', result_files[args.deriv_mask][seed]['corrp'])
        return binarize_img(fname, threshold=result_files[args.deriv_mask][seed]['thr'])
    elif voi in seed_to_voi.keys():
        if args.unilateral_seed:
            return nltools.create_sphere(seed_loc[voi+'R'], radius=3.5)
        else:
            L = nltools.create_sphere(seed_loc[voi+'L'], radius=3.5)
            R = nltools.create_sphere(seed_loc[voi+'R'], radius=3.5)
            mask = nilearn.masking.intersect_masks([L,R], threshold=0, connected=False) # thr=1 : intersection; thr=0 : union
            return mask
    else:
        return None



def seed_to_voxel(subj, seeds, metrics, atlases, smoothing_fwhm=8.):
    """ perform seed-to-voxel analysis of bold data -- based on ROIs (e.g. using Tian subcortical parcellation)"""
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for metric in metrics:
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
                                 subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        bold_img = nib.load(bold_file)
        brain_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm, standardize=True, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        for atlas in atlases:
            # prepare output file
            hfname = subj+'_task-rest_'+atlas+'_desc-'+metric+'_'+''.join(seeds)+'_seeds_ts.h5'
            hf = h5py.File(os.path.join(deriv_dir, 'post-fmriprep-fix', subj, 'timeseries' ,hfname), 'w')

            # get atlas utility
            atlazer = Atlaser(atlas)

            # extract seed timeseries and perform seed-to-voxel correlation
            for seed in seeds:
                seed_img = atlazer.create_subatlas_img(seed)
                seed_masker = NiftiLabelsMasker(seed_img, standardize='zscore')
                seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
                seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/seed_ts.shape[0]
                seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr.mean(axis=-1).T)
                fname = '_'.join([subj,atlas,metric,seed])+'_seed_to_voxel_corr.nii.gz'
                nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
                hf.create_dataset(seed+'_ts', data=seed_ts)
            hf.close()
    print('{} seed_to_voxel performed in {}s'.format(subj,int(time()-t0)))



def sphere_seed_to_voxel(subj, seeds, metrics, atlases=['Harrison2009'], args=None):
    """ perform seed-to-voxel analysis of bold data using Harrison2009 3.5mm sphere seeds"""
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for atlas,metric in itertools.product(atlases,metrics):
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        #bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
        #                       subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        bold_file = os.path.join(deriv_dir, 'post-fmriprep-fix', subj,'func', \
                                 subj+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz')
        bold_img = nib.load(bold_file)
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, standardize='zscore', verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for seed in seeds:
            seed_masker = NiftiSpheresMasker([np.array(seed_loc[seed])], radius=3.5, t_r=0.81, \
                                low_pass=0.1, high_pass=0.01, standardize='zscore', verbose=0)
            seed_ts = np.squeeze(seed_masker.fit_transform(bold_img))
            seed_to_voxel_corr = np.dot(voxels_ts.T, seed_ts)/voxels_ts.shape[0]
            seed_to_voxel_corr_img = brain_masker.inverse_transform(seed_to_voxel_corr)
            fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
            fname = '_'.join([subj,metric,fwhm,atlas,seed,'ns_sphere_seed_to_voxel_corr.nii.gz'])
            nib.save(seed_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} seed_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))

def voi_to_voxel(subj, vois, metrics, atlases=['Harrison2009'], args=None):
    """ perform voi-to-voxel analysis of bold data VOI seeds from the previous FC analysis """
    # prepare output directory
    out_dir = os.path.join(proj_dir, 'postprocessing', subj)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    t0 = time()

    for atlas,metric in itertools.product(atlases,metrics):
        # get bold time series for each voxel
        img_space = 'MNI152NLin2009cAsym'
        #bold_file = os.path.join(lukeH_deriv_dir, 'post-fmriprep-fix', subj,'func', \
        #                       subj+'_task-rest_space-'+img_space+'_desc-'+metric+'_scrub.nii.gz')
        bold_file = os.path.join(deriv_dir, 'post-fmriprep-fix', subj,'func', \
                                 subj+'_task-rest_space-'+img_space+'_desc-'+metric+'.nii.gz')
        bold_img = nib.load(bold_file)
        brain_masker = NiftiMasker(smoothing_fwhm=args.brain_smoothing_fwhm, t_r=0.81, \
            low_pass=0.1, high_pass=0.01, standardize='zscore', verbose=0)
        voxels_ts = brain_masker.fit_transform(bold_img)

        # extract seed timeseries and perform seed-to-voxel correlation
        for voi in vois:
            voi_img = get_voi_mask(voi, args)
            voi_masker = NiftiMasker(voi_img, t_r=0.81, low_pass=0.1, high_pass=0.01, verbose=0)
            voi_ts = np.mean(voi_masker.fit_transform(bold_img), axis=1)
            voi_ts = (voi_ts - np.mean(voi_ts))/np.std(voi_ts)
            voi_to_voxel_corr = np.dot(voxels_ts.T, voi_ts)/voxels_ts.shape[0]
            voi_to_voxel_corr_img = brain_masker.inverse_transform(voi_to_voxel_corr.squeeze())
            fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
            fname = '_'.join([subj,metric,fwhm,atlas,voi,'avg_seed_to_voxel_corr.nii.gz'])
            nib.save(voi_to_voxel_corr_img, os.path.join(out_dir, fname))
    print('{} voi_to_voxel correlation performed in {}s'.format(subj,int(time()-t0)))


def merge_LR_hemis(subjs, seeds, metrics, seed_type='sphere_seed_to_voxel', atlas='Harrison2009', args=None):
    """ merge the left and right correlation images for each seed in each subject """
    hemis = ['L', 'R']
    in_fnames = dict( ( ((seed,metric),[]) for seed,metric in itertools.product(seeds,metrics) ) )
    for metric in metrics:
        for i,seed in enumerate(seeds):
            for k,subj in enumerate(subjs):
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                fnames = [os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj,metric,fwhm,atlas,seed+hemi,'ns_sphere_seed_to_voxel_corr.nii.gz']))
                          for hemi in hemis]
                new_img = nilearn.image.mean_img(fnames)
                #fname = s+'_detrend_gsr_filtered_'+seed+'_sphere_seed_to_voxel_corr.nii'
                fname = '_'.join([subj,metric,fwhm,atlas,seed])+'_ns_sphere_seed_to_voxel_corr.nii'
                os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, coh), exist_ok=True)
                nib.save(new_img, os.path.join(args.in_dir, metric, fwhm, seed, coh, fname))
                in_fnames[(seed,metric)].append(os.path.join(args.in_dir, metric, fwhm, seed, coh, fname))
    print('Merged L-R hemishperes')
    return in_fnames


def prep_fsl_randomise(in_fnames, seeds, metrics, args):
    """ prepare 4D images for FSL randomise
    Note: deprecated. use of FSL randomise is now integrated in function use_randomise(...)
    """
    gm_mask = datasets.load_mni152_gm_mask()
    masker = NiftiMasker(gm_mask)

    ind_mask = False  # whether to apply GM mask to individual 3D images separatetely or already combined in 4D image
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    for metric,seed in itertools.product(metrics,seeds):
        if ind_mask:
            imgs = None
            for in_fname in in_fnames[(seed,metric)]:
                img = load_img(in_fname)
                masker.fit(img)
                masker.generate_report()
                masked_data = masker.transform(img)
                masked_img = masker.inverse_transform(masked_data)
                if imgs==None:
                    imgs = img
                else:
                    imgs = nilearn.image.concat_imgs([imgs, masked_img], auto_resample=True)
        else:
            img = nilearn.image.concat_imgs(in_fnames[(seed,metric)], auto_resample=True, verbose=0)
            masker.fit(img)
            masker.generate_report()
            masked_data = masker.transform(img)
            imgs = masker.inverse_transform(masked_data) #nib.Nifti1Image(masked_data, img.affine, img.header)
        nib.save(imgs, os.path.join(args.in_dir, metric, fwhm, 'masked_resampled_pairedT4D_'+seed))

    # saving design matrix and contrast
    design_matrix = create_design_matrix(subjs)
    design_con = np.hstack((1, -1)).astype(int)
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_mat'), design_matrix, fmt='%i')
    np.savetxt(os.path.join(args.in_dir, metric, fwhm, 'design_con'), design_con, fmt='%i', newline=' ')


def unzip_correlation_maps(subjs, metrics, atlases, seeds, args):
    """ extract .nii files from .nii.gz and put them in place for analysis with SPM (not used if only analysing with nilearn) """
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    [os.makedirs(os.path.join(args.in_dir, metric, fwhm, seed, coh), exist_ok=True) \
            for metric,seed,coh in itertools.product(metrics,seeds,cohorts)]

    print('Unzipping seed-based correlation maps for use in SPM...')

    for subj,metric,atlas,seed in itertools.product(subjs,metrics,atlases,seeds):
        fname = '_'.join([subj,metric,fwhm,atlas,seed])+'_ns_sphere_seed_to_voxel_corr.nii.gz'
        infile = os.path.join(proj_dir, 'postprocessing', subj, fname)
        if 'control' in subj:
            coh = 'controls'
        else:
            coh = 'patients'
        with gzip.open(infile, 'rb') as f_in:
            with open(os.path.join(args.in_dir, metric, fwhm, seed, coh, fname[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_subjs_after_scrubbing(subjs, metrics, min_time=5):
    """ return list of subjects (in dataframe) with more than min_time left after scrubbing.
        Also returns the subjects discarded.
    """
    scrub_key = 'scrubbed_length_min'
    scrub_thr = min_time
    proc_dir = 'post-fmriprep-fix'
    d_dir = deriv_dir #lukeH_deriv_dir

    revoked = []
    for subj,metric in itertools.product(subjs, metrics):
        fname = 'fmripop_'+metric+'_parameters.json'
        fpath = os.path.join(d_dir, proc_dir, subj, 'func', fname)
        with open(fpath, 'r') as f:
            f_proc = json.load(f)
            if f_proc[scrub_key] < scrub_thr:
                print("{} has less than {:.2f} min of data left after scrubbing, removing it..".format(subj, f_proc[scrub_key]))
                revoked.append(subj)

    rev_inds = [np.where(s==subjs)[0][0] for s in revoked]
    subjs = subjs.drop(rev_inds)
    return subjs, revoked


def create_local_sphere_within_cluster(vois, rois, metrics, args=None, sphere_radius=3.5):
    """ create a sphere VOIs of given radius within cluster VOIs (for DCM analysis) """
    max_locals = dict( ( ( (roi,metric) , {'controls':[], 'patients':[]}) for roi,metric in itertools.product(rois, metrics) ) )
    for metric in metrics:
        for roi,voi in zip(roi,vois):
            for subj in subjs:
                if 'control' in subj:
                    coh = 'controls'
                else:
                    coh = 'patients'

                fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
                corr_file = '_'.join([subj,metric,fwhm,atlas,roi,'ns_sphere_roi_to_voxel_corr.nii'])
                corr_fname = os.path.join(corr_path, roi, coh, corr_file)
                img = load_img(corr_fname)
                mask = load_img(proj_dir+'/postprocessing/SPM/rois_and_rois/'+voi+'.nii')
                mask = nilearn.image.new_img_like(img, mask.get_fdata())
                new_data = np.abs(img.get_fdata()) * mask.get_fdata()
                local_max = new_data.max()
                max_coords = np.where(new_data>=local_max)
                local_max_coords = nilearn.image.coord_transform(max_coords[0], max_coords[1], max_coords[2], img.affine)
                new_mask = nltools.create_sphere(local_max_coords, radius=sphere_radius)
                out_path = os.path.join(proj_dir, 'postprocessing', subj, 'spm', 'masks')
                os.makedirs(out_path, exist_ok=True)
                fname = os.path.join(out_path, 'local_'+voi+'_'+metric+'.nii')
                nib.save(new_mask, fname)
                max_locals[roi,metric][coh].append(new_mask)
    return max_locals


def resample_masks(masks):
    """ resample all given masks to the affine of the first in list """
    ref_mask = masks[0]
    out_masks = [ref_mask]
    for mask in masks[1:]:
        out_masks.append(resample_to_img(mask, ref_mask, interpolation='nearest'))
    return out_masks

def inflate_mask(mask, args):
    """ inflate mask using a kernel of n_vox voxels """
    data = mask.get_fdata().copy()
    n = args.n_inflate + 1
    kernel = np.ones((n, n, n))
    new_data = scipy.signal.convolve(data, kernel, method='direct', mode='same')
    new_mask = binarize_img(new_img_like(mask, np.abs(new_data)))
    return new_mask

def mask_imgs(flist, masks=[], seed=None, args=None):
    """ mask input images using intersection of template masks and pre-computed within-groups union mask """
    # mask images to improve SNR
    t_mask = time()
    if args.use_gm_mask:
        gm_mask = datasets.load_mni152_gm_mask()
        masks.append(binarize_img(gm_mask))
    if args.use_fspt_mask: ## not sure it works fine
        fspt_mask = load_img(os.path.join(proj_dir, 'utils', 'Larger_FrStrPalThal_schaefer400_tianS4MNI_lps_mni.nii'), dtype=np.float64)
        masks.append(binarize_img(fspt_mask))
    if args.use_cortical_mask:
        ctx_mask = load_img(os.path.join(proj_dir, 'utils', 'schaefer_cortical.nii'), dtype=np.float64)
        masks.append(binarize_img(ctx_mask))
    if args.use_frontal_mask:
        Fr_node_ids, _ = qsiprep_analysis.get_fspt_Fr_node_ids('schaefer400_tianS4')
        atlazer = atlaser.Atlaser(atlas='schaefer400_tianS4')
        Fr_img = atlazer.create_brain_map(Fr_node_ids, np.ones([len(Fr_node_ids),1]))
        masks.append(binarize_img(Fr_img))
    if args.use_seed_specific_mask:
        atlazer = atlaser.Atlaser(atlas='schaefer400_tianS4')
        frontal_atlas = atlazer.create_subatlas_img(rois=pathway_mask[seed])
        masks.append(binarize_img(frontal_atlas))
    if args.use_jaspers_mask:
        fname = os.path.join(proj_dir, 'utils', 'ProbabilityMaps_CorticoStriatalConnectivity_RightStriatum', jaspers_files[seed])
        mask = binarize_img(fname, threshold=args.jaspers_thr)
        masks.append(get_x_sym(mask))
    if args.use_hcp_mask:
        fname = os.path.join(proj_dir, 'utils', hcp_files[seed])
        img = load_img(fname)
        data = img.get_fdata().copy()
        data[np.isnan(data)] = 0
        data = data[data!=0]
        sigma = np.std(data)
        mask = binarize_img(fname, threshold=args.hcp_sd_thr*sigma)
        masks.append(mask)
    if masks != []:
        masks = resample_masks(masks)
        mask = nilearn.masking.intersect_masks(masks, threshold=1, connected=False) # thr=1 : intersection; thr=0 : union
        if args.inflate_mask:
            mask = inflate_mask(mask, args)
        masker = NiftiMasker(mask)
        masker.fit(imgs=list(flist))
        masker.generate_report() # use for debug
        masked_data = masker.transform(imgs=flist.tolist())
        imgs = masker.inverse_transform(masked_data)
        imgs = list(iter_img(imgs))  # 4D to list of 3D
    else:
        imgs = list(flist)
        masker=None
        mask = None
    print('Masking took {:.2f}s'.format(time()-t_mask))
    return imgs, masker, mask

def perform_second_level_analysis(seed, metric, design_matrix, cohorts=['controls', 'patients'], args=None, masks=[]):
    """ Perform second level analysis based on seed-to-voxel correlation maps
        Note: for use of non-parametric approach (randomise) rather than paramteric, see use_randomise(...)
    """
    # naming convention in file system
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))

    # get images path
    con_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'patients', '*'))
    flist = np.hstack([con_flist, pat_flist])

    # remove revoked subjects
    if args.revoked != []:
        flist = [l for l in flist if ~np.any([s in l for s in revoked])]

    imgs, masker, mask = mask_imgs(flist, masks=masks, seed=seed, args=args)

    # perform analysis
    t_glm = time()
    glm = SecondLevelModel(mask_img=masker)
    glm.fit(imgs, design_matrix=design_matrix)
    print('GLM fitting took {:.2f}s'.format(time()-t_glm))

    contrasts = dict()
    t0 = time()
    contrasts['within_con'] = glm.compute_contrast(np.array([1, 0]), output_type='all')
    t1 = time()
    contrasts['within_pat'] = glm.compute_contrast(np.array([0, 1]), output_type='all')
    t2 =  time()
    contrasts['between'] = glm.compute_contrast(np.array([1, -1]), output_type='all')
    print('within groups and between group contrasts took {:.2f}, {:.2f} and {:.2f}s'.format(t1-t0, t2-t1, time()-t2))
    n_voxels = np.sum(nilearn.image.get_data(glm.masker_.mask_img_))
    params = glm.get_params()
    return contrasts, n_voxels, params

def threshold_contrast(contrast, height_control='fpr', alpha=0.005, cluster_threshold=10):
    """ cluster threshold contrast at alpha with height_control method for multiple comparisons """
    thresholded_img, thresh = threshold_stats_img(
        contrast, alpha=alpha, height_control=height_control, cluster_threshold=cluster_threshold)
    cluster_table = get_clusters_table(
        contrast, stat_threshold=thresh, cluster_threshold=cluster_threshold,
        two_sided=True, min_distance=5.0)
    return thresholded_img, thresh, cluster_table

def create_within_group_mask(subroi_glm_results, args):
    """ create within group masks to use for between group contrasts to improve SNR """
    con_img, con_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_con']['z_score'],
                                    cluster_threshold=100, alpha=args.within_group_threshold)
    con_mask = binarize_img(con_img, threshold=con_thr)
    pat_img, pat_thr, c_table = threshold_contrast(subroi_glm_results['first_pass', 'contrasts']['within_pat']['z_score'],
                                    cluster_threshold=100, alpha=args.within_group_threshold)
    pat_mask = binarize_img(pat_img, threshold=pat_thr)
    mask = nilearn.masking.intersect_masks([con_mask, pat_mask], threshold=1, connected=False) # thr=1: intersection; thr=0: union
    return mask, con_mask, pat_mask


def run_second_level(subjs, metrics, subrois, args):
    """ Run second level analysis (parametric approach).
        Can be done in one or two passes, depending of what masking strategy used.
        One pass: independent masks (i.e. HCP or Shephard derived)
        Two passes: compute second level mask based on group-level stats (a la Harrison et al. 2009)
    """
    design_matrix = create_design_matrix(subjs)

    glm_results = dict() # to store the outputs
    for metric,subroi in itertools.product(metrics,subrois):
        print('Starting 2nd level analysis for '+subroi+' subroi.')
        t0 = time()
        glm_results[subroi] = dict()

        t_fp = time()
        contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args, masks=[])
        glm_results[subroi]['first_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['first_pass','params'] = contrasts, n_voxels, params
        print('{} first pass in {:.2f}s'.format(subroi,time()-t_fp))

        passes = ['first_pass']

        if args.use_within_group_mask:
            t_wmask = time()
            within_group_mask, con_mask, pat_mask = create_within_group_mask(glm_results[subroi], args)
            glm_results[subroi]['within_group_mask'], glm_results[subroi]['con_mask'], glm_results[subroi]['pat_mask'] = within_group_mask, con_mask, pat_mask
            print('created within groups mask in {:.2f}s'.format(time()-t_wmask))

            t_sp = time()
            contrasts, n_voxels, params = perform_second_level_analysis(subroi, metric, design_matrix, args=args, masks=[within_group_mask])
            glm_results[subroi]['second_pass','contrasts'], glm_results[subroi]['n_voxels'], glm_results[subroi]['second_pass','params'] = contrasts, n_voxels, params
            print('{} second pass in {:.2f}s'.format(subroi,time()-t_sp))

            passes.append('second_pass')

        # extracts and prints stats, clusters, etc.
        t_thr = time()
        for pss in passes:
            glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresholded_img')], \
                glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')], \
                glm_results[subroi][(pss,'fpr',args.fpr_threshold,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'])

            print(' '.join([subroi,pss,'clusters at p<{:.3f} uncorrected:'.format(args.fpr_threshold)]))
            print(glm_results[subroi][(pss,'fpr',args.fpr_threshold,'cluster_table')])

            glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresholded_img')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')], \
                glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')] = threshold_contrast( \
                                glm_results[subroi][pss,'contrasts']['between']['z_score'], height_control='fdr', alpha=args.fdr_threshold)

            print(' '.join([subroi,pss,'clusters at p<{:.2f} FDR corrected:'.format(args.fdr_threshold)]))
            print(glm_results[subroi][(pss,'fdr',args.fdr_threshold,'cluster_table')])

        print('Thresholding and clustering took {:.2f}s'.format(time()-t_thr))

        # show contrasts with significant clusters
        if args.plot_figs:
            t_plt = time()
            for pss in passes:
                fig = plt.figure(figsize=[16,4])
                ax1 = plt.subplot(1,2,1)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')],
                                axes=ax1, title='_'.join([pss,subroi,'contrast_fpr'+str(args.fpr_threshold)]))
                ax2 = plt.subplot(1,2,2)
                plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fdr',args.fdr_threshold,'thresh')],
                                axes=ax2, title='_'.join([pss,subroi,'contrast_fdr'+str(args.fdr_threshold)]))
                print('{} plotting took {:.2f}s'.format(subroi,time()-t_plt))

                if args.save_figs:
                    plot_stat_map(glm_results[subroi][pss,'contrasts']['between']['stat'], draw_cross=False, threshold=glm_results[subroi][(pss,'fpr',args.fpr_threshold,'thresh')],
                    output_file=os.path.join(args.out_dir,subroi+'_'+pss+'_contrast_fpr{:.3f}.pdf'.format(args.fpr_threshold)))

        print('Finished 2nd level analysis for '+subroi+' ROI in {:.2f}s'.format(time()-t0))

    # savings
    if args.save_outputs:
        suffix = '_'+metric
        if args.min_time_after_scrubbing!=None:
            suffix += '_minLength'+str(int(args.min_time_after_scrubbing*10))
        if args.use_fspt_mask:
            suffix += '_fsptMask'
        if args.use_cortical_mask:
            suffix += '_corticalMask'
        today = datetime.datetime.now().strftime("%Y%m%d")
        suffix += '_'+today
        with gzip.open(os.path.join(args.out_dir,'glm_results'+suffix+'.pkl.gz'), 'wb') as of:
            pickle.dump(glm_results, of)

    return glm_results



def compute_voi_corr(subjs, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ compute correlation between seed and VOI for each pathway, to extract p-values, effect size, etc. """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            if 'control' in subj:
                cohort = 'controls'
            else:
                cohort = 'patients'
            for seed,voi in combi[args.combi](seeds, vois):
                # load correlation map
                fname = '_'.join([subj, metric, fwhm, atlas, seed, 'ns_sphere_seed_to_voxel_corr.nii'])
                corr_map = load_img(os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed',
                                    metric, fwhm, seed, cohort, fname))
                # load voi mask
                voi_mask = get_voi_mask(voi, args)
                voi_mask = resample_to_img(voi_mask, corr_map, interpolation='nearest')

                # extract correlations
                voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                avg_corr = np.mean(voi_corr[voi_corr!=0])
                df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'pathway':'_'.join([seed,voi]), 'corr':avg_corr}
                dfs.append(df_line)
    df_voi_corr = pd.DataFrame(dfs)
    return df_voi_corr


def plot_voi_corr(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ violinplots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.52, 0.52]
    #plt.subplot(1,2,1)
    #sbn.violinplot(data=df_voi_corr, y='corr', x='pathway', hue='cohort', orient='v', split=True, scale_hue=True,
    #               inner='quartile', dodge=True, width=0.8, cut=0)

    fig = plt.figure(figsize=[4*len(seeds)*len(vois),6])
    for i,(seed,voi) in enumerate(combi[args.combi](seeds, vois)):
      ax = plt.subplot(1,len(seeds)*len(vois),i+1)
      #sbn.barplot(data=df_voi_corr[df_voi_corr['pathway']=='_'.join([seed,voi])], y='corr', x='pathway', hue='cohort', orient='v')
      sbn.boxplot(data=df_voi_corr[df_voi_corr['pathway']=='_'.join([seed,voi])], y='corr', x='pathway', hue='cohort', orient='v', dodge=True, showfliers=False)
      sbn.swarmplot(data=df_voi_corr[df_voi_corr['pathway']=='_'.join([seed,voi])], y='corr', x='pathway', hue='cohort', orient='v', dodge=True, alpha=0.7, edgecolor='black', linewidth=1)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      ax.set_ylim(ylim)
      if i==(len(seeds)*len(vois)-1):
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      else:
        ax.get_legend().set_visible(False)
      plt.tight_layout()

    if args.save_figs:
        figname = 'seeds_to_vois_FC_'+args.deriv_mask+'_'+args.combi+'_'+datetime.now().strftime('_%d%m%Y')+'.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))

def get_df_behav():
    """ import behavioral scores """
    with open(os.path.join(proj_dir, 'postprocessing', 'df_con.pkl'), 'rb') as f:
        df_con = pickle.load(f)
    with open(os.path.join(proj_dir, 'postprocessing', 'df_pat.pkl'), 'rb') as f:
        df_pat = pickle.load(f)
    return df_con, df_pat


def plot_fc_ybocs(df_con, df_pat, title):
    """ plot fc vs ybocs """
    plt.figure()
    #sbn.scatterplot(df, x='YBOCS_Total', y='corr', hue='cohort', title=title)
    plt.scatter(df_con['YBOCS_Total'], df_con['corr'], color='blue')
    plt.scatter(df_pat['YBOCS_Total'], df_pat['corr'], color='orange')
    plt.title(title)
    plt.show()

def print_voi_stats(df_voi_corr, seeds = ['Acc', 'dPut', 'vPut'], vois = ['lvPFC_R', 'lPFC_R', 'dPFC_L'], args=None):
    """ print seed to VOI stats """
    print('Seed to VOI statistics:\n-------------------------')
    df_con, df_pat = get_df_behav()
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
        seeds_vois = combi[args.combi](seeds, vois)
        for seed,voi in seeds_vois:
            key = '_'.join([seed, voi])
            df_voi_con = df_voi_corr.loc[ (df_voi_corr['cohort']=='controls')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key) ]
            df_voi_pat = df_voi_corr.loc[ (df_voi_corr['cohort']=='patients')
                                    & (df_voi_corr['atlas']==atlas)
                                    & (df_voi_corr['metric']==metric)
                                    & (df_voi_corr['pathway']==key) ]
            t,p = scipy.stats.ttest_ind(df_voi_con['corr'], df_voi_pat['corr'])
            d = cohen_d(df_voi_con['corr'], df_voi_pat['corr'])
            print("{} {} {} {}   T={:.3f}   p={:.3f}   cohen's d={:.2f}".format(atlas,metric,fwhm,key,t,p,d))

            df_voi_con = df_voi_con.dropna().merge(df_con)
            r,p = scipy.stats.pearsonr(df_voi_con['corr'], df_voi_con['YBOCS_Total'])
            print("FC-YBOCS controls {}: r={:.3f}, p={:.3f}".format(key,r,p))
            
            df_voi_pat = df_voi_pat.dropna().merge(df_pat)
            r,p = scipy.stats.pearsonr(df_voi_pat['corr'], df_voi_pat['YBOCS_Total'])
            print("FC-YBOCS patients {}: r={:.3f}, p={:.3f}".format(key, r,p))

            if args.plot_figs:
                ybocs_analysis.plot_corr_voi(df_voi_con, args)
                ybocs_analysis.plot_corr_voi(df_voi_pat, args)




def compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, seed, args):
    """ reconstruct within-group masks using non-parametric inference """
    # controls
    imgs, masker, mask = mask_imgs(con_flist, masks=masks, seed=seed, args=args)
    neg_log_pvals_within_con = non_parametric_inference(list(np.sort(con_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(con_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_con_mask = binarize_img(neg_log_pvals_within_con, threshold=0.) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    # patients
    imgs, masker, mask = mask_imgs(pat_flist, masks=masks, seed=seed, args=args)
    neg_log_pvals_within_pat = non_parametric_inference(list(np.sort(pat_flist)),
                                 design_matrix=pd.DataFrame(np.ones([len(pat_flist),1])),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_within_group, mask=masker, n_jobs=10, verbose=1)
    within_pat_mask = binarize_img(neg_log_pvals_within_pat, threshold=0.) # -log(p)>1.3 corresponds to p<0.05 (p are already bonferroni corrected)
    within_groups_mask = nilearn.masking.intersect_masks([within_con_mask, within_pat_mask], threshold=0, connected=False) # thr=0:union, thr=1:intersection
    return within_groups_mask


def non_parametric_analysis(subjs, seed, metric, pre_metric, masks=[], args=None):
    """ Performs a non-parametric inference in nilearn
        Note: this was experimental, worked but the extraction of stats and clusters do not
        follow standards widely used in the imaging community.
    """
    post_metric = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/', pre_metric, metric, post_metric)

    # create imgs file list
    con_flist = glob.glob(os.path.join(in_dir, seed, 'controls', '*'))
    pat_flist = glob.glob(os.path.join(in_dir, seed, 'patients', '*'))
    pat_flist = [pf for pf in pat_flist if 'sub-patient16' not in pf]
    flist = np.hstack([con_flist, pat_flist])

    design_matrix = create_design_matrix(subjs)

    if args.use_SPM_mask:
        # template masking (& SPM within-group masks) need to be set manually via commenting others
        mask = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, seed, 'scrub_out_1/'+seed+'_fl_0001unc_005fwe.nii.gz') # within-group
        #fspt_mask = binarize_img(os.path.join(proj_dir, 'utils', 'Larger_FrStrPalThal_schaefer100_tianS1MNI_lps_mni.nii.gz')) # fronto-striato-pallido-thalamic (fspt) mask
        #mask = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, seed, 'scrub_out_1/'+seed+'_fl_0001unc_and_FrStrPalThal_mask.nii') # fspt & within-group
        #gm_mask = os.path.join(proj_dir, 'utils', 'schaefer_cortical.nii.gz') # gray matter only
    elif args.use_within_group_mask:
        # create within-group masks non-parametrically
        mask = compute_non_parametric_within_groups_mask(con_flist, pat_flist, design_matrix, masks, seed, args)
    else:
        imgs, masker, mask = mask_imgs(flist, masks=masks, seed=seed, args=args)

    # between-groups non-parametric inference
    neg_log_pvals_permuted_ols_unmasked = \
        non_parametric_inference(list(np.sort(flist)),
                                 design_matrix=pd.DataFrame(design_matrix['con'] - design_matrix['pat']),
                                 model_intercept=True, n_perm=args.n_perm,
                                 two_sided_test=args.two_sided_between_group, mask=mask, n_jobs=10, verbose=1)

    return neg_log_pvals_permuted_ols_unmasked, mask

def plot_non_param_maps(neg_log_pvals, seed, suffix, args=None):
    """ plot and save outputs of the non-parametric analysis """
    if args.plot_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False) #cut_coords=[-24,55,34]
    if args.save_figs:
        plot_stat_map(neg_log_pvals, threshold=0.2, colorbar=True, title=' '.join([seed,suffix]), draw_cross=False,
                      output_file=os.path.join(proj_dir, 'img', '_'.join([seed,suffix])+'.pdf')) #cut_coords=[-24,55,34]


def compute_FC_within_masks(subjs, np_results, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ compute FC within masks used for the between-group analysis """
    dfs = []
    fwhm = 'brainFWHM{}mm'.format(int(args.brain_smoothing_fwhm))
    for atlas,metric in itertools.product(args.atlases, args.metrics):
        for subj in subjs:
            if 'control' in subj:
                cohort = 'controls'
            else:
                cohort = 'patients'
            for seed in seeds:
                # load correlation map
                fname = '_'.join([subj, metric, fwhm, atlas, seed, 'ns_sphere_seed_to_voxel_corr.nii'])
                corr_map = load_img(os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/Harrison2009Rep/seed_not_smoothed',
                                    metric, fwhm, seed, cohort, fname))
                voi_mask = resample_to_img(np_results[seed]['mask'], corr_map, interpolation='nearest')

                # extract correlations
                voi_corr = corr_map.get_fdata().copy() * voi_mask.get_fdata().copy()
                for corr in np.ravel(voi_corr[voi_corr!=0]):
                    df_line = {'subj':subj, 'metric':metric, 'atlas':atlas, 'fwhm':fwhm, 'cohort':cohort, 'seed':seed, 'corr':corr}
                    dfs.append(df_line)
    df_mask_corr = pd.DataFrame(dfs)
    return df_mask_corr

def plot_within_mask_corr(df_mask_corr, seeds = ['Acc', 'dPut', 'vPut'], args=None):
    """ bar plots of FC in pahtways """
    colors = ['lightgrey', 'darkgrey']
    sbn.set_palette(colors)
    plt.rcParams.update({'font.size': 20, 'axes.linewidth':2})
    ylim = [-0.15, 0.3]
    fig = plt.figure(figsize=[12,6])
    df_mask_corr['corr'] = df_mask_corr['corr'] / 880.
    df_mask_corr['corr'].loc[df_mask_corr['corr']>1] = 1
    df_mask_corr['corr'].loc[df_mask_corr['corr']<-1] = -1

    for i,seed in enumerate(seeds):
      ax = plt.subplot(1,len(seeds),i+1)
      #sbn.barplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v')
      #sbn.swarmplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', dodge=True, size=0.1)
      sbn.violinplot(data=df_mask_corr[df_mask_corr['seed']==seed], y='corr', x='seed', hue='cohort', orient='v', split=True, scale_hue=True,
                     inner='quartile', dodge=True, width=0.8, cut=1)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.tick_params(width=2)
      if i==len(seeds)-1:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      else:
        ax.get_legend().set_visible(False)
      plt.tight_layout()

    if args.save_figs:
        figname = 'seed_to_mask_corr_3seeds.svg'
        plt.savefig(os.path.join(proj_dir, 'img', figname))


def get_file_lists(subjs, seed, metric, atlas, args):
    """ returns 3 file lists corresponding to controls, patients, and combined
    controls+patients paths of imgs to process """
    # naming convention in file system
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    con_flist = []
    pat_flist = []
    # get images path
    if args.group_by_hemi:
        for hemi in ['L', 'R']:
            cl = np.sort(glob.glob(os.path.join(args.in_dir, metric, fwhm, seed+hemi, 'controls', '*')))
            con_flist.append(cl)
            pl = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed+hemi, 'patients', '*'))
            pat_flist.append(pl)
        con_flist = np.concatenate(con_flist)
        pat_flist = np.concatenate(pat_flist)

        # remove revoked subjects -- do controls and patients separately on purpose
        if args.revoked != []:
            con_flist = [l for l in con_flist if ~np.any([s in l for s in args.revoked])]
            pat_flist = [l for l in pat_flist if ~np.any([s in l for s in args.revoked])]

    # case where interactions are precomputed as L-R, only do group diff
    elif args.group_by_hemi_precomputed:
        for subj in subjs:
            if subj not in args.revoked:
                coh = get_cohort(subj)
                L_img = os.path.join(args.in_dir, metric, fwhm, seed+'L', coh,
                                        '_'.join([subj,metric,fwhm,atlas,seed+'L',seed_suffix[args.seed_type],'corr.nii']))
                R_img = os.path.join(args.in_dir, metric, fwhm, seed+'R', coh,
                                        '_'.join([subj,metric,fwhm,atlas,seed+'R',seed_suffix[args.seed_type],'corr.nii']))
                sub_img = math_img('img1 - img2', img1=L_img, img2=R_img)
                if 'control' in coh:
                    con_flist.append(sub_img)
                elif 'patient' in coh:
                    pat_flist.append(sub_img)
                else:
                    continue;

    else:
        con_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'controls', '*'))
        pat_flist = glob.glob(os.path.join(args.in_dir, metric, fwhm, seed, 'patients', '*'))

        # remove revoked subjects -- do controls and patients separately on purpose
        if args.revoked != []:
            con_flist = [l for l in con_flist if ~np.any([s in l for s in args.revoked])]
            pat_flist = [l for l in pat_flist if ~np.any([s in l for s in args.revoked])]

    con_flist = np.array(con_flist)
    pat_flist = np.array(pat_flist)
    flist = np.hstack([con_flist, pat_flist])
    return con_flist, pat_flist, flist


def create_contrast_vector(subjs, args):
    """ create contrast vector based on options given in arguments (default: only group difference) """
    suffix = ''
    n_con = np.sum(['control' in s for s in subjs])
    n_pat = np.sum(['patient' in s for s in subjs])
    if args.group_by_hemi:
        suffix += '_group_by_hemi'
        if args.paired_design:
          cv = []
          cv.append(np.concatenate([np.ones((1,2*n_con)).ravel(), -np.ones((1,2*n_pat)).ravel()]))
          cv.append(np.concatenate([np.ones((1,n_con)).ravel(), -np.ones((1,n_con)).ravel(), np.ones((1,n_pat)).ravel(), -np.ones((1,n_pat)).ravel()]))
          cv.append(np.concatenate([np.ones((1,n_con)).ravel(), -np.ones((1,n_con)).ravel(), -np.ones((1,n_pat)).ravel(), np.ones((1,n_pat)).ravel()]))
          cv = np.array(cv)
          suffix += '_paired'
        elif args.repeated2wayANOVA:
          cv = np.zeros( (1, n_con + n_pat + 2) ) # 3 contrasts (one positive + one negative each)
          # group main effect
          cv[0,:n_con] = 1
          cv[0,n_con:n_pat] = -1
          cv[1,:n_con] = -1
          cv[1,n_con:n_pat] = 1
          # session  main effect
          cv[2,-2] = 1
          cv[3,-2] = -1
          # interactions
          cv[4,-1] = 1
          cv[5,-1] = -1
          # F test
          cm = np.array([[1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1]])
          #grp = np.ones((2*n_con+2*n_pat,1))
          grp = np.concatenate([np.arange(n_con), np.arange(n_con), np.arange(n_con,n_con+n_pat), np.arange(n_con,n_con+n_pat)])+1 # offset of 1 because not sure what 0 would do
          suffix += '_repeated2wayANOVA'
        else:
          cv = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]])
          cm = np.array([[1,0], [0,1]])
          grp = np.ones((2*n_con+2*n_pat,1))
          #grp = np.concatenate([np.arange(n_con), np.arange(n_con), np.arange(n_con,n_con+n_pat), np.arange(n_con,n_con+n_pat)])+1 # offset of 1 because not sure what 0 would do
        suffix += '_Ttest_grp111'
    elif args.group_by_hemi_precomputed:
        cv = np.array([[1,-1], [-1, 1]])
        grp = np.ones((n_con+n_pat,1))
        cm = np.array([[1,0], [0,1]])
        suffix += 'group_by_hemi_precomputed'
    else:
        if args.OCD_minus_HC:
            cv = np.array([[-1, 1]])
            suffix += '_OCD_minus_HC'
            con_type = 't'
        else:
            cv = np.array([[1, -1]])
            suffix += '_HC_minus_OCD'
            con_type = 't'
        grp = np.concatenate([np.arange(n_con), np.arange(n_con,n_con+n_pat)])+1 # offset of 1 because not sure what 0 would do
        suffix += '_Ftest'
    return cv.astype(int), cm.astype(int), grp.astype(int), suffix


def use_randomise(subjs, seed, metric, atlas, args=None):
    """ perform non-parametric inference using FSL randomise and cluster-based enhancement """
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    _,_,flist = get_file_lists(subjs, seed, metric, atlas, args)
    # create 4D image from list of 3D
    imgs_4D = nilearn.image.concat_imgs(flist, auto_resample=True)
    dm = create_design_matrix(subjs, args)

    # outputs/savings to file
    out_dir = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, fwhm, seed, 'randomise')
    os.makedirs(out_dir, exist_ok=True)
    dm.to_csv(os.path.join(out_dir, 'design_mat'), sep=' ', index=False, header=False)

    cv, cm, grp, suffix = create_contrast_vector(subjs, args)
    np.savetxt(os.path.join(out_dir, 'design_con'), cv, fmt='%i')
    np.savetxt(os.path.join(out_dir, 'design_fts'), cm, fmt='%i')
    np.savetxt(os.path.join(out_dir, 'design_grp'), grp, fmt='%i')
    if args.use_jaspers_mask:
        suffix += '_jaspersThr{}'.format(int(args.jaspers_thr))
    if args.use_hcp_mask:
        suffix += '_100hcpThr{}SD'.format(int(args.hcp_sd_thr*100))
    if args.use_fspt_mask:
        suffix += '_FSPT'
    if args.use_cortical_mask:
        suffix += '_GM'
    if args.use_frontal_mask:
        suffix += '_Fr'
    if args.inflate_mask:
        suffix += '_inflateAll{}mm'.format(int(args.n_inflate))
    suffix += '_'+datetime.now().strftime('%d%m%Y')
    dmat = os.path.join(out_dir, 'design.mat')
    dcon = os.path.join(out_dir, 'design.con')
    dfts = os.path.join(out_dir, 'design.fts')
    dgrp = os.path.join(out_dir, 'design.grp')
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_mat'), dmat))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_con'), dcon))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_fts'), dfts))
    os.system('Text2Vest {} {}'.format(os.path.join(out_dir, 'design_grp'), dgrp))

    in_file = os.path.join(out_dir, seed+'_imgs_4D.nii.gz')
    nib.save(imgs_4D, in_file)
    mask_file = os.path.join(out_dir, seed+'_pathway_mask'+suffix+'.nii.gz')
    _,_,mask = mask_imgs(flist, seed=seed, masks=[],  args=args)
    mask = resample_to_img(mask, imgs_4D, interpolation='nearest')
    nib.save(mask, mask_file)


    if args.use_TFCE:
        out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_TFCE'+suffix) #'_c'+str(int(args.cluster_thresh*10)))
        cmd = 'randomise -i '+in_file+' -o '+out_file+' -d '+dmat+' -t '+dcon+' -f '+dfts+' -e '+dgrp+' -m '+mask_file+' -n '+str(args.n_perm)+' -T --uncorrp'
        #cmd = 'randomise -i '+in_file+' -o '+out_file+' -d '+dmat+' -f '+dfts+' -m '+mask_file+' -n '+str(args.n_perm)+' -T --uncorrp'
    else:
        out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_c'+str(int(args.cluster_thresh*10))+suffix)
        cmd = 'randomise -i '+in_file+' -o '+out_file+' -d '+dmat+' -t '+dcon+' -m '+mask_file+' -n '+str(args.n_perm)+' -c '+str(args.cluster_thresh)+' --uncorrp'
    print(cmd)
    os.system(cmd)


def plot_randomise_outputs(subjs, seed, metric, args, stat='t'):
    """ plot the outcomes of the non-paramteric infernece using randomise and TFCE """
    locs = {'Acc':None,
            'dCaud':None,
            'dPut':None,
            'vPut':[-49,30,12]}
    fwhm = 'brainFWHM{}mm'.format(str(int(args.brain_smoothing_fwhm)))
    cv,cm,grp,suffix = create_contrast_vector(subjs, args)
    if args.use_jaspers_mask:
        suffix += '_jaspersThr{}'.format(int(args.jaspers_thr))
    if args.use_hcp_mask:
        suffix += '_100hcpThr{}SD'.format(int(args.hcp_sd_thr*100))
    if args.use_fspt_mask:
        suffix += '_FSPT'
    if args.use_cortical_mask:
        suffix += '_GM'
    if args.use_frontal_mask:
        suffix += '_Fr'
    if args.inflate_mask:
        suffix += '_inflateAll{}mm'.format(int(args.n_inflate))
    suffix += '_'+datetime.now().strftime('%d%m%Y')
    out_dir = os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/', metric, fwhm, seed, 'randomise')

    for i in np.arange(1,3):
        if args.use_TFCE:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_TFCE'+suffix+'_tfce_corrp_'+stat+'stat{}.nii.gz'.format(i))
        else:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_c'+str(int(args.cluster_thresh*10))+suffix+'_clustere_corrp_'+stat+'stat{}.nii.gz'.format(i))

        plt.figure(figsize=[16,12])

        # FWE p-values
        ax1 = plt.subplot(3,2,1)
        plot_stat_map(out_file, axes=ax1, draw_cross=False, title=seed+' randomise -- corrp {}stat{}'.format(stat,i))
        ax2 = plt.subplot(3,2,2)
        plot_stat_map(out_file, threshold=0.95, axes=ax2, draw_cross=False, cmap='Oranges',
                        title=seed+' randomise -- {}stat{} -- corrp>0.95 (p<0.05)'.format(stat,i),
                        cut_coords=locs[seed])

        # stats
        if args.use_TFCE:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_TFCE'+suffix+'_{}stat{}.nii.gz'.format(stat,i))
        else:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_c'+str(int(args.cluster_thresh*10))+'_{}stat{}.nii.gz'.format(stat,i))
        ax3 = plt.subplot(3,2,3)
        plot_stat_map(out_file, axes=ax3, draw_cross=False, title=seed+' randomise -- {}stat{}'.format(stat,i))
        ax4 = plt.subplot(3,2,4)
        plot_stat_map(out_file, threshold=args.cluster_thresh, axes=ax4, draw_cross=False, title=seed+' randomise -- {}stat{}>{:.1f}'.format(stat,i,args.cluster_thresh))

        # FDR p-vals
        if args.use_TFCE:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_TFCE'+suffix+'_tfce_p_{}stat{}.nii.gz'.format(stat,i))
        else:
            out_file = os.path.join(out_dir, seed+'_outputs_n'+str(args.n_perm)+'_c'+str(int(args.cluster_thresh*10))+suffix+'_p_{}stat{}.nii.gz'.format(stat,i))
        ax5 = plt.subplot(3,2,5)
        plot_stat_map(out_file, axes=ax5, draw_cross=False, title=seed+' p_unc '+stat)
        ax6 = plt.subplot(3,2,6)
        plot_stat_map(out_file, threshold=0.999, axes=ax6, draw_cross=False, cmap='Oranges', title=seed+' p_unc<0.001 '+stat)


def plot_within_group_masks(subrois, glm_results, args):
    """ display maps of within-group contrasts """
    for subroi in subrois:
        plt.figure(figsize=[18,4])
        ax1 = plt.subplot(1,3,1)
        plot_stat_map(glm_results[subroi]['con_mask'],
                    axes=ax1, title=subroi+' within_con p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)
        ax2 = plt.subplot(1,3,2)
        plot_stat_map(glm_results[subroi]['pat_mask'],
                    axes=ax2, title=subroi+' within_pat p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)
        ax3 = plt.subplot(1,3,3)
        plot_stat_map(glm_results[subroi]['within_group_mask'],
                    axes=ax3, title=subroi+' within-group mask p<{}'.format(args.fpr_threshold),
                    cut_coords=cut_coords[subroi], draw_cross=False, cmap='Oranges', colorbar=False)


if __name__=='__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--seed_type', default='Harrison2009', type=str, action='store', help='choose Harrison2009, TianS4, etc')
    parser.add_argument('--compute_seed_corr', default=False, action='store_true', help="Flag to (re)compute seed to voxel correlations")
    parser.add_argument('--merge_LR_hemis', default=False, action='store_true', help="Flag to merge hemisphere's correlations")
    parser.add_argument('--n_jobs', type=int, default=10, action='store', help="number of parallel processes launched")
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--subj', default=None, action='store', help='to process a single subject, give subject ID (default: process all subjects)')
    parser.add_argument('--run_second_level', default=False, action='store_true', help='run second level statistics')
    parser.add_argument('--use_gm_mask', default=False, action='store_true', help='use a whole brain gray matter mask to reduce the space of the second level analysis')
    parser.add_argument('--use_fspt_mask', default=False, action='store_true', help='use a fronto-striato-pallido-thalamic mask to reduce the space of the second level analysis')
    parser.add_argument('--use_cortical_mask', default=False, action='store_true', help='use a cortical gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_frontal_mask', default=False, action='store_true', help='use a frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_seed_specific_mask', default=False, action='store_true', help='use a seed-spefici frontal gm mask to reduce the space of the second level analysis')
    parser.add_argument('--use_within_group_mask', default=False, action='store_true', help='use a union of within-group masks to reduce the space of the second level analysis')
    parser.add_argument('--unzip_corr_maps', default=False, action='store_true', help='unzip correlation maps for use in SPM (not necessary if only nilearn analysis)')
    parser.add_argument('--min_time_after_scrubbing', default=None, type=float, action='store', help='minimum time (in minutes) needed per subject needed to be part of the analysis (after scrubbing (None=keep all subjects))')
    parser.add_argument('--prep_fsl_randomise', default=False, action='store_true', help='Prepare 4D images for running FSL randomise')
    parser.add_argument('--use_randomise', default=False, action='store_true', help='run FSL randomise -- independent from prep_fsl_randomise')
    parser.add_argument('--cluster_thresh', type=float, default=4., action='store', help="T stat to threshold to create clusters from voxel stats")
    parser.add_argument('--use_TFCE', default=False, action='store_true', help="use Threshold-Free Cluster Enhancement with randomise ")
    parser.add_argument('--OCD_minus_HC', default=False, action='store_true', help='direction of the t-test in FSL randomise -- default uses F-test')
    parser.add_argument('--HC_minus_OCD', default=False, action='store_true', help='direction of the t-test in FSL randomise -- default uses F-test')
    parser.add_argument('--create_sphere_within_cluster', default=False, action='store_true', help='export sphere around peak within VOI cluster in prep for DCM analysis')
    parser.add_argument('--brain_smoothing_fwhm', default=8., type=none_or_float, action='store', help='brain smoothing FWHM (default 8mm as in Harrison 2009)')
    parser.add_argument('--fdr_threshold', type=float, default=0.05, action='store', help="cluster level threshold, FDR corrected")
    parser.add_argument('--fpr_threshold', type=float, default=0.001, action='store', help="cluster level threshold, uncorrected")
    parser.add_argument('--within_group_threshold', type=float, default=0.005, action='store', help="threshold to create within-group masks")
    parser.add_argument('--compute_voi_corr', default=False, action='store_true', help="compute seed to VOI correlation and print stats")
    parser.add_argument('--non_parametric_analysis', default=False, action='store_true', help="compute between group analysis using non-parametric inference")
    parser.add_argument('--use_SPM_mask', default=False, action='store_true', help="use within-group masks generated from SPM (one-tailed)")
    parser.add_argument('--two_sided_within_group', default=False, action='store_true', help="use two-tailed test to recreate within-group mask with parametric inference")
    parser.add_argument('--two_sided_between_group', default=False, action='store_true', help="use two-tailed test for between-group analysis with parametric inference")
    parser.add_argument('--n_perm', type=int, default=5000, action='store', help="number of permutation for non-parametric analysis")
    parser.add_argument('--within_mask_corr', default=False, action='store_true', help="compute FC within group masks and plot")
    parser.add_argument('--plot_within_group_masks', default=False, action='store_true', help="plot within-group masks used in second pass")
    parser.add_argument('--group_by_hemi', default=False, action='store_true', help="use a 4 columns design matrix with group by hemisphere interactions")
    parser.add_argument('--group_by_hemi_precomputed', default=False, action='store_true', help="simplified 2 columns design matrix where L-R diff is precomputed")
    parser.add_argument('--repeated2wayANOVA', default=False, action='store_true', help="2 way ANOVA with repeated measures")
    parser.add_argument('--paired_design', default=False, action='store_true', help="makes diagonal design matrix")
    parser.add_argument('--use_jaspers_mask', default=False, action='store_true', help='use a seed-specific mask from Jaspers et al. corticostriatal probability maps')
    parser.add_argument('--jaspers_thr', default=50., type=none_or_float, action='store', help='Probability Map threshold to create seed specific mask')
    parser.add_argument('--use_hcp_mask', default=False, action='store_true', help='use a seed-specific mask from HCP corticostriatal probability maps')
    parser.add_argument('--hcp_sd_thr', default=1., type=none_or_float, action='store', help='Probability Map threshold (in SD of values) to create seed specific mask with HCP')
    parser.add_argument('--inflate_mask', default=False, action='store_true', help='inflate the cortical mask by n mm to be more conservative')
    parser.add_argument('--n_inflate', default=1, type=int, action='store', help='number of mm to inflate mask')
    parser.add_argument('--combi', type=str, default='zip', action='store', help='whether to compute correlation within pathway (default=zip) or between pathways (product)')
    args = parser.parse_args()

    # if a subject or list of subjects  is given (e.g. if running on HCP), then only process them
    if args.subj!=None:
        subjs = [args.subj]
    # otherwise process all subjects
    else:
        subjs = pd.read_table(os.path.join(code_dir, 'subject_list_all.txt'), names=['name'])['name']

    # options
    atlases= ['Harrison2009'] #['schaefer100_tianS1', 'schaefer200_tianS2', 'schaefer400_tianS4'] #schaefer400_harrison2009
    pre_metric = 'seed_not_smoothed' #'unscrubbed_seed_not_smoothed'
    metrics = ['detrend_gsr_filtered_scrubFD05'] #'detrend_gsr_smooth-6mm', 'detrend_gsr_filtered_scrubFD06'

    args.atlases = atlases
    args.pre_metric = pre_metric
    args.metrics = metrics

    # to compute the correlation between VOIs, choose which mask to use
    if args.use_hcp_mask:
        args.deriv_mask = 'HCP'
    else:
        args.deriv_mask = 'Shephard'

    #TODO: in_dir must be tailored to the atlas. ATM everything is put in Harrison2009 folder
    args.in_dir = os.path.join(proj_dir, 'postprocessing/SPM/input_imgs/', args.seed_type+'Rep', pre_metric)

    seeds = list(seed_loc.keys()) #['AccL', 'AccR', 'dCaudL', 'dCaudR', 'dPutL', 'dPutR', 'vPutL', 'vPutR', 'vCaudSupL', 'vCaudSupR', 'drPutL', 'drPutR']
    subrois = np.unique([seed[:-1] for seed in seeds])#['Acc', 'dCaud', 'dPut', 'vPut', 'drPut']
    vois = [seed_to_voi[s] for s in subrois]


    seedfunc = {'Harrison2009':sphere_seed_to_voxel,
            'TianS4':seed_to_voxel}

    # First remove subjects without enough data
    if args.min_time_after_scrubbing != None:
        subjs, revoked = get_subjs_after_scrubbing(subjs, metrics, min_time=args.min_time_after_scrubbing)
    else:
        revoked=[]
    args.revoked=revoked

    # Then process data
    if args.compute_seed_corr:
        for atlas in atlases:
            Parallel(n_jobs=args.n_jobs)(delayed(seedfunc[args.seed_type])(subj,seeds,metrics,atlases,args) for subj in subjs)

    if args.unzip_corr_maps:
        unzip_correlation_maps(subjs, metrics, atlases, seeds, args)

    if args.merge_LR_hemis:
        in_fnames = merge_LR_hemis(subjs, subrois, metrics, seed_type=str(seedfunc[args.seed_type]), args=args)

        # can only prep fsl with in_fnames from merge_LR_hemis
        if args.prep_fsl_randomise:
            prep_fsl_randomise(in_fnames, subrois, metrics, args)

    # use randomise (independent from prep_fsl_randomise)
    if args.use_randomise:
        for subroi,metric,atlas in itertools.product(subrois,metrics, atlases):
            use_randomise(subjs, subroi, metric, atlas, args)
            if args.plot_figs:
                plot_randomise_outputs(subjs, subroi, metric, args)

    # parametric approach
    if args.run_second_level:
        out_dir = os.path.join(proj_dir, 'postprocessing', 'glm', pre_metric)
        os.makedirs(out_dir, exist_ok=True)
        args.out_dir = out_dir
        glm_results = run_second_level(subjs, metrics, subrois, args)

        if args.plot_within_group_masks:
            plot_within_group_masks(subrois, glm_results, args)

    # correlation between seeds and significant clusters
    if args.compute_voi_corr:
        df_voi_corr = compute_voi_corr(subjs, seeds=subrois, vois=vois, args=args)
        print_voi_stats(df_voi_corr, seeds=subrois, vois=vois, args=args)
        plot_voi_corr(df_voi_corr, seeds=subrois, vois=vois, args=args)

        if args.save_outputs:
            with open(os.path.join(proj_dir, 'postprocessing', 'df_voi_corr_replicate_T_15022023.pkl'), 'wb') as f:
                pickle.dump(df_voi_corr,f)

    # non-parametric using nilearn (experimental)
    if args.non_parametric_analysis:
        suffix = 'non_parametric'
        if args.two_sided_within_group:
            suffix += '_within2tailed'
        if args.two_sided_between_group:
            suffix += '_between2tailed'

        np_results = dict()
        for seed,metric in itertools.product(subrois, metrics):
            neg_log_pvals, mask = non_parametric_analysis(subjs, seed, metric, pre_metric, args=args, masks=[])
            plot_non_param_maps(neg_log_pvals, seed, suffix, args)
            np_results[seed] = {'neg_log_pvals':neg_log_pvals, 'mask':mask}

        if args.save_outputs:
            with gzip.open(os.path.join(proj_dir, 'postprocessing', 'non_parametric', suffix+'.pkl.gz'), 'wb') as f:
                pickle.dump(np_results,f)

    # FC bar/box plots
    if args.within_mask_corr:
        df_mask_corr = compute_FC_within_masks(subjs, np_results, args=args)
        if args.plot_figs:
            plot_within_mask_corr(df_mask_corr, args=args)

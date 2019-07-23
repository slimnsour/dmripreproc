import os
import multiprocessing

import numpy as np
import nibabel as nib
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu
from nipype.utils import NUMPY_MMAP
from nipype.utils.filemanip import fname_presuffix
from dipy.segment.mask import median_otsu
from numba import cuda
from bids import BIDSLayout

from ...interfaces import mrtrix3
from ...interfaces import fsl as dmri_fsl

from nipype.interfaces import fsl, utility as niu

def init_tract_wf():

    tract_wf = pe.Workflow(name="tract_wf")
    tract_wf.base_dir = '/scratch/smansour/tract_testing/scratch'

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subject_id",
                "t1_file",
                "eddy_file",
                "dwi_mask",
                "bval",
                "bvec",
                "bvecbval",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file"]),
        name="outputnode",
    )

    #register T1 to diffusion space first
    #flirt -dof 6 -in T1w_brain.nii.gz -ref nodif_brain.nii.gz -omat xformT1_2_diff.mat -out T1_diff
    flirt = pe.Node(fsl.FLIRT(dof=6), name="t1_flirt")

    # T1 should already be skull stripped and minimally preprocessed (from Freesurfer will do)
    #5ttgen fsl -nocrop -premasked T1_diff.nii.gz 5TT.mif
    gen5tt = pe.Node(mrtrix3.Generate5tt(algorithm='fsl', out_file='5TT.mif'), name="gen5tt")
    #5tt2gmwmi 5TT.mif gmwmi.mif
    gen5ttMask = pe.Node(mrtrix3.Generate5ttMask(out_file='gmwmi.mif'), name="gen5ttMask")

    #SINGLE SHELL
    # generate response function
    #dwi2response tournier data.nii.gz -fslgrad data.eddy_rotated_bvecs dwi.bval response.txt
    responseSD = pe.Node(mrtrix3.ResponseSD(algorithm='tournier'), name="responseSD")
    # generate FODs
    #dwi2fod csd data.nii.gz response.txt FOD.mif -mask nodif_brain_mask.nii.gz -fslgrad data.eddy_rotated_bvecs dwi.bval
    estimateFOD = pe.Node(mrtrix3.EstimateFOD(algorithm='csd', wm_odf='FOD.mif'), name="estimateFOD")
    # perform probabilistic tractography
    #tckgen FOD.mif prob.tck -act 5TT.mif -seed_gmwmi gmwmi.mif -select 5000000 ## seeding from a binarised gmwmi
    tckgen = pe.Node(mrtrix3.Tractography(select=5000000), name="tckgen")
    #mrview data.nii.gz -tractography.load prob.tck

    #use sift to filter tracks based on spherical harmonics
    #tcksift2 prob.tck FOD.mif prob_weights.txt

    ## atlas reg
    #flirt -in T1w_brain.nii.gz -ref MNI152_T1_1mm_brain.nii.gz -omat xformT1_2_MNI.mat
    #convert_xfm -omat xformMNI_2_T1.mat -inverse xformT12MNI.mat
    #convert_xfm -omat xformMNI_2_diff.mat -concat xformT1_2_diff.mat xformMNI_2_T1.mat

    #flirt -in shen268.nii.gz -ref T1_diff.nii.gz -applyxfm -init xformMNI_2_diff.mat -interp nearestneighbour -out shen_diff_space.nii.gz

    ## generate connectivity matrices
    #tck2connectome prob.tck shen_diff_space.nii.gz conmat_shen.csv -scale_invlength -zero_diagonal -symmetric -tck_weights_in prob_weights.txt -assignment_radial_search 2 -scale_invnodevol
    #tck2connectome prob.tck shen_diff_space.nii.gz conmat_length_shen.csv -zero_diagonal -symmetric -scale_length -stat_edge mean -assignment_radial_search 2

    tract_wf.connect(
        [
            (
                inputnode,
                flirt,
                [
                    ("t1_file", "in_file"),
                    ("dwi_mask", "reference")
                ]
            ),
            (flirt, gen5tt, [("out_file", "in_file")]),
            (gen5tt, gen5ttMask, [("out_file", "in_file")]),
            (
                inputnode,
                responseSD,
                [
                    ("eddy_file", "in_file"),
                    ("bvecbval", "grad_fsl")
                ]
            ),
            (
                inputnode,
                estimateFOD,
                [
                    ("eddy_file", "in_file"),
                    ("bvecbval", "grad_fsl")
                ]
            ),
            (responseSD, estimateFOD, [("wm_file", "wm_txt")]),
            (estimateFOD, tckgen, [("wm_odf", "in_file")]),
            (gen5tt, tckgen, [("out_file", "act_file")]),
            (gen5ttMask, tckgen, [("out_file", "seed_gmwmi")])
        ]
    )
    return tract_wf

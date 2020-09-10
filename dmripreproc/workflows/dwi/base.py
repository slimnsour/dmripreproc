#!/usr/bin/env python

import os
import multiprocessing

import numpy as np
import nibabel as nib
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu
from nipype.utils.filemanip import fname_presuffix
from dipy.segment.mask import median_otsu
from numba import cuda
from bids import BIDSLayout

from ...interfaces import mrtrix3
from ...interfaces import fsl as dmri_fsl
from ..fieldmap.base import init_sdc_prep_wf
from .dwiprep import init_dwiprep_wf

from niworkflows.anat.ants import init_brain_extraction_wf

FMAP_PRIORITY = {"epi": 0, "fieldmap": 1, "phasediff": 2, "phase": 3, "syn": 4}

def init_dwi_preproc_wf(subject_id, dwi_file, metadata, parameters):
    fmaps = []
    synb0 = ""

    # If use_synb0 set, get synb0 from files
    if parameters.synb0_dir:
        synb0_layout =  BIDSLayout(parameters.synb0_dir, validate=False, derivatives=True)
        synb0 = synb0_layout.get(subject=subject_id, return_type='file')[0]
    else:
        fmaps = parameters.layout.get_fieldmap(dwi_file, return_list=True)
        if not fmaps:
            raise Exception(
                "No fieldmaps found for participant {}. "
                "All workflows require fieldmaps".format(subject_id)
            )

        for fmap in fmaps:
            fmap["metadata"] = parameters.layout.get_metadata(fmap[fmap["suffix"]])

    sdc_wf = init_sdc_prep_wf(
        subject_id,
        fmaps,
        metadata,
        parameters.layout,
        parameters.bet_mag,
        synb0=synb0,
        acqp_file=parameters.acqp_file,
        ignore_nodes=parameters.ignore_nodes
    )

    dwi_wf = pe.Workflow(name="dwi_preproc_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subject_id",
                "dwi_file",
                "dwi_meta",
                "bvec_file",
                "bval_file",
                "out_dir",
                "eddy_niter",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_file", "out_mask", "out_bvec", "out_qc_folder"]),
        name="outputnode",
    )

    # Create the dwi prep workflow
    dwi_prep_wf = init_dwiprep_wf(parameters.ignore_nodes)

    def gen_index(in_file):
        import os
        import numpy as np
        import nibabel as nib
        from nipype.pipeline import engine as pe
        from nipype.interfaces import fsl, utility as niu
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(
            in_file,
            suffix="_index.txt",
            newpath=os.path.abspath("."),
            use_ext=False,
        )
        vols = nib.load(in_file).shape[-1]
        index_lines = np.ones((vols,))
        index_lines_reshape = index_lines.reshape(1, index_lines.shape[0])
        np.savetxt(out_file, index_lines_reshape, fmt="%i")
        return out_file

    gen_idx = pe.Node(
        niu.Function(
            input_names=["in_file"],
            output_names=["out_file"],
            function=gen_index,
        ),
        name="gen_index",
    )

    def gen_acqparams(in_file, metadata, total_readout_time):
        import os
        import numpy as np
        import nibabel as nib
        from nipype.utils.filemanip import fname_presuffix

        out_file = fname_presuffix(
            in_file,
            suffix="_acqparams.txt",
            newpath=os.path.abspath("."),
            use_ext=False,
        )

        acq_param_dict = {
            "j": "0 1 0 %.7f",
            "j-": "0 -1 0 %.7f",
            "i": "1 0 0 %.7f",
            "i-": "-1 0 0 %.7f",
            "k": "0 0 1 %.7f",
            "k-": "0 0 -1 %.7f",
        }

        pe_dir = metadata.get("PhaseEncodingDirection")

        if not(pe_dir):
            pe_dir = metadata.get("PhaseEncodingAxis")

        if total_readout_time:
            total_readout = total_readout_time
        else:
            total_readout = metadata.get("TotalReadoutTime")

        acq_param_lines = acq_param_dict[pe_dir] % total_readout

        with open(out_file, "w") as f:
            f.write(acq_param_lines)

        return out_file

    acqp = pe.Node(
        niu.Function(
            input_names=["in_file", "metadata", "total_readout_time"],
            output_names=["out_file"],
            function=gen_acqparams,
        ),
        name="acqp",
    )

    acqp.inputs.total_readout_time = parameters.total_readout

    def b0_average(in_dwi, in_bval, b0_thresh=10.0, out_file=None):
        """
        A function that averages the *b0* volumes from a DWI dataset.
        As current dMRI data are being acquired with all b-values > 0.0,
        the *lowb* volumes are selected by specifying the parameter b0_thresh.
        .. warning:: *b0* should be already registered (head motion artifact
        should be corrected).
        """
        import os
        import numpy as np
        import nibabel as nib
        from nipype.pipeline import engine as pe
        from nipype.interfaces import fsl, utility as niu
        from nipype.utils.filemanip import fname_presuffix

        if out_file is None:
            out_file = fname_presuffix(
                in_dwi, suffix="_avg_b0", newpath=os.path.abspath(".")
            )

        imgs = np.array(nib.four_to_three(nib.load(in_dwi)))
        bval = np.loadtxt(in_bval)
        index = np.argwhere(bval <= b0_thresh).flatten().tolist()

        b0s = [im.get_data().astype(np.float32) for im in imgs[index]]
        b0 = np.average(np.array(b0s), axis=0)

        hdr = imgs[0].header.copy()
        hdr.set_data_shape(b0.shape)
        hdr.set_xyzt_units("mm")
        hdr.set_data_dtype(np.float32)
        nib.Nifti1Image(b0, imgs[0].affine, hdr).to_filename(out_file)
        return out_file

    avg_b0_0 = pe.Node(
        niu.Function(
            input_names=["in_dwi", "in_bval"],
            output_names=["out_file"],
            function=b0_average,
        ),
        name="b0_avg_pre",
    )

    eddy_avg_b0 = pe.Node(
        niu.Function(
            input_names=["in_dwi", "in_bval"],
            output_names=["out_file"],
            function=b0_average,
        ),
        name="eddy_avg_b0",
    )

    # dilate mask
    bet_dwi0 = pe.Node(
        fsl.BET(frac=parameters.bet_dwi, mask=True, robust=True),
        name="bet_dwi_pre",
    )

    # mrtrix3.MaskFilter

    ecc = pe.Node(
        dmri_fsl.Eddy(repol=True, cnr_maps=True, residuals=True, method="jac"),
        name="fsl_eddy",
    )

    # if nthreads not specified, do this
    ecc.inputs.num_threads = multiprocessing.cpu_count()

    try:
        if cuda.gpus:
            ecc.inputs.use_cuda = True
    except:
        ecc.inputs.use_cuda = False

    denoise_eddy = pe.Node(mrtrix3.DWIDenoise(), name="denoise_eddy")

    denoise_eddy_mask = pe.Node(fsl.ApplyMask(), name="denoise_eddy_mask")

    eddy_quad = pe.Node(fsl.EddyQuad(verbose=True), name="eddy_quad")

    get_path = lambda x: x.split(".nii.gz")[0].split("_fix")[0]
    get_qc_path = lambda x: x.split(".nii.gz")[0] + ".qc"

    fslroi = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name="fslroi")

    def get_b0_mask_fn(b0_file):
        import os
        import nibabel as nib
        from nipype.pipeline import engine as pe
        from nipype.interfaces import fsl, utility as niu
        from nipype.utils.filemanip import fname_presuffix
        from dipy.segment.mask import median_otsu

        mask_file = fname_presuffix(
            b0_file, suffix="_mask", newpath=os.path.abspath(".")
        )
        img = nib.load(b0_file)
        data, aff = img.get_data(), img.affine
        _, mask = median_otsu(data, 2, 1)
        nib.Nifti1Image(mask.astype(float), aff).to_filename(mask_file)
        return mask_file

    b0mask_node = pe.Node(
        niu.Function(
            input_names=["b0_file"],
            output_names=["mask_file"],
            function=get_b0_mask_fn,
        ),
        name="getB0Mask",
    )

    eddy_b0mask_node = pe.Node(
        niu.Function(
            input_names=["b0_file"],
            output_names=["mask_file"],
            function=get_b0_mask_fn,
        ),
        name="geteddyB0Mask",
    )

    t1_skullstrip = init_brain_extraction_wf()

    to_list = lambda x: [x]

    # If synb0 is meant to be used
    if parameters.synb0_dir:
        dwi_wf.connect(
            [
                (
                    sdc_wf,
                    ecc,
                    [
                        ("outputnode.out_topup", "in_topup_fieldcoef"),
                        ("outputnode.out_movpar", "in_topup_movpar"),
                    ],
                )
            ]
        )
        ecc.inputs.in_acqp = parameters.acqp_file
        eddy_quad.inputs.param_file = parameters.acqp_file
    else:
        # Decide what ecc will take: topup or fmap
        fmaps.sort(key=lambda fmap: FMAP_PRIORITY[fmap["suffix"]])
        fmap = fmaps[0]
        # Else If epi files detected
        if fmap["suffix"] == "epi":
            dwi_wf.connect(
                [
                    (
                        sdc_wf,
                        ecc,
                        [
                            ("outputnode.out_topup", "in_topup_fieldcoef"),
                            ("outputnode.out_enc_file", "in_acqp"),
                            ("outputnode.out_movpar", "in_topup_movpar"),
                        ],
                    ),
                    (sdc_wf, eddy_quad, [("outputnode.out_enc_file", "param_file")])
                ]
            )
        # Otherwise (fieldmaps)
        else:
            if not(parameters.avoid_fieldmap_eddy):
                dwi_wf.connect(
                    [
                        (sdc_wf, ecc, [(("outputnode.out_fmap", get_path), "field")]),
                        (
                            inputnode,
                            acqp,
                            [("dwi_file", "in_file"), ("dwi_meta", "metadata")],
                        ),
                        (acqp, ecc, [("out_file", "in_acqp")]),
                        (acqp, eddy_quad, [("out_file", "param_file")])
                    ]
                )
            else:
                dwi_wf.connect(
                    [
                        (
                            inputnode,
                            acqp,
                            [("dwi_file", "in_file"), ("dwi_meta", "metadata")],
                        ),
                        (acqp, ecc, [("out_file", "in_acqp")]),
                        (acqp, eddy_quad, [("out_file", "param_file")])
                    ]
                )

    dtifit = pe.Node(fsl.DTIFit(save_tensor=True, sse=True), name="dtifit")

    dwi_wf.connect(
        [
            (
                inputnode,
                dwi_prep_wf,
                [("dwi_file", "dwi_prep_inputnode.dwi_file")],
            ),
            (
                dwi_prep_wf,
                avg_b0_0,
                [("dwi_prep_outputnode.out_file", "in_dwi")],
            ),
            (inputnode, avg_b0_0, [("bval_file", "in_bval")]),
            (avg_b0_0, bet_dwi0, [("out_file", "in_file")]),
            (inputnode, gen_idx, [("dwi_file", "in_file")]),
            (dwi_prep_wf, ecc, [("dwi_prep_outputnode.out_file", "in_file")]),
            (
                inputnode,
                ecc,
                [("bval_file", "in_bval"), ("bvec_file", "in_bvec")],
            ),
            (bet_dwi0, ecc, [("mask_file", "in_mask")]),
            (gen_idx, ecc, [("out_file", "in_index")]),
            (ecc, denoise_eddy, [("out_corrected", "in_file")]),
            (ecc, fslroi, [("out_corrected", "in_file")]),
            (fslroi, b0mask_node, [("roi_file", "b0_file")]),
            (
                ecc,
                eddy_quad,
                [
                    (("out_corrected", get_path), "base_name"),
                    (("out_corrected", get_qc_path), "output_dir"),
                ],
            ),
            (inputnode, eddy_quad, [("bval_file", "bval_file")]),
            (ecc, eddy_quad, [("out_rotated_bvecs", "bvec_file")]),
            (b0mask_node, eddy_quad, [("mask_file", "mask_file")]),
            (gen_idx, eddy_quad, [("out_file", "idx_file")]),
            (ecc, outputnode, [("out_corrected", "out_file")]),
            (b0mask_node, outputnode, [("mask_file", "out_mask")]),
            (ecc, outputnode, [("out_rotated_bvecs", "out_bvec")]),
            (bet_dwi0, sdc_wf, [("out_file", "inputnode.b0_stripped")]),
            (sdc_wf, eddy_quad, [("outputnode.out_fmap", "field")]),
            (
                ecc,
                dtifit,
                [("out_corrected", "dwi"), ("out_rotated_bvecs", "bvecs")],
            ),
            (b0mask_node, dtifit, [("mask_file", "mask")]),
            (inputnode, dtifit, [("bval_file", "bvals")]),
            # New avg b0 for eddy
            (ecc, eddy_avg_b0, [("out_corrected", "in_dwi")]),
            (eddy_avg_b0, eddy_b0mask_node, [("out_file", "b0_file")]),
            (denoise_eddy, denoise_eddy_mask, [("noise", "in_file")]),
            (eddy_b0mask_node, denoise_eddy_mask, [("mask_file", "mask_file")]),
            (inputnode, eddy_avg_b0, [("bval_file", "in_bval")]),
        ]
    )

    if (parameters.skullstrip_t1):
        dwi_wf.connect(
            [
                (inputnode, t1_skullstrip, [(("t1_file", to_list), "inputnode.in_files")])
            ]
        )

    return dwi_wf

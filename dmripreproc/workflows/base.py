#!/usr/bin/env python

import os
from copy import deepcopy

from nipype.pipeline import engine as pe
from .dwi.base import init_dwi_preproc_wf
from .dwi.outputs import init_output_wf

def init_dmripreproc_wf(parameters):
    dmripreproc_wf = pe.Workflow(name="dmripreproc_wf")
    dmripreproc_wf.base_dir = parameters.work_dir

    for subject_id in parameters.subject_list:

        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            name="single_subject_" + subject_id + "_wf",
            parameters=parameters,
        )

        single_subject_wf.config["execution"]["crashdump_dir"] = os.path.join(
            parameters.output_dir, "dmriprep_crash", "sub-" + subject_id, "log"
        )

        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        dmripreproc_wf.add_nodes([single_subject_wf])

    return dmripreproc_wf


def init_single_subject_wf(subject_id, name, parameters):

    dwi_files = parameters.layout.get(
        subject=subject_id,
        datatype="dwi",
        suffix="dwi",
        extension=["nii", "nii.gz"],
        return_type="filename"
    )

    if not dwi_files:
        raise Exception(
            "No dwi images found for participant {}. "
            "All workflows require dwi images".format(subject_id)
        )

    subject_wf = pe.Workflow(name=name)

    for dwi_file in dwi_files:
        entities = parameters.layout.parse_file_entities(dwi_file)
        # Get T1 for subject
        if "session" in entities:
            session_id = entities["session"]
            t1_file = parameters.layout.get(
                subject=subject_id,
                session=session_id,
                suffix="T1w",
                extension=[".nii", ".nii.gz"],
                return_type="filename"
            )[0]
        else:
            session_id = "01"
            t1_file = parameters.layout.get(
                subject=subject_id,
                suffix="T1w",
                extension=[".nii", ".nii.gz"],
                return_type="filename"
            )[0]

        metadata = parameters.layout.get_metadata(dwi_file)
        dwi_preproc_wf = init_dwi_preproc_wf(
            subject_id=subject_id,
            dwi_file=dwi_file,
            metadata=metadata,
            parameters=parameters,
        )
        datasink_wf = init_output_wf(
            subject_id=subject_id,
            session_id=session_id,
            output_folder=parameters.output_dir,
        )

        dwi_preproc_wf.base_dir = os.path.join(
            os.path.abspath(parameters.work_dir), subject_id
        )

        inputspec = dwi_preproc_wf.get_node("inputnode")
        inputspec.inputs.subject_id = subject_id
        inputspec.inputs.dwi_file = dwi_file
        inputspec.inputs.t1_file = t1_file
        inputspec.inputs.dwi_meta = metadata
        inputspec.inputs.bvec_file = parameters.layout.get_bvec(dwi_file)
        inputspec.inputs.bval_file = parameters.layout.get_bval(dwi_file)
        inputspec.inputs.out_dir = os.path.abspath(parameters.output_dir)

        ds_inputspec = datasink_wf.get_node("inputnode")
        ds_inputspec.inputs.subject_id = subject_id
        ds_inputspec.inputs.session_id = session_id
        ds_inputspec.inputs.output_folder = parameters.output_dir
        ds_inputspec.inputs.metadata = metadata

        wf_name = "sub_" + subject_id + "_ses_" + session_id + "_preproc_wf"
        full_wf = pe.Workflow(name=wf_name)

        full_wf.connect(
            [
                (
                    dwi_preproc_wf,
                    datasink_wf,
                    [
                        #("fsl_eddy.out_corrected", "inputnode.out_file"),
                        #("fsl_eddy.out_rotated_bvecs", "inputnode.out_bvec"),
                        #("getB0Mask.mask_file", "inputnode.out_mask"),
                        ("outputnode.out_corrected", "inputnode.out_file"),
                        ("outputnode.out_mask", "inputnode.out_mask"),
                        ("inputnode.bval_file", "inputnode.out_bval"),
                        ("outputnode.out_bvec", "inputnode.out_bvec"),
                        ("bet_dwi_pre.out_file", "inputnode.out_b0_brain"),
                        ("bet_dwi_pre.mask_file", "inputnode.out_b0_mask"),
                        ("outputnode.out_FA", "inputnode.out_FA"),
                        ("outputnode.out_V1", "inputnode.out_V1"),
                        ("outputnode.out_MD", "inputnode.out_MD"),
                        ("outputnode.out_L1", "inputnode.out_L1"),
                        ("outputnode.out_RD", "inputnode.out_RD"),
                        #("denoise_eddy.noise", "inputnode.out_sh_residual"),
                        #("denoise_eddy_mask.out_file", "inputnode.out_sh_residual_mask"),
                        # Dtifit SSE
                        ("dtifit.sse", "inputnode.out_sse"),
                        #("outputnode.out_avg_b0", "inputnode.out_eddy_b0"),
                        #("geteddyB0Mask.mask_file", "inputnode.out_eddy_b0_mask"),
                        # Eddy qc
                        ("eddy_quad.qc_json", "inputnode.out_eddy_qc_json"),
                        ("eddy_quad.qc_pdf", "inputnode.out_eddy_qc_pdf")
                    ],
                )
            ]
        )

        subject_wf.add_nodes([full_wf])

    return subject_wf

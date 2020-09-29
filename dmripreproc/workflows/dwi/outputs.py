#!/usr/bin/env python

import os

from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio, utility as niu


def init_output_wf(subject_id, session_id, output_folder):

    op_wf = pe.Workflow(name="output_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subject_id",
                "session_id",
                "out_file",
                "out_mask",
                "out_bvec",
                "out_bval",
                "out_b0_brain",
                "out_b0_mask",
                "out_fieldmap_brain",
                "out_eddy_qc",
                "out_FA",
                "out_V1",
                "out_MD",
                "out_L1",
                "out_RD",
                "out_sh_residual",
                "out_sh_residual_mask",
                "output_folder",
                "out_sse",
                "out_eddy_b0",
                "out_eddy_b0_mask",
                "out_eddy_qc_json",
                "out_eddy_qc_pdf",
            ]
        ),
        name="inputnode",
    )

    def build_path(output_folder, subject_id, session_id):
        import os

        return os.path.join(
            output_folder,
            "dmripreproc",
            "sub-" + subject_id,
            "ses-" + session_id,
            "dwi",
        )

    concat = pe.Node(
        niu.Function(
            input_names=["output_folder", "subject_id", "session_id"],
            output_names=["built_folder"],
            function=build_path,
        ),
        name="build_path",
    )

    datasink = pe.Node(nio.DataSink(), name="datasink")

    # Rename Nodes
    prefix = "sub-{}_ses-{}".format(subject_id, session_id)
    template = prefix + "_desc-preproc_{}"
    dti_template = prefix + "_model-DTI_desc-preproc_{}"
    dwi_rename = pe.Node(niu.Rename(format_string=template.format("dwi"), keep_ext=True), name="dwi_rename")
    fa_rename = pe.Node(niu.Rename(format_string=dti_template.format("FA"), keep_ext=True), name="fa_rename")
    ad_rename = pe.Node(niu.Rename(format_string=dti_template.format("AD"), keep_ext=True), name="ad_rename")
    md_rename = pe.Node(niu.Rename(format_string=dti_template.format("MD"), keep_ext=True), name="md_rename")
    rd_rename = pe.Node(niu.Rename(format_string=dti_template.format("RD"), keep_ext=True), name="rd_rename")
    mask_rename = pe.Node(niu.Rename(format_string=template.format("mask"), keep_ext=True), name="mask_rename")
    bvec_rename = pe.Node(niu.Rename(format_string=template.format("dwi") + ".bvec"), name="bvec_rename")
    bval_rename = pe.Node(niu.Rename(format_string=template.format("dwi"), keep_ext=True), name="bval_rename")

    op_wf.connect(
        [
            (
                inputnode,
                concat,
                [
                    ("subject_id", "subject_id"),
                    ("session_id", "session_id"),
                    ("output_folder", "output_folder"),
                ],
            ),
            (concat, datasink, [("built_folder", "base_directory")]),
            # Renaming the files
            (inputnode, dwi_rename, [("out_file", "in_file")]),
            (inputnode, fa_rename, [("out_FA", "in_file")]),
            (inputnode, ad_rename, [("out_L1", "in_file")]),
            (inputnode, md_rename, [("out_MD", "in_file")]),
            (inputnode, rd_rename, [("out_RD", "in_file")]),
            (inputnode, mask_rename, [("out_mask", "in_file")]),
            (inputnode, bvec_rename, [("out_bvec", "in_file")]),
            (inputnode, bval_rename, [("out_bval", "in_file")]),
            # Outputting the renamed files
            (dwi_rename, datasink, [("out_file", "@result.@dwi")]),
            (fa_rename, datasink, [("out_file", "@result.@FA")]),
            (ad_rename, datasink, [("out_file", "@result.@AD")]),
            (md_rename, datasink, [("out_file", "@result.@MD")]),
            (rd_rename, datasink, [("out_file", "@result.@RD")]),
            (mask_rename, datasink, [("out_file", "@result.@mask")]),
            (bvec_rename, datasink, [("out_file", "@result.@bvec")]),
            (bval_rename, datasink, [("out_file", "@result.@bval")]),
            # Outputting the qc pdf
            (inputnode, datasink, [("out_eddy_qc_pdf", "@result.@eddyqcpdf")])
            # (
            #     inputnode,
            #     datasink,
            #     [
            #         #("out_file", "@result.@dwi"),
            #         #("out_bvec", "@result.@bvec"),
            #         #("out_bval", "@result.@bval"),
            #         #("out_mask", "@result.@mask"),
            #         ("out_b0_brain", "@result.@b0brain"),
            #         ("out_b0_mask", "@result.@b0mask"),
            #         # ("out_fieldmap_brain", "@result.@fmapbrain"),
            #         ("out_eddy_qc", "@result.@eddyqc"),
            #         ("out_FA", "@result.@fa"),
            #         ("out_V1", "@result.@v1"),
            #         ("out_MD", "@result.@MD"),
            #         ("out_L1", "@result.@L1"),
            #         ("out_RD", "@result.@RD"),
            #         ("out_sse", "@result.@sse"),
            #         ("out_sh_residual", "@result.@residual"),
            #         ("out_sh_residual_mask", "@result.@residualmask"),
            #         ("out_eddy_b0", "@result.@eddyb0"),
            #         ("out_eddy_b0_mask", "@result.@eddyb0mask"),
            #         ("out_eddy_qc_json", "@result.@eddyqcjson"),
            #         ("out_eddy_qc_pdf", "@result.@eddyqcpdf"),
            #     ],
            # ),
        ]
    )

    return op_wf

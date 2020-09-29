#!/usr/bin/env python

from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu


def init_pepolar_wf(subject_id, dwi_meta, epi_fmaps, acqp_file=None):

    wf = pe.Workflow(name="pepolar_wf")

    inputnode = pe.Node(niu.IdentityInterface(fields=["b0_stripped"]), name = "inputnode")

    outputnode = pe.Node(niu.IdentityInterface(fields=["out_topup", "out_movpar", "out_fmap", "out_enc_file"]), name = "outputnode")

    dwi_file_pe = dwi_meta["PhaseEncodingDirection"]

    file2dir = dict()

    usable_fieldmaps_matching_pe = []
    usable_fieldmaps_opposite_pe = []

    for fmap, pe_dir in epi_fmaps:
        if pe_dir == dwi_file_pe:
            usable_fieldmaps_matching_pe.append(fmap)
            file2dir[fmap] = pe_dir
        elif pe_dir[0] == dwi_file_pe[0]:
            usable_fieldmaps_opposite_pe.append(fmap)
            file2dir[fmap] = pe_dir

    if not usable_fieldmaps_opposite_pe:
        raise Exception("None of the discovered fieldmaps for "
                        "participant {} has the right phase "
                        "encoding direction".format(subject_id))

    topup_wf = init_topup_wf(acqp_file=acqp_file)
    
    wf.add_nodes([inputnode])

    dir_map = {
        'i': 'x',
        'i-': 'x-',
        'j': 'y',
        'j-': 'y-',
        'k': 'z',
        'k-': 'z-'
    }

    # For resizing the epi file to dwi size
    epi_resize = pe.Node(fsl.ExtractROI(x_min=0, x_size=140, y_min=0, y_size=140, z_min=0, z_size=80), name="epi_resize")

    # If there is no matching direction fieldmap
    if not usable_fieldmaps_matching_pe:
        # Feed the b0 into topup
        wf.connect(
            [
                (inputnode, topup_wf, [("b0_stripped", "inputnode.epi_file")])
            ]
        )
        epi_resize.inputs.in_file = usable_fieldmaps_opposite_pe[0]
        wf.connect(
            [
                (epi_resize, topup_wf, [("roi_file", "inputnode.altepi_file")])
            ]
        )
        epi_list = ['b0', usable_fieldmaps_opposite_pe[0]]
        file2dir['b0'] = dwi_file_pe
    # Otherwise (both directions are available)
    else:
        # topup_wf.inputs.inputnode.altepi_file = usable_fieldmaps_opposite_pe[0]
        # Resize the epi file
        # epi_resize.inputs.in_file = usable_fieldmaps_matching_pe[0]
        # Feed the matching direction fieldmap into topup
        # wf.connect(
        #     [
        #         (epi_resize, topup_wf, [("roi_file", "inputnode.epi_file")])
        #     ]
        # )
        topup_wf.inputs.inputnode.altepi_file = usable_fieldmaps_opposite_pe[0]
        topup_wf.inputs.inputnode.epi_file = usable_fieldmaps_matching_pe[0]
        epi_list = [topup_wf.inputs.inputnode.epi_file, topup_wf.inputs.inputnode.altepi_file]
    
    # Get the directions for the epi files in case acqp not provided
    topup_wf.inputs.inputnode.encoding_directions = [dir_map[file2dir[file]] for file in epi_list]

    wf.connect(
        [
            (
                topup_wf,
                outputnode,
                [
                    ("outputnode.out_fmap", "out_fmap"),
                    ("outputnode.out_movpar", "out_movpar"),
                    ("outputnode.out_base", "out_topup"),
                    ("outputnode.out_enc_file", "out_enc_file"),
                ]
            ),
        ]
    )

    return wf


def init_synb0_wf(subject_id, dwi_meta, synb0, acqp_file, ignore_nodes):
    file2dir = dict()

    usable_fieldmaps_matching_pe = []
    usable_fieldmaps_opposite_pe = []

    wf = pe.Workflow(name="synb0_wf")

    inputnode = pe.Node(niu.IdentityInterface(fields=["b0_stripped"]), name = "inputnode")

    outputnode = pe.Node(niu.IdentityInterface(fields=["out_topup", "out_movpar", "out_fmap", "out_enc_file"]), name = "outputnode")

    topup_wf = init_topup_wf(ignore_nodes=ignore_nodes, acqp_file=acqp_file)
    topup_wf.inputs.inputnode.altepi_file = synb0
    wf.add_nodes([inputnode])
    wf.connect(
        [
            (inputnode, topup_wf, [("b0_stripped", "inputnode.epi_file")])
        ]
    )

    wf.connect(
        [
            (
                topup_wf,
                outputnode,
                [
                    ("outputnode.out_fmap", "out_fmap"),
                    ("outputnode.out_movpar", "out_movpar"),
                    ("outputnode.out_base", "out_topup"),
                ]
            ),
        ]
    )

    return wf

def init_topup_wf(ignore_nodes="r", acqp_file=None):
    from ...interfaces import mrtrix3

    wf = pe.Workflow(name="topup_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["epi_file", "altepi_file", "encoding_directions", "topup_name", "acqp"]),
        name="inputnode")
    inputnode.inputs.topup_name = "topup_base"

    #epi_flirt = pe.Node(fsl.FLIRT(), name="epi_flirt")

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_fmap", "out_movpar", "out_base", "out_enc_file"]),
        name="outputnode")

    list_merge = pe.Node(niu.Merge(numinputs=2), name="list_merge")

    merge = pe.Node(fsl.Merge(dimension="t"), name="mergeAPPA")

    # Resize (make optional)
    resize = pe.Node(mrtrix3.MRResize(voxel_size=[1]), name="epi_resize")

    # Resize to dwi
    appa_resize = pe.Node(fsl.ExtractROI(x_min=0, x_size=140, y_min=0, y_size=140, z_min=0, z_size=80), name="appa_resize")

    topup = pe.Node(fsl.TOPUP(), name="topup")

    get_base_movpar = lambda x: x.split("_movpar.txt")[0]

    if acqp_file:
        topup.inputs.encoding_file = acqp_file
    else:
        topup.inputs.readout_times = [0.05, 0.05]
        wf.connect(
            [
                (inputnode, topup, [("encoding_directions", "encoding_direction")])
            ]
        )

    wf.connect(
        [
            # (
            #     inputnode,
            #     epi_flirt,
            #     [("altepi_file", "in_file"), ("epi_file", "reference")]
            # ),
            (inputnode, list_merge, [("epi_file", "in1")]),
            (inputnode, list_merge, [("altepi_file", "in2")]),
            (list_merge, merge, [("out", "in_files")]),
            (merge, appa_resize, [("merged_file", "in_file")]),
            (appa_resize, topup, [("roi_file", "in_file")]),
            (
                topup,
                outputnode,
                [
                    ("out_field", "out_fmap"),
                    ("out_movpar", "out_movpar"),
                    ("out_enc_file", "out_enc_file"),
                    (("out_movpar", get_base_movpar), "out_base")
                ]
            ),
        ]
    )

    return wf

#!/usr/bin/env python


def init_topup_wf(epi_fmaps):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import fsl, utility as niu

    wf = pe.Workflow(name="topup_prep_wf")

    PE_DIRECTION_MAPPING = {
        "i": "x",
        "i-": "x-",
        "j": "y",
        "j-": "y-",
        "k": "z",
        "k-": "z-",
    }

    epi_file = epi_fmaps[0][0]
    epi_dir = PE_DIRECTION_MAPPING[epi_fmaps[0][1]]
    altepi_file = epi_fmaps[1][0]
    altepi_dir = PE_DIRECTION_MAPPING[epi_fmaps[1][1]]

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["epi_file", "altepi_file"]), name="inputnode"
    )

    inputnode.epi_file = epi_file
    inputnode.altepi_file = altepi_file

    outputnode = pe.Node(niu.IdentityInterface(fields=["out_fmap"]), name="outputnode")

    list_merge = pe.Node(niu.Merge(numinputs=2), name="list_merge")

    topup = pe.Node(fsl.TOPUP(), name="topup")

    topup.inputs.encoding_direction = [epi_dir, altepi_dir]
    topup.inputs.readout_times = [0.05, 0.05]

    wf.connect(
        [
            (inputnode, list_merge, [("epi_file", "in1")], ("altepi_file", "in2")),
            (list_merge, topup, [("out", "in_file")]),
            (topup, outputnode, [("out_field", "out_fmap")]),
        ]
    )

    return wf

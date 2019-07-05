#!/usr/bin/env python

FMAP_PRIORITY = {"epi": 0, "fieldmap": 1, "phasediff": 2, "phase": 3, "syn": 4}


def init_sdc_prep_wf(
    subject_id, fmaps, metadata, layout, bids_dir, omp_nthreads=1, fmap_bspline=False
):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import utility as niu

    sdc_prep_wf = pe.Workflow(name="sdc_prep_wf")

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["b0_stripped", "bids_dir"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "out_fmap",
                "bold_ref",
                "bold_mask",
                "bold_ref_brain",
                "out_warp",
                "syn_bold_ref",
                "method",
            ]
        ),
        name="outputnode",
    )

    fmaps.sort(key=lambda fmap: FMAP_PRIORITY[fmap["suffix"]])
    fmap = fmaps[0]

    if fmap["suffix"] == "epi":

        epi_fmaps = [
            (fmap_["epi"], fmap_["metadata"]["PhaseEncodingDirection"])
            for fmap_ in fmaps
            if fmap_["suffix"] == "epi"
        ]

        dwi_pe_dir = metadata["PhaseEncodingDirection"]

        fmaps_matching_pe = []
        fmaps_opposite_pe = []

        for fmap, pe_dir in epi_fmaps:
            if pe_dir == dwi_pe_dir:
                fmaps_matching_pe.append(fmap)
            elif pe_dir[0] == dwi_pe_dir[0]:
                fmaps_opposite_pe.append(fmap)

        if not fmaps_opposite_pe:
            raise Exception(
                "No opposite pepolar image found for participant {}. "
                "Topup requires opposing pepolar images".format(subject_id)
            )

        from .topup import init_topup_wf

        topup_wf = init_topup_wf()
        topup_wf.inputs.inputnode.epi_fmaps = epi_fmaps

        # need to feed output not eddy topup instead of field
        sdc_prep_wf.connect(
            [topup_wf, outputnode, [("outputnode.out_fmap", "out_fmap")]]
        )

    if fmap["suffix"] == "fieldmap":
        from .fmap import init_fmap_wf

        fmap_wf = init_fmap_wf()
        fmap_wf.inputs.inputnode.fieldmap = fmap["fieldmap"]
        fmap_wf.inputs.inputnode.magnitude = fmap["magnitude"]

        sdc_prep_wf.connect(
            [
                (inputnode, fmap_wf, [("b0_stripped", "inputnode.b0_stripped")]),
                (fmap_wf, outputnode, [("outputnode.out_fmap", "out_fmap")]),
            ]
        )

    if fmap["suffix"] in ("phasediff", "phase"):
        from .phasediff import init_phase_wf
        from .fmap import init_fmap_wf

        phase_wf = init_phase_wf()
        if fmap["suffix"] == "phasediff":
            phase_wf.inputs.inputnode.phasediff = fmap["phasediff"]
        elif fmap["suffix"] == "phase":
            phase_wf.inputs.inputnode.phasediff = [fmap["phase1"], fmap["phase2"]]

        phase_wf.inputs.inputnode.magnitude1 = [
            fmap_ for key, fmap_ in sorted(fmap.items()) if key.startswith("magnitude1")
        ][0]

        phase_wf.inputs.inputnode.phases_meta = [
            layout.get_metadata(i) for i in phase_wf.inputs.inputnode.phasediff
        ]

        post_phase_wf = init_fmap_wf()

        sdc_prep_wf.connect(
            [
                (inputnode, post_phase_wf, [("b0_stripped", "inputnode.b0_stripped")]),
                (
                    phase_wf,
                    post_phase_wf,
                    [("outputnode.out_fmap", "inputnode.fieldmap")],
                ),
                (
                    phase_wf,
                    post_phase_wf,
                    [("outputnode.out_mag", "inputnode.magnitude")],
                ),
                (post_phase_wf, outputnode, [("outputnode.out_fmap", "out_fmap")]),
            ]
        )
    return sdc_prep_wf

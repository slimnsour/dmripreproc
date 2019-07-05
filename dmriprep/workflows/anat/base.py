#!/usr/bin/env python
import os


def init_anat_preproc_wf(subject_id, t1_file, metadata, layout, bids_dir):
    from nipype.pipeline import engine as pe
    from nipype.interfaces import freesurfer as fs, fsl, utility as niu
    from nipype.utils.filemanip import fname_presuffix

    import nibabel as nib

    bbreg = pe.Node(
        fs.BBRegister(contrast_type="t2", init="fsl", out_fsl_file=True, epi_mask=True),
        name="bbreg",
    )

    bbreg.inputs.subject_id = "freesurfer"  # bids_sub_name
    wf.connect(fslroi, "roi_file", bbreg, "source_file")

    voltransform = pe.Node(
        fs.ApplyVolTransform(inverse=True),
        iterfield=["source_file", "reg_file"],
        name="transform",
    )
    voltransform.inputs.subjects_dir = subjects_dir

    vt2 = voltransform.clone("transform_aparcaseg")
    vt2.inputs.interp = "nearest"

    def binarize_aparc(aparc_aseg):
        img = nib.load(aparc_aseg)
        data, aff = img.get_data(), img.affine
        outfile = fname_presuffix(
            aparc_aseg, suffix="bin.nii.gz", newpath=os.path.abspath("."), use_ext=False
        )
        nib.Nifti1Image((data > 0).astype(float), aff).to_filename(outfile)
        return outfile

    # wf.connect(inputspec2, "mask_nii", voltransform, "target_file")
    create_mask = pe.Node(
        niu.Function(
            input_names=["aparc_aseg"],
            output_names=["outfile"],
            function=binarize_aparc,
        ),
        name="bin_aparc",
    )

    def get_aparc_aseg(subjects_dir, sub):
        return os.path.join(subjects_dir, sub, "mri", "aparc+aseg.mgz")

    create_mask.inputs.aparc_aseg = get_aparc_aseg(subjects_dir, "freesurfer")
    wf.connect(create_mask, "outfile", voltransform, "target_file")

import sys
import time
from pathlib import Path
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
from typing import List

import numpy as np
import torch
import nibabel as nib
from impl_recon import train
from impl_recon.models import implicits
from impl_recon.utils import config_io, data_generation_auto, impl_utils, io_utils, patch_utils

Nifti = nib.Nifti1Image


def main():
    lumbar = Path("/media/data/robert/datasets/verse19/test_seg/reconstruction/thin_rn_eval_ax0_x3/prediction/")
    thorax = Path("/media/data/robert/datasets/verse19/test_seg/reconstruction/thin_rn_thorakal_eval_ax0_x3/prediction/")
    neck = Path("/media/data/robert/datasets/verse19/test_seg/reconstruction/thin_rn_hals_eval_ax0_x3/prediction/")
    out = Path("/media/data/robert/datasets/verse19/test_seg/reconstruction/thin_rn_combined_eval_ax0_x3/prediction/")
    out.mkdir(exist_ok=True, parents=True)
    for path in neck.glob("*.nii.gz"):
        try:
            name = path.name
            neck_nii: Nifti = nib.load(str(path))
            thorax_nii: Nifti = nib.load(str(Path(thorax, name)))
            lumbar_nii: Nifti = nib.load(str(Path(lumbar, name)))
            out_arr = np.array(neck_nii.dataobj) + np.array(thorax_nii.dataobj) + np.array(lumbar_nii.dataobj)
            nib.save(Nifti(out_arr, neck_nii.affine, neck_nii.header), str(Path(out, name)))
        except Exception as e:
            print(e)


if __name__ == "__main__":

    main()

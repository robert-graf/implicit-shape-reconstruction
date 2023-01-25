import math
import sys
from pathlib import Path
from typing import Union


file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))


from configargparse import ArgumentParser
import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
import recon.arguments as arguments
from recon.load import create_model, create_data_loader
from recon.train import ReconNetTraining
from preprocessing import extract_vertebra
from impl_recon.utils import nn_utils
import numpy as np
from BIDS import NII
import nibabel as nib

if __name__ == "__main__":
    # opt = get_opt()
    # LWH
    opt = arguments.ReconNetTraining_Option()
    # ALL
    # opt = arguments.ReconNetTraining_Option(model_name="recon_net_all", labels=list(range(1, 100)), batch_size_train=12, batch_size_val=12)

    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name=opt.data_basedir, best=True)
    assert last_checkpoint is not None
    net = ReconNetTraining.load_from_checkpoint(last_checkpoint)
    bbox_size = 128
    vert_nii = NII.load(
        Path("/media/data/robert/datasets/verse19/test_seg/sub-104852-30_acq-sag_run-1_seg-vert_msk.nii.gz"), True
    )  # Vert must not be a seg=True
    info = []
    size_full_image = 1
    for id in np.unique(vert_nii.get_array()):
        if id == 0:
            continue
        mask = np.equal(vert_nii.get_seg_array(), id).astype(np.uint8)
        vert_nii.seg = False
        mask = vert_nii.set_array(mask.astype(float), inplace=False).rescale_nib_((1, 1, 1), c_val=0).get_array()
        size_full_image = mask
        size, center_pos = extract_vertebra.get_fg_size_and_position(mask.astype(np.uint8))
        cropped_vert = extract_vertebra.crop_and_pad(mask, center_pos, bbox_size)
        with torch.no_grad():
            out = net(cropped_vert).cpu()
        info.append((out, size, center_pos, id))
    ## Revert prediction to ordinal
    out_nii = np.zeros(size_full_image)
    for arr_vert, size, position, vert_id in info:
        arr_vert = extract_vertebra.invert_crop_and_pad(arr_vert, size)
        from math import floor, ceil

        out_nii[
            position[0] - ceil(size[0] / 2) : position[0] + floor(size[0] / 2),
            position[1] - ceil(size[1] / 2) : position[1] + floor(size[1] / 2),
            position[2] - ceil(size[2] / 2) : position[2] + floor(size[2] / 2),
        ][arr_vert != 0] = (
            vert_id * arr_vert[arr_vert != 0]
        )
    # save_last
    vert_nii.rescale_nib_((1, 1, 1), c_val=0).set_array(out_nii, inplace=False).save("test.nii.gz")

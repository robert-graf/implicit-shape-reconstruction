import time
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional
from torch.utils import data
from torch import Tensor
from impl_recon.utils import config_io, geometry_utils, image_utils, io_utils

# Sparsification start IDs for training/validataion/test data
SPRSF_START_MAIN_SEED = 1
# Splitting data into training and validation
SPLIT_TRAIN_VAL_SEED = 2
# Sparsification start IDs for internal train/val in AD
SPRSF_START_SHARED_VAL_SEED = 3
# Selection of full-res subset from training data
SELECT_FULL_RES_SUBSET_SEED = 4

from recon.arguments import PhaseType, ReconNetTraining_Option


def load_volume(raw: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load axis-aligned volumes."""
    volume, affine = io_utils.load_nifti_file(raw)
    if not geometry_utils.is_matrix_scaling_and_transform(affine):
        raise ValueError(
            "Local to global image matrix is supposed to be 4x4, have scaling "
            "and translation components only, and positive scaling. Instead got\n"
            f"{affine}"
        )
    spacing = np.diagonal(affine)[:3].copy()
    return torch.from_numpy(volume).to(torch.float32), torch.from_numpy(spacing).to(torch.float32)


def random_crop(array: Tensor, array2: Tensor, crop_size):
    if crop_size > 0:
        assert all(shp >= crop_size for shp in array.shape)
        crop_start_x = torch.randint(0, array.shape[0] - crop_size + 1, [1]).item()
        crop_start_y = torch.randint(0, array.shape[1] - crop_size + 1, [1]).item()
        crop_start_z = torch.randint(0, array.shape[2] - crop_size + 1, [1]).item()
        crop_end_x = crop_start_x + crop_size
        crop_end_y = crop_start_y + crop_size
        crop_end_z = crop_start_z + crop_size
        return (
            array[crop_start_x:crop_end_x, crop_start_y:crop_end_y, crop_start_z:crop_end_z],
            array2[crop_start_x:crop_end_x, crop_start_y:crop_end_y, crop_start_z:crop_end_z],
        )
    return array, array2


# Wants:
# a,a,b -> Iso 1mm | a in [0.7,1.0] b in [2.7,3.5]
# Iso docker -> Iso 1 mm
# any docker -> Iso 1 mm
# any underpredicted -> Iso 1 mm
# 4 a,a,b  -> 1
# DataAug: subregion partial removal

# Dataset 1
# File Docker unscaled to iso -> GT
# File generation iso-ct with GT -> any-ct + docker -> upscale segmentation to iso -> extract each labels
# Folder structure:
# [Base]/[train|val|test]/[1-28]/*_[gt|00-99].nii.gz
# Req: all iso, ("P", "I", "R");
# Buffer? # https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960


# Dataset 2 full segmenation inverence
# any segmentation -> dict with all given labels in iso
# feed prediction in to DS and spit out iso-nii and input-scaling-nii

import warnings


class RNDataset(data.Dataset):
    """For training encoder-free implicit functions. Generated coordinates take spacing into
    account.
    """

    def __init__(
        self,
        opt: ReconNetTraining_Option,
        phase_type: PhaseType,
    ):

        # [Base]/[train|val|test]/[1-28]/*_[gt|00-99].nii.gz
        base_dir = Path(opt.data_basedir)
        self.opt = opt
        phase_folder = (
            opt.train_folder_name
            if phase_type == PhaseType.TRAIN
            else opt.val_folder_name
            if phase_type == PhaseType.VAL
            else opt.test_folder_name
        )
        label_folder = base_dir / phase_folder
        print(label_folder)
        self.cases: list[Path] = []
        for folder_number in opt.labels:
            self.cases.extend(label_folder.glob(f"{folder_number:02}/*.npz"))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, item: int):

        file = self.cases[item]
        # label_lr, spacing_lr = load_volume(raw)
        # offset_lr = Tensor([x / 2 for x in spacing_lr])
        try:
            raw = np.load(file)
            label_upsampled = torch.Tensor(raw["low_res"])
            label_gt = torch.Tensor(raw["high_res"])
        except Exception as ex:
            print(file)
            print(file)
            print(file)
            print(file)
            Path(file).unlink()
            del self.cases[item]
            return self.__getitem__(item + 1)

        # low_res=cropped_vert, high_res=cropped_vert_org)
        # label_gt, spacing_gt = load_volume(gt)
        # offset_gt = Tensor([x / 2 for x in spacing_gt])
        # Use sparse (1D) or dense (3D) volumes

        # label_upsampled = image_utils.interpolate_volume(
        #    label_lr,
        #    spacing_lr,
        #    offset_lr,
        #    torch.tensor(label_gt.shape),
        #    spacing_gt,
        #    offset_gt,
        #    "bilinear",
        # )
        (
            label_gt,
            label_upsampled,
        ) = random_crop(label_gt, label_upsampled, self.opt.crop_size)
        # Create channel dim
        label_gt = label_gt.unsqueeze(0)
        label_upsampled = label_upsampled.unsqueeze(0)

        return {
            "labels_lr": label_upsampled,
            "labels": label_gt,
            "spacings": (1, 1, 1),
            "casenames": file.stem,
        }

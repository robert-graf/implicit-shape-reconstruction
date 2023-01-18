from pathlib import Path
import time
from recon.arguments import ReconNetTraining_Option, TaskType, PhaseType
import torch
from impl_recon.models.implicits import AutoDecoder, ReconNet
from torch.utils.data import DataLoader
from recon.datasets import RNDataset


def create_model(params: ReconNetTraining_Option) -> torch.nn.Module:
    task_type = params.task_type

    net: torch.nn.Module
    if task_type == TaskType.AD:
        image_size = torch.Tensor(params.image_size)
        latent_dim = params.latent_dim
        op_num_layers = params.op_num_layers
        op_coord_layers = params.op_coord_layers
        net = AutoDecoder(latent_dim, len(image_size), image_size, op_num_layers, op_coord_layers)
    elif task_type == TaskType.RN:
        net = ReconNet()
    else:
        raise ValueError(f"Unknown task type {task_type}.")
    return net


def create_data_loader(params: ReconNetTraining_Option, phase_type: PhaseType, verbose: bool) -> tuple[DataLoader, int | None]:
    """For AD, use shared training and validation during train/val, and not during inferernce.
    For other tasks, there is no difference between validation and inference.
    """
    is_training = phase_type == PhaseType.TRAIN

    if is_training:
        do_shuffle = True
        data_name = "training data"
        batch_size = params.batch_size_train
        do_drop_last = True
    else:
        if params.batch_size_val != 1:
            print(
                f"Warning: validation employs full volumes, which may differ in shape between "
                f"examples. Therefore, running it with batch size {params.batch_size_val} > 1 "
                f"may lead to crashes."
            )
        do_shuffle = False
        data_name = "validation data" if phase_type == PhaseType.VAL else "test data"
        batch_size = params.batch_size_val
        do_drop_last = False
    num_workers = params.num_workers
    task_type = params.task_type
    num_examples_train = None
    if verbose:
        print(f"Loading {data_name} into memory...")
    t0 = time.time()
    if task_type == TaskType.AD:
        raise ValueError(f"Unknown task type {task_type}.")
        num_examples_train = len(ds.case_2_id)
    elif task_type == TaskType.RN:
        print('test')
        ds = RNDataset(params, phase_type)

    else:
        raise ValueError(f"Unknown task type {task_type}.")
    t1 = time.time()
    if verbose:
        print(f"Loading {data_name} done ({len(ds)} images): {t1 - t0:.2f}s")

    dl = DataLoader(ds, batch_size, do_shuffle, num_workers=num_workers, drop_last=do_drop_last)
    return dl, num_examples_train

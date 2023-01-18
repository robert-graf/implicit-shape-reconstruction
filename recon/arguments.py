from dataclasses import dataclass, asdict, field
from inspect import signature
from configargparse import ArgumentParser
from enum import Enum, auto
from typing import get_origin, get_args, Type
import types


class PhaseType(Enum):
    """Training, validation and inference."""

    TRAIN = auto()
    VAL = auto()
    INF = auto()


def enum2choices(enum: Type[Enum]) -> list[str]:
    choice = []
    for v in enum:
        choice.append(v.name)
    return choice


@dataclass()
class Option_to_Dataclass:
    # C = Literal[tuple(range(100))]
    from configargparse import ArgumentParser

    @classmethod
    def get_opt(cls, parser: None | ArgumentParser = None, config=None):

        keys = []
        if parser is None:
            p: ArgumentParser = ArgumentParser()
            p.add_argument("-config", "--config", is_config_file=True, default=config, help="config file path")
        else:
            p = parser

        # fetch the constructor's signature
        parameters = signature(cls).parameters
        cls_fields = sorted({field for field in parameters})
        # split the kwargs into native ones and new ones
        def n(s):
            return str(s).replace("<class '", "").replace("'>", "")

        for name in cls_fields:
            key = "--" + name
            if key in keys:
                continue
            else:
                keys.append(key)
            default = parameters[name].default
            annotation = parameters[name].annotation
            if get_origin(annotation) == types.UnionType:
                for i in get_args(annotation):
                    if i == types.NoneType:
                        default = None
                    else:
                        annotation = i
            if annotation is None:
                continue
            # print(annotation)
            # print(type(annotation))
            if annotation == bool:
                if default:
                    p.add_argument(key, action="store_true", default=False)
                else:
                    p.add_argument(key, action="store_false", default=True)

            elif isinstance(default, Enum) or issubclass(annotation, Enum):
                p.add_argument(key, default=default, choices=enum2choices(annotation))
            elif get_origin(annotation) == list or get_origin(annotation) == tuple:
                for i in get_args(annotation):
                    if i == types.NoneType:
                        default = None
                    else:
                        annotation = i
                p.add_argument(key, nargs="+", default=default, type=annotation, help="List of " + n(annotation))
            else:
                # print(annotation, key, default, annotation)
                p.add_argument(key, default=default, type=annotation, help=n(annotation))
        return p

    @classmethod
    def from_kwargs(cls, **kwargs):
        # fetch the constructor's signature
        parameters = signature(cls).parameters
        cls_fields = {field for field in parameters}
        # split the kwargs into native ones and new ones
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                if isinstance(parameters[name].default, Enum):
                    try:
                        if not isinstance(val, Enum):
                            val = parameters[name].annotation[val]
                    except KeyError as e:
                        print(f"Enum {type(parameters[name].default)} has no {val}")
                        exit(1)
                native_args[name] = val
            else:
                new_args[name] = val
        ret = cls(**native_args)
        # ... and add the new ones by hand
        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        return ret


class TaskType(Enum):
    """Type of task trained."""

    AD = auto()  # Auto-decoder with implicit functions
    RN = auto()  # Convolution auto-encoder (ReconNet)


@dataclass()
class ReconNetTraining_Option(Option_to_Dataclass):
    model_name: str = "reconnet"
    task_type: TaskType = TaskType.RN
    image_size: list[int] = field(default_factory=lambda: [256, 256, 256])
    # For RN: crop size for training. If 0, disabled.
    crop_size: int = 96
    # For AD/RN: thin out volumes by keeping every x slice. Should be at least 2.
    slice_step_size: int = 8
    # For AD/RN: axis for volume thinning in LPS coordinate system: sagittal, coronal, axial.
    slice_step_axis: int = 0
    # For AD/RN: whether to use averaged thick slices instead of exact thin slices.
    use_thick_slices: bool = False
    # For AD task: latent dimension
    latent_dim: int = 128
    # For AD: number of layers in OccupancyPredictor
    op_num_layers: int = 8
    # For AD: layers with coordinates in OccupancyPredictor
    op_coord_layers: list[int] = field(default_factory=lambda: [0, 4])
    gpus: list[int] | None = None
    batch_size_train: int = 3
    batch_size_val: int = 3
    # For AD: number of points per example per dim for training. If -1, use the whole volume.
    num_points_per_example_per_dim_train: int = 64
    # For AD: latent regularization weight. If 0, disabled.
    lat_reg_lambda: float = 1.0e-4
    # Scaled with batch size
    learning_rate: float = 1.0e-4
    # For AD: not scaled with batch size; only used for training.
    learning_rate_lat: float = 1.0e-3
    num_epochs: int = 80
    num_workers: int = 16
    prevent_nan: bool = False

    # DATASET
    train_folder_name: str = "train"
    val_folder_name: str = "val"
    test_folder_name: str = "test"
    data_basedir: str = "/media/data/robert/datasets/verse19/docker_ds/"  # "my_models"
    new: bool = False
    ds_len = None
    labels: list[int] = field(default_factory=lambda: list(range(20, 36)))


def get_latest_Checkpoint(opt: str | ReconNetTraining_Option, version="*", log_dir_name="lightning_logs", best=False) -> str | None:
    import glob
    import os

    ckpt = "*"
    if best:
        ckpt = "*best*"
    print()
    checkpoints = None

    if isinstance(opt, str) or not opt.new:
        if isinstance(opt, str):
            checkpoints = sorted(glob.glob(f"{log_dir_name}/{opt}/version_{version}/checkpoints/{ckpt}.ckpt"), key=os.path.getmtime)
        else:
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt.model_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )

        if len(checkpoints) == 0:
            checkpoints = None
        else:
            checkpoints = checkpoints[-1]
        print("Reload recent Checkpoint and continue training:", checkpoints)
    else:
        return None

    return checkpoints

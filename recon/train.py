# %reload_ext tensorboard
# tensorboard --logdir logs_diffusion
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

from impl_recon.utils import nn_utils


def create_loss() -> torch.nn.Module:
    return nn_utils.BCEWithDiceLoss("mean", 1.0)


class ReconNetTraining(pl.LightningModule):
    def __init__(self, opt: arguments.ReconNetTraining_Option, num_examples_train: int | None = None) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.opt = opt
        #### Initialize Models ####

        self.generator = create_model(opt)
        if opt.task_type == arguments.TaskType.AD:
            assert num_examples_train != None
            latent_dim = opt.latent_dim
            # Initialization scaling follows DeepSDF
            self.latents = torch.nn.Parameter(  # type:ignore
                torch.normal(0.0, 1 / math.sqrt(latent_dim), [num_examples_train, latent_dim], device=self.device)
            )
        else:
            self.latents: Tensor = torch.nn.Parameter(torch.empty(0))  # type:ignore

        self.criterion = create_loss().to(self.device)
        self.metric = nn_utils.DiceLoss(0.5, "sum", True).to(self.device)

        # This is a tensor so that it is mutable within other functions
        # self.global_step = torch.tensor(0, dtype=torch.int64)  # type: ignore
        self.num_epochs_trained = 0
        self.num_metrics_t = 0
        self.num_metrics = 0
        self.train_dice = []
        self.metric_avg_train = 1

    @torch.no_grad()
    def forward(self, batch) -> Tensor:
        return self.generator(batch)

    def configure_optimizers(self):
        opt = self.opt
        assert opt != None
        learning_rate = opt.learning_rate * opt.batch_size_train
        if opt.task_type == arguments.TaskType.AD:
            lr_lats = opt.learning_rate_lat
            return torch.optim.Adam(
                [
                    {"params": self.generator.parameters(), "lr": learning_rate},
                    {"params": self.latents, "lr": lr_lats},
                ]
            )
        else:
            return torch.optim.Adam(self.generator.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        """Optimize latent vectors for a single example.
        max_num_const_train_dsc: if train dice doesn't change this number of times, stop training. -1
                                means never stop early.
        """
        lat_reg = None
        labels = batch["labels"]
        if self.opt.task_type == arguments.TaskType.AD:
            latents_batch = self.latents[batch["caseids"]]
            coords = batch["coords"]
            labels_pred = self.generator(latents_batch, coords)
            lat_reg = torch.mean(torch.sum(torch.square(latents_batch), dim=1))
        elif self.opt.task_type == arguments.TaskType.RN:
            labels_lr = batch["labels_lr"]
            labels_pred = self.generator(labels_lr)
        else:
            raise ValueError(f"Unknown task type {self.opt.task_type}.")
        loss: Tensor = self.criterion(labels_pred, labels)
        if lat_reg is not None and self.opt.lat_reg_lambda > 0:
            # Gradually build up for the first 100 epochs (follows DeepSDF)
            loss += min(1.0, self.current_epoch / 100) * self.opt.lat_reg_lambda * lat_reg
        self.log("metric/train", loss.item())
        with torch.no_grad():
            self.num_metrics_t += labels_pred.shape[0]
            self.train_dice.append(self.metric(labels_pred, labels).item())
        return loss

    def training_epoch_end(self, outputs) -> None:
        metric_avg = Tensor(self.train_dice).sum() / self.num_metrics_t
        self.num_metrics_F = 0
        self.train_dice.clear()
        self.metric_avg_train = metric_avg
        # print(f"[train] metric {metric_avg:.3f}{'  '}")

        print(f"\r[train] metric {1-self.metric_avg_train:.3f}{'  ':100}")
        print(f"[val] metric {1-self.metric_avg_val:.3f}{' ':100}")

    def validation_step(self, batch, batch_idx, logger: TensorBoardLogger | None = None, verbose=True):
        if logger is None:
            tb_logger: TensorBoardLogger = self.logger  # type: ignore
        else:
            tb_logger = logger
        labels = batch["labels"].to(self.device)
        if self.opt.task_type == arguments.TaskType.AD:
            latents_batch = self.latents[batch["caseids"]]
            coords = batch["coords"].to(self.device)
            labels_pred = self.generator(latents_batch, coords)
        elif self.opt.task_type == arguments.TaskType.RN:
            labels_lr = batch["labels_lr"].to(self.device)
            labels_pred = self.generator(labels_lr)
        else:
            raise ValueError(f"Unknown task type {self.opt.task_type}.")

        # Metric returns the sum
        metric_running = self.metric(labels_pred, labels).item()
        self.num_metrics += batch["labels"].shape[0]

        if tb_logger is not None:
            self.log("metric/val", metric_running)

        return metric_running

    def validation_epoch_end(self, outputs: Union[Tensor, list[Tensor]]) -> None:
        metric_avg = Tensor(outputs).sum() / self.num_metrics
        self.num_metrics = 0
        self.metric_avg_val = metric_avg


def main(opt: arguments.ReconNetTraining_Option, limit_train_batches=1.0):
    #### Define transformation. ####

    #### Define dataset ####
    train_loader, num_examples_train = create_data_loader(opt, arguments.PhaseType.TRAIN, True)
    val_loader, _ = create_data_loader(opt, arguments.PhaseType.VAL, True)
    opt.ds_len = len(train_loader)  # type: ignore
    model = ReconNetTraining(opt=opt, num_examples_train=num_examples_train)
    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name=opt.data_basedir, best=True)

    # Define Last and best Checkpoints to be saved.

    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}-{train_All:.8f}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(200, len(train_loader)),
        save_top_k=3,
    )

    mc_best = ModelCheckpoint(
        monitor="metric/val",
        mode="max",
        filename="{epoch}-{step}-{train_All:.8f}_best",
        every_n_train_steps=min(200, len(train_loader)),
        save_top_k=2,
    )
    from pytorch_lightning.callbacks import Callback

    print(opt)
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger(opt.data_basedir, name=opt.model_name, default_hp_metric=False)
    limit_train_batches = limit_train_batches if limit_train_batches != 1 else None

    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
    elif -1 in gpus:
        gpus = None
        accelerator = "cpu"

    # training
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=1,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.num_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last],
        resume_from_checkpoint=last_checkpoint,
        detect_anomaly=opt.prevent_nan,
    )
    trainer.tune(model)
    trainer.fit(
        model,
        train_loader,
        val_loader,
    )


def get_opt(parser: None | ArgumentParser = None, config=None) -> arguments.ReconNetTraining_Option:
    torch.cuda.empty_cache()
    opt = arguments.ReconNetTraining_Option.get_opt(parser=parser, config=config)
    opt = arguments.ReconNetTraining_Option.from_kwargs(**opt.parse_args().__dict__)
    return opt


if __name__ == "__main__":
    # opt = get_opt()
    opt = arguments.ReconNetTraining_Option()
    main(opt)

import os
from typing import Optional

import torch
from lightning import pytorch as pl

from train.architectures import Architecture
from train.metrics import OnlinePsdRatio, PsdRatio
from train.plotting import plot_psds

Tensor = torch.Tensor


class DeepClean(pl.LightningModule):
    def __init__(
        self,
        arch: Architecture,
        loss: PsdRatio,
        metric: OnlinePsdRatio,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["arch", "loss", "metric"])

        self.model = arch
        self.loss = loss
        self.metric = metric
        self.metric.loss_fn = self.loss

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y = batch
        y_hat = self(X)
        loss = self.loss(y_hat, y).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, X, _):
        X, y_true = X
        if y_true is not None:
            self.metric.update(y_true[:, 0], "strain")
        if X is not None:
            y_pred = self(X)
            self.metric.update(y_pred, "predictions")

        # we could just log the self.metric object here
        # and let lightning take care of calling `.compute`,
        # but we'll opt for just updating the metric and
        # then computing the result ourselves at the end
        # fo the validation epoch so that we can add in
        # a little plotting of the ASD ratio vs. frequency

    def on_validation_epoch_end(self):
        # use our metric to produce the online-cleaned
        # noise prediction and strain timeseries, calling
        # compute to handle any distributed-training related
        # aggregation, then compute our loss functions on
        # these timeseries and log the output
        noise, strain = self.metric.compute(reduce=False)
        loss = self.loss(noise, strain)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            logger=True,
            prog_bar=True,
        )
        self.metric.reset()

        # use these timeseries to plot their ASDs
        # as well as their ratios
        step = str(self.trainer.global_step).zfill(5)
        fname = f"val-psds_{step}.html"
        plot_dir = os.path.join(self.trainer.logger.save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, fname)

        spectral_density = self.loss.spectral_density
        fftlength = spectral_density.nperseg / self.loss.sample_rate
        plot_psds(
            noise,
            strain,
            self.loss.mask,
            spectral_density,
            fftlength,
            self.loss.asd,
            fname=fname,
        )

    def configure_callbacks(self) -> list[pl.Callback]:
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )
        callbacks = [checkpoint]
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.hparams.patience,
                mode="min",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks

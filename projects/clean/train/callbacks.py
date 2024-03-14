import os

import h5py
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from train.plotting import plot_psds
from utils.plotting.utils import save


class PsdPlotter(Callback):
    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir or trainer.logger.save_dir

        # TODO: support s3 here
        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_test_start(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir or trainer.logger.save_dir
        self.test_dir = os.path.join(log_dir, "test")
        os.makedirs(self.test_dir, exist_ok=True)

    def log_plots(self, layout, fname, trainer):
        # always save the plots locally
        save(layout, fname, title="DeepClean PSDs")

        # if using W&B, log the plots as artifacts
        if isinstance(trainer.logger, WandbLogger):
            import wandb

            key = os.path.basename(fname).split("-")[0]
            html = wandb.Html(fname)
            trainer.logger.log_table(
                "samples", columns=[f"{key}-psds"], data=[[html]]
            )

    def _shared_eval(self, pl_module):
        # use our metric to produce the online-cleaned
        # noise prediction and strain timeseries, calling
        # compute to handle any distributed-training related
        # aggregation, then compute our loss functions on
        # these timeseries and log the output
        noise, strain = pl_module.metric.compute(reduce=False)
        pl_module.metric.reset()

        spectral_density = pl_module.loss.spectral_density
        fftlength = spectral_density.nperseg / pl_module.loss.sample_rate
        p = plot_psds(
            noise,
            strain,
            pl_module.loss.mask,
            spectral_density,
            fftlength,
            pl_module.loss.asd,
        )
        return noise, strain, p

    def on_validation_epoch_end(self, trainer, pl_module):
        # use our metric to produce the online-cleaned
        # noise prediction and strain timeseries, calling
        # compute to handle any distributed-training related
        # aggregation, then compute our loss functions on
        # these timeseries and log the output
        noise, strain, p = self._shared_eval(pl_module)
        loss = pl_module.loss(noise, strain)
        pl_module.log(
            "val_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            logger=True,
            prog_bar=True,
        )

        # use these timeseries to plot their ASDs
        # as well as their ratios
        step = str(trainer.global_step).zfill(5)
        fname = f"val-psds_step-{step}.html"
        fname = os.path.join(self.plot_dir, fname)
        self.log_plots(p, fname, trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        noise, strain, p = self._shared_eval(pl_module)
        loss = pl_module.loss(noise, strain)
        pl_module.log(
            "test_loss", loss, on_epoch=True, sync_dist=True, logger=True
        )

        fname = os.path.join(self.test_dir, "test-psds.html")
        self.log_plots(p, fname, trainer)

        fname = os.path.join(self.test_dir, "outputs.hdf5")
        with h5py.File(fname, "w") as f:
            f["noise"] = noise.cpu().numpy()
            f["strain"] = strain.cpu().numpy()


class ModelCheckpoint(ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        module = pl_module.__class__.load_from_checkpoint(
            self.best_model_path,
            arch=pl_module.model,
            metric=pl_module.metric,
            loss=pl_module.loss,
        )

        # TODO: we should probably establish an explicit
        # validation_kernel_length that matches what
        # we use at test time. If there were any issues
        # with it, we would have caught it by now during
        # validation, but worth making it explicit that
        # these are different values.
        datamodule = trainer.datamodule
        kernel_size = int(
            datamodule.hparams.kernel_length * datamodule.sample_rate
        )

        num_witnesses = len(datamodule.witness_channels)
        sample_input = torch.randn(1, num_witnesses, kernel_size)
        model = module.model.to("cpu")
        trace = torch.jit.trace(model, sample_input)

        save_dir = trainer.logger.log_dir or trainer.logger.save_dir
        if save_dir.startswith("s3://"):
            import s3fs

            s3 = s3fs.S3FileSystem()
            with s3.open(f"{save_dir}/model.pt", "wb") as f:
                torch.jit.save(trace, f)
        else:
            with open(os.path.join(save_dir, "model.pt"), "wb") as f:
                torch.jit.save(trace, f)
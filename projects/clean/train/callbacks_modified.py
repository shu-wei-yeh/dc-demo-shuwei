import os
import io
import numpy as np
import PIL.Image
import h5py
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from train.plotting import plot_psds
from utils.plotting.utils import save
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class PsdPlotter(Callback):
    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir or trainer.logger.save_dir
        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        # Initialize SummaryWriter only if TensorBoardLogger is used
        if isinstance(trainer.logger, TensorBoardLogger):
            self.writer = SummaryWriter(log_dir)

    def on_test_start(self, trainer, pl_module):
        log_dir = trainer.logger.log_dir or trainer.logger.save_dir
        self.test_dir = os.path.join(log_dir, "test")
        os.makedirs(self.test_dir, exist_ok=True)

    def log_plots(self, layout, fname, trainer):
        # Save layout (figure) to buffer, then convert to image and log to TensorBoard
        if isinstance(trainer.logger, TensorBoardLogger):
            buf = io.BytesIO()
            layout.savefig(buf, format='png')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image_tensor = np.array(image)
            # Convert image to CHW format expected by TensorBoard
            image_tensor = np.transpose(image_tensor, (2, 0, 1))
            self.writer.add_image("PSD Plots", image_tensor, trainer.global_step, dataformats='CHW')
        
        # Save plots locally and handle other loggers like WandbLogger if necessary.
        save(layout, fname, title="DeepClean PSDs")
        
        if isinstance(trainer.logger, WandbLogger):
            import wandb
            key = os.path.basename(fname).split("-")[0]
            html = wandb.Html(fname)
            trainer.logger.experiment.log({f"{key}-psds": wandb.Html(fname)})

    def on_fit_end(self, trainer, pl_module):
        if hasattr(self, 'writer'):
            self.writer.close()

    def _shared_eval(self, pl_module):
        noise, strain = pl_module.metric.compute(reduce=False)
        pl_module.metric.reset()

        spectral_density = pl_module.loss.spectral_density
        fftlength = spectral_density.nperseg / pl_module.loss.sample_rate
        # Assuming plot_psds now correctly returns a plot alongside noise and strain
        plot = plot_psds(noise, strain, pl_module.loss.mask, spectral_density, fftlength, pl_module.loss.asd)
        return noise, strain, plot


    def on_validation_epoch_end(self, trainer, pl_module):
        noise, strain, p = self._shared_eval(pl_module)
        loss = pl_module.loss(noise, strain).item()
        # Log using SummaryWriter if available
        if hasattr(self, 'writer'):
            self.writer.add_scalar("Loss/Validation", loss, trainer.global_step)
        else:
            # Fallback to PL's logging mechanism if TensorBoardLogger isn't being used
            pl_module.log("val_loss", loss, on_epoch=True, sync_dist=True, logger=True, prog_bar=True)
        fname = f"val-psds_step-{trainer.global_step}.html"
        self.log_plots(p, os.path.join(self.plot_dir, fname), trainer)

    def on_test_epoch_end(self, trainer, pl_module):
        noise, strain, p = self._shared_eval(pl_module)
        loss = pl_module.loss(noise, strain).item()
        # Log test loss using PL's logger
        pl_module.log("test_loss", loss, on_epoch=True, sync_dist=True, logger=True)
        fname = "test-psds.html"
        self.log_plots(p, os.path.join(self.test_dir, fname), trainer)
        # Save noise and strain data
        with h5py.File(os.path.join(self.test_dir, "outputs.hdf5"), "w") as f:
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
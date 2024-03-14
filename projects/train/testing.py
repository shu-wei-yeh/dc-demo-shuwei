# __main__.py
from train.cli import main

if __name__ == "__main__":
    main()

# callbacks.py
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

# cli.py
import os

import torch
from lightning.pytorch.cli import LightningCLI

from train.data import DeepCleanDataset
from train.model import DeepClean
from utils.logging import configure_logging


class AframeCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--verbose", type=bool, default=False)

        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.OneCycleLR)

        parser.link_arguments(
            "data.num_witnesses",
            "model.arch.init_args.num_witnesses",
            apply_on="instantiate",
        )

        # link data arguments to loss function
        parser.link_arguments(
            "data.sample_rate",
            "model.loss.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.freq_low", "model.loss.init_args.freq_low", apply_on="parse"
        )
        parser.link_arguments(
            "data.freq_high",
            "model.loss.init_args.freq_high",
            apply_on="parse",
        )

        # link data arguments to metric
        parser.link_arguments(
            "data.inference_sampling_rate",
            "model.metric.init_args.inference_sampling_rate",
            apply_on="parse",
        )
        parser.link_arguments(
            "data.sample_rate",
            "model.metric.init_args.sample_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.bandpass",
            "model.metric.init_args.bandpass",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.y_scaler",
            "model.metric.init_args.y_scaler",
            apply_on="instantiate",
        )

        # link optimizer and scheduler args
        parser.link_arguments(
            "data.steps_per_epoch",
            "lr_scheduler.steps_per_epoch",
            apply_on="instantiate",
        )
        parser.link_arguments("optimizer.lr", "lr_scheduler.max_lr")
        parser.link_arguments("trainer.max_epochs", "lr_scheduler.epochs")


def main(args=None):
    cli = AframeCLI(
        model_class=DeepClean,
        datamodule_class=DeepCleanDataset,
        seed_everything_default=101588,
        run=False,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},
        args=args,
    )

    log_dir = cli.trainer.logger.log_dir or cli.trainer.logger.save_dir
    if not log_dir.startswith("s3://"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "train.log")
        configure_logging(log_file)
    else:
        configure_logging()
    cli.trainer.fit(cli.model, cli.datamodule)

    cli.trainer.fit(cli.model, cli.datamodule)
    if cli.datamodule.hparams.test_duration > 0:
        cli.trainer.test(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()

# data.py

import logging

import h5py
import torch
from lightning import pytorch as pl
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter


# TODO: using this right now because
# lightning.pytorch.utilities.CombinedLoader
# is not supported when calling `.fit`. Once
# this has been fixed in
# https://github.com/Lightning-AI/lightning/issues/16830,
# we should switch to using a CombinedLoader for validation
class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return max(lengths)

    def __iter__(self):
        its = [iter(i) for i in self.datasets]
        for _ in range(len(self)):
            ret = []
            for it in its:
                try:
                    x = next(it)
                except (StopIteration, TypeError):
                    x = None
                ret.append(x)
            yield tuple(ret)


class DeepCleanDataset(pl.LightningDataModule):
    def __init__(
        self,
        fname: str,
        channels: list[str],
        kernel_length: float,
        freq_low: list[float],
        freq_high: list[float],
        batch_size: int,
        train_duration: float,
        test_duration: float,
        valid_frac: float,
        train_stride: float,
        inference_sampling_rate: float,
        start_offset: float = 0,
        filt_order: float = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        # infer the sample rate of the data from the
        # metadata of the provided file and use that
        # to ensure that we have enough data to train on
        with h5py.File(fname, "r") as f:
            dataset = f[channels[0]]
            self.sample_rate = 1 / dataset.attrs["dx"]
            total_duration = train_duration + test_duration

            size = int(total_duration * self.sample_rate)
            offset = int(start_offset * self.sample_rate)
            if (len(dataset) - offset) < size:
                datadur = len(dataset) / self.sample_rate
                raise ValueError(
                    "Dataset {} only contains {}s worth of data, "
                    "expected at least {}s to train with offset {}".format(
                        fname,
                        datadur,
                        total_duration,
                        self.hparams.start_offset,
                    )
                )

        # compute our number of batches per epoch
        # up front so that this can be communicated
        # to any other modules that need it, e.g.
        # learning rate scheduler
        train_size = int(train_duration * self.sample_rate)
        stride = int(train_stride * self.sample_rate)
        kernel_size = int(kernel_length * self.sample_rate)
        size = int(size * (1 - valid_frac))
        samples_per_epoch = (train_size - kernel_size) // stride + 1
        self.steps_per_epoch = int(samples_per_epoch // batch_size)

        # create some modules we'll use for pre/postprocessing
        self.X_scaler = ChannelWiseScaler(len(self.witness_channels))
        self.y_scaler = ChannelWiseScaler()
        self.bandpass = BandpassFilter(
            freq_low, freq_high, self.sample_rate, filt_order
        )

    @property
    def strain_channel(self):
        return self.hparams.channels[0]

    @property
    def witness_channels(self):
        return sorted(self.hparams.channels[1:])

    @property
    def num_witnesses(self):
        return len(self.hparams.channels) - 1

    @property
    def kernel_size(self):
        return int(self.hparams.kernel_length * self.sample_rate)

    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            y = batch[:, 0]
            X = batch[:, 1:]
            return X, y
        return batch

    def load_timeseries(self, split):
        train_size = int(self.hparams.train_duration * self.sample_rate)
        start = int(self.hparams.start_offset * self.sample_rate)
        if split == "test":
            start += train_size
            size = int(self.hparams.test_duration * self.sample_rate)
        else:
            size = train_size
        idx = slice(start, start + size)

        self.__logger.info(f"Loading {split} data")
        X = torch.zeros((self.num_witnesses, size))
        with h5py.File(self.hparams.fname, "r") as f:
            y = torch.Tensor(f[self.strain_channel][idx])
            for i, channel in enumerate(self.witness_channels):
                X[i] = torch.Tensor(f[channel][idx])
        return X, y

    def setup(self, stage):
        self.__logger = logging.getLogger("DeepClean Dataset")
        self.__logger.info(f"Inferred data sample rate {self.sample_rate}Hz")
        self.__logger.info(f"Setting up data for {stage} stage")

        if stage != "fit":
            self.test_X, self.test_y = self.load_timeseries("test")
            self.test_X = self.X_scaler(self.test_X)
            return

        # if we're training, split the data into
        # training and validation segments. Only
        # use integer-second length validation segments
        # since that's all we'll encounter at test time,
        # and some of the validation logic relies on this
        X, y = self.load_timeseries("train")
        valid_size = int(self.hparams.valid_frac * len(y))
        valid_length = int(valid_size / self.sample_rate)
        valid_size = int(valid_length * self.sample_rate)

        train_size = len(y) - valid_size
        split = [train_size, valid_size]
        self.__logger.info(
            "Training on first {} seconds, validating on "
            "remaining {} seconds".format(
                *[i / self.sample_rate for i in split]
            )
        )
        train_X, valid_X = torch.split(X, split, dim=1)
        train_y, valid_y = torch.split(y, split, dim=0)

        self.__logger.info("Preprocessing training data")
        # preprocess our inputs by standardizing
        # them to 0 mean unit variance across
        # each channel. Ignore any channels that
        # are constant in the training data
        self.X_scaler.fit(train_X)
        std = self.X_scaler.std
        std[std == 0] = 1

        self.train_X = self.X_scaler(train_X)
        self.valid_X = self.X_scaler(valid_X)

        # do the same to our output timeseries,
        # but bandpass filter the targets up front
        # in case we have a time-domain component
        # to our loss function. We have to do the
        # bandpass filtering back in numpy because
        # I can't get torchaudio to work properly
        self.y_scaler.fit(train_y)
        train_y = self.y_scaler(train_y)
        train_y = self.bandpass(train_y.numpy())
        self.train_y = torch.Tensor(train_y)

        # we don't need to do any preprocessing on the
        # validation target timeseries since we're
        # going to do the cleaning in the "true" space
        self.valid_y = valid_y
        self.__logger.info("Data loading complete")

    def train_dataloader(self):
        # iterate through our data as a single
        # tensor since this will be slightly faster
        X = torch.cat([self.train_y[None], self.train_X])

        # move our dataset onto the GPU up front since
        # each batch will be roughly the same amount of
        # data anyway. TODO: generalize for arbitrary
        # device ids in distributed training
        dataset = InMemoryDataset(
            X,
            kernel_size=self.kernel_size,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.steps_per_epoch,
            coincident=True,
            shuffle=True,
            device=f"cuda:{self.trainer.device_ids[0]}",
        )
        return dataset

    def _async_loader(self, X, y):
        """
        We don't need our inputs and our outputs to
        step at the same pace since in general
        we won't clean one stride at a time. So rather
        than create a single dataloader for both the
        inputs and the targets, create an input
        dataloader that steps at the rate at which we
        want data to go into the network, and an output
        one that steps at the rate at which we actually
        do cleaning (1s frames).
        """

        stride = int(self.sample_rate / self.hparams.inference_sampling_rate)
        witnesses = InMemoryDataset(
            X,
            kernel_size=int(self.sample_rate),
            stride=stride,
            batch_size=4 * self.hparams.batch_size,
            coincident=True,
            shuffle=False,
            device="cpu",
        )

        strain = InMemoryDataset(
            y[None],
            kernel_size=int(self.sample_rate),
            stride=int(self.sample_rate),
            batch_size=4 * self.hparams.batch_size,
            coincident=True,
            shuffle=False,
            device="cpu",
        )
        return ZippedDataset(witnesses, strain)

    def val_dataloader(self):
        return self._async_loader(self.valid_X, self.valid_y)

    def test_dataloader(self):
        return self._async_loader(self.test_X, self.test_y)
    
# metrics.py
from collections.abc import Callable
from typing import Optional

import torch
from ml4gw.transforms import SpectralDensity
from torchmetrics import Metric


class PsdRatio(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        freq_low: list[float],
        freq_high: list[float],
        overlap: Optional[float] = None,
        asd: bool = False,
    ) -> None:
        super().__init__()
        self.spectral_density = SpectralDensity(
            sample_rate,
            fftlength,
            overlap=overlap,
            average="median",
            fast=True,
        )
        self.asd = asd
        self.sample_rate = sample_rate

        N = int(fftlength * sample_rate / 2) + 1
        mask = torch.zeros((N,), dtype=torch.bool)
        for fl, fh in zip(freq_low, freq_high):
            low = int(fl * fftlength)
            high = int(fh * fftlength)
            mask[low : high + 1] = 1
        self.register_buffer("mask", mask)

    def forward(self, pred, strain):
        cleaned = strain - pred
        residual = self.spectral_density(cleaned.double())
        target = self.spectral_density(strain.double())

        ratio = residual / target
        ratio = ratio[:, self.mask]
        if self.asd:
            ratio = ratio**0.5
        loss = ratio.mean(dim=-1)
        return loss


class OnlinePsdRatio(Metric):
    def __init__(
        self,
        inference_sampling_rate: float,
        edge_pad: float,
        filter_pad: float,
        sample_rate: float,
        bandpass: Callable,
        y_scaler: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.stride = int(sample_rate / inference_sampling_rate)
        self.filter_pad = int(filter_pad * sample_rate)
        self.edge_pad = int(edge_pad * sample_rate)
        self.sample_rate = sample_rate

        self.loss_fn = None
        self.bandpass = bandpass
        self.y_scaler = y_scaler

        self.add_state("predictions", default=[])
        self.add_state("strain", default=[])

    def update(self, y, kind):
        if self.loss_fn is None:
            raise ValueError("Must provide loss_fn before calling update")
        getattr(self, kind).append(y)

    def clean(self):
        # first build our overlapping predictions
        # into a single timeseries of noise predictions
        size = sum([i.numel() for i in self.strain])
        batch_size = len(self.predictions[0])
        device = self.predictions[0].device
        dtype = self.predictions[0].dtype

        y_pred = torch.zeros(
            (size - self.edge_pad,), device=device, dtype=dtype
        )

        # for each predicted window, only slice out
        # the single stride of new data from it that's
        # sufficiently far from the edge to be considered
        # "safe" and place it in the corresponding position
        # in the full timeseries. We can do this array-style
        # with some fancy indexing. We'll start by building
        # the array of indices we'll grab from the predicted batches
        get_idx = torch.arange(self.stride, device=device)
        offset = int(self.sample_rate) - self.edge_pad - self.stride
        get_idx += offset

        # then turn this into a matrix of indices where
        # each row of predictions will go in the timeseries
        set_idx = get_idx.view(1, -1).repeat(batch_size, 1)
        batch_offset = torch.arange(batch_size, device=device)
        set_idx += batch_offset[:, None] * self.stride

        for i, y in enumerate(self.predictions):
            sidx = set_idx[: len(y)]
            y_pred[sidx + i * batch_size * self.stride] = y[:, get_idx]

            # for the very first frame, we have no choice
            # but to fill the left side with our predictions.
            # This won't really matter since we don't end up
            # measuring ourselves on this frame, but we'll need
            # it for providing filter padding.
            if not i:
                y_pred[:offset] = y[0, :offset]

        # now clean the target timeseries in the
        # online fashion, one frame at a time, plus
        # some filter settle-in padding on each side.
        # Ignore the first and last frames to account
        # for this filter settle-in.
        num_frames = int((len(y_pred) - self.filter_pad) // self.sample_rate)
        noise = []
        for i in range(1, num_frames - 1):
            start = int(i * self.sample_rate) - self.filter_pad
            stop = int((i + 1) * self.sample_rate) + self.filter_pad
            noise.append(y_pred[start:stop])

        # postprocess, doing the bandpass filtering back
        # in numpy because torchaudio won't work
        noise = torch.stack(noise)
        noise = self.y_scaler(noise, reverse=True)
        noise = self.bandpass(noise.cpu().numpy())
        noise = torch.tensor(noise, device=device)

        # slice out the filter padding so that the
        # frames in each row are no longer overlapping,
        # then reshape them to a proper timeseries
        noise = noise[:, self.filter_pad : -self.filter_pad]
        noise = noise.reshape(1, -1)

        # reshape our raw strain into a timeseries
        raw = torch.cat(self.strain, dim=0)[1 : num_frames - 1]
        raw = raw.view(1, -1)
        return noise, raw

    def compute(self, reduce: bool = True):
        noise, raw = self.clean()
        if reduce:
            return self.loss_fn(noise, raw).mean()
        return noise, raw

# model.py
from typing import Optional

import torch
from lightning import pytorch as pl

from train.architectures import Architecture
from train.callbacks import ModelCheckpoint, PsdPlotter
from train.metrics import OnlinePsdRatio, PsdRatio

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
        X, y_true = batch
        y_pred = self(X)
        loss = self.loss(y_pred, y_true).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _shared_eval_step(self, X, y_true) -> None:
        """
        Note that the actual computation of the loss function
        happens via the PsdPlotter Callback
        """

        if y_true is not None:
            self.metric.update(y_true[:, 0], "strain")
        if X is not None:
            y_pred = self(X)
            self.metric.update(y_pred, "predictions")

    def validation_step(self, batch, _) -> None:
        return self._shared_eval_step(*batch)

    def test_step(self, batch, _) -> None:
        return self._shared_eval_step(*batch)

    def configure_callbacks(self) -> list[pl.Callback]:
        # first callback actually computes all of our
        # validation metrics and any associated plots
        callbacks = [PsdPlotter()]

        # then tack on a checkpointer that uses these
        # metrcis for checkpointing the model
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="max",
        )
        callbacks.append(checkpoint)

        # if we specified an early-stopping patience
        # interval, add early stopping
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.hparams.patience,
                mode="min",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks
    
# plotting.py
from typing import Optional

import torch

from utils.plotting import plot_psds as plot_psds_


def get_psd(x, spectral_density: torch.nn.Module, asd: bool):
    x = x - x.mean()
    fft = torch.stft(
        x.double(),
        n_fft=spectral_density.nperseg,
        hop_length=spectral_density.nstride,
        window=spectral_density.window,
        normalized=False,
        center=False,
        return_complex=True,
    )
    fft = (fft * torch.conj(fft)).real
    stop = None if spectral_density.nperseg % 2 else -1
    fft[1:stop] *= 2
    fft *= spectral_density.scale
    if asd:
        fft = fft**0.5
    return fft.cpu().numpy()[0].T


@torch.no_grad()
def plot_psds(
    pred: torch.Tensor,
    strain: torch.Tensor,
    mask: torch.Tensor,
    spectral_density: torch.nn.Module,
    fftlength: float,
    asd: bool = True,
    fname: Optional[str] = None,
):
    mask = mask.cpu().numpy()
    cleaned = strain - pred

    cleaned = get_psd(cleaned, spectral_density, asd)[:, mask]
    raw = get_psd(strain, spectral_density, asd)[:, mask]
    pred = get_psd(pred, spectral_density, asd)[:, mask]

    freqs = torch.arange(len(mask)).cpu().numpy()
    freqs = freqs / fftlength
    freqs = freqs[mask]
    return plot_psds_(freqs, pred, raw, cleaned, asd=asd, fname=fname)
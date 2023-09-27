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
        duration: float,
        valid_frac: float,
        train_stride: float,
        valid_stride: float,
        offset: float = 0,
        filt_order: float = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.__logger = logging.getLogger("DeepClean Dataset")

        # infer the sample rate of the data from the
        # metadata of the provided file and use that
        # to ensure that we have enough data to train on
        with h5py.File(fname, "r") as f:
            # TODO: start using TimeSeries.write and then
            # sample_rate will be 1 / dataset.attrs["dx"]
            dataset = f[channels[0]]
            self.sample_rate = dataset.attrs["sample_rate"]
            self.__logger.info(f"Inferred sample rate {self.sample_rate}Hz")

            size = int(duration * self.sample_rate)
            offset = int(offset * self.sample_rate)
            if (len(dataset) - offset) < size:
                datadur = len(dataset) / self.sample_rate
                raise ValueError(
                    "Dataset {} only contains {}s worth of data, "
                    "expected at least {}s to train with offset {}".format(
                        fname, datadur, duration, self.hparams.offset
                    )
                )

        # compute our number of batches per epoch
        # up front so that this can be communicated
        # to any other modules that need it, e.g.
        # learning rate scheduler
        size = int(duration * self.sample_rate)
        stride = int(train_stride * self.sample_rate)
        kernel_size = int(kernel_length * self.sample_rate)
        size = int(size * (1 - valid_frac))
        samples_per_epoch = (size - kernel_size) // stride + 1
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

    def setup(self, stage):
        offset = int(self.hparams.offset * self.sample_rate)
        size = int(self.hparams.duration * self.sample_rate)

        # load in full training and validation data
        self.__logger.info("Loading training data")
        X = torch.zeros((self.num_witnesses, size))
        with h5py.File(self.hparams.fname, "r") as f:
            y = torch.Tensor(f[self.strain_channel][offset : offset + size])
            for i, channel in enumerate(self.witness_channels):
                X[i] = torch.Tensor(f[channel][offset : offset + size])

        # split it into training and validation segments
        valid_dur = self.hparams.valid_frac * self.hparams.duration
        valid_size = int(valid_dur * self.sample_rate)
        train_size = size - valid_size
        split = [train_size, valid_size]

        self.__logger.info(
            "Training on first {} seconds, validating on "
            "remaining {} seconds".format(
                self.hparams.duration - valid_dur, valid_dur
            )
        )
        train_X, valid_X = torch.split(X, split, dim=1)
        train_y, valid_y = torch.split(y, split, dim=0)

        self.__logger.info("Preprocessing training data")
        # preprocess our inputs by standardizing
        # them to 0 mean unit variance across
        # each channel
        self.X_scaler.fit(train_X)
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
        self.valid_y = valid_y
        self.__logger.info("Data loading complete")

    def train_dataloader(self):
        # iterate through our data as a single
        # tensor since this will be slightly faster
        X = torch.cat([self.train_y[None], self.train_X])
        dataset = InMemoryDataset(
            X,
            kernel_size=self.kernel_size,
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.steps_per_epoch,
            coincident=True,
            shuffle=True,
            device="cuda:0",
        )
        return dataset

    def val_dataloader(self):
        # We don't need our inputs and our outputs to
        # step at the same pace since in general
        # we won't clean one stride at a time. So rather
        # than create a single dataloader for both the
        # inputs and the targets, create an input
        # dataloader that steps at the rate at which we
        # want data to go into the network, and an output
        # one that steps at the rate at which we actually
        # do cleaning (1s frames).
        witnesses = InMemoryDataset(
            self.valid_X,
            kernel_size=int(self.sample_rate),
            stride=int(self.hparams.valid_stride * self.sample_rate),
            batch_size=4 * self.hparams.batch_size,
            coincident=True,
            shuffle=False,
            device="cpu",
        )

        strain = InMemoryDataset(
            self.valid_y[None],
            kernel_size=int(self.sample_rate),
            stride=int(self.sample_rate),
            batch_size=4 * self.hparams.batch_size,
            coincident=True,
            shuffle=False,
            device="cpu",
        )
        return ZippedDataset(witnesses, strain)

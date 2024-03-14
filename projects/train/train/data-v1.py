import logging

import h5py
import torch
from lightning import pytorch as pl
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
            # Apply bandpass filter to test_X and then scale it
            test_X_np = self.test_X.numpy()  # Convert to numpy for bandpass
            for i in range(test_X_np.shape[0]):
                test_X_np[i] = self.bandpass(test_X_np[i])  # Apply bandpass
            self.test_X = torch.tensor(test_X_np, dtype=torch.float32)  # Convert back to tensor
            self.test_X = self.X_scaler(self.test_X)  # Scale
            return

        # Loading and splitting data into training and validation
        X, y = self.load_timeseries("train")
        valid_size = int(self.hparams.valid_frac * len(y))
        valid_length = int(valid_size / self.sample_rate)
        valid_size = int(valid_length * self.sample_rate)
        train_size = len(y) - valid_size
        split = [train_size, valid_size]
        train_X, valid_X = torch.split(X, split, dim=1)
        train_y, valid_y = torch.split(y, split, dim=0)

        # Preprocessing training data
        self.__logger.info("Preprocessing training data")
        self.X_scaler.fit(train_X)

        # Apply bandpass filter to train_X and valid_X
        train_X_np = train_X.numpy()  # Convert to numpy for bandpass
        valid_X_np = valid_X.numpy()  # Convert to numpy for bandpass
        for i in range(train_X_np.shape[0]):
            train_X_np[i] = self.bandpass(train_X_np[i])  # Apply bandpass
        for i in range(valid_X_np.shape[0]):
            valid_X_np[i] = self.bandpass(valid_X_np[i])  # Apply bandpass
        # Convert back to tensors
        train_X = torch.tensor(train_X_np, dtype=torch.float32)
        valid_X = torch.tensor(valid_X_np, dtype=torch.float32)

        # Scale train_X and valid_X after bandpass filtering
        self.train_X = self.X_scaler(train_X)
        self.valid_X = self.X_scaler(valid_X)

        # Processing train_y with bandpass filter and scaling
        self.y_scaler.fit(train_y)
        train_y_np = train_y.numpy()  # Convert to numpy for bandpass
        train_y_np = self.bandpass(train_y_np)  # Apply bandpass
        self.train_y = torch.tensor(train_y_np, dtype=torch.float32)  # Convert back to tensor
        self.train_y = self.y_scaler(self.train_y)  # Scale

        # No preprocessing on validation target timeseries
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
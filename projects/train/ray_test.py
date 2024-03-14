# import os
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from train.data import DeepCleanDataset
# from train.model import DeepClean
# import torch

# from lightning import pytorch as pl
# from train.architectures import Architecture
# from train.metrics import OnlinePsdRatio, PsdRatio
# from train.callbacks import ModelCheckpoint, PsdPlotter
# from collections.abc import Callable
# from typing import Optional

# from ml4gw.transforms import SpectralDensity
# from torchmetrics import Metric

import os
import torch
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from train.model import DeepClean
from train.data import DeepCleanDataset
from train.architectures import Architecture
from train.metrics import OnlinePsdRatio, PsdRatio

# def train_deepclean(config, data_dir=None, num_epochs=10, num_gpus=0):
#     model = DeepClean(
#         arch=Architecture(),
#         loss=PsdRatio(
#             sample_rate=config["sample_rate"],
#             fftlength=config["fftlength"],
#             freq_low=config["freq_low"],
#             freq_high=config["freq_high"],
#             asd=config["asd"]
#         ),
#         metric=OnlinePsdRatio(
#             inference_sampling_rate=config["inference_sampling_rate"],
#             edge_pad=config["edge_pad"],
#             filter_pad=config["filter_pad"],
#             sample_rate=config["sample_rate"],
#             bandpass=Callable,  # Define or import your bandpass function
#             y_scaler=torch.nn.Module  # Define or import your scaler function
#         ),
#         patience=config["patience"],
#         save_top_k_models=config["save_top_k_models"]
#     )

#     data_module = DeepCleanDataset(
#         fname=os.path.join(data_dir, '/K-K1_lldata-1369291863-12288.hdf5'),
#         channels=['K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ', 'K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ'],
#         kernel_length=config["kernel_length"],
#         freq_low=[config["freq_low"]],
#         freq_high=[config["freq_high"]],
#         batch_size=int(config["batch_size"]),
#         train_duration=4096,
#         test_duration=8192,
#         valid_frac=0.33,
#         train_stride=0.0625,
#         inference_sampling_rate=64,
#         start_offset=0,
#         filt_order=8
#     )

#     trainer = Trainer(
#         logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
#         max_epochs=num_epochs,
#         gpus=num_gpus,
#         callbacks=[
#             TuneReportCallback({"loss": "val_loss"}, on="validation_end")
#         ]
#     )

#     trainer.fit(model, datamodule=data_module)

# def tune_hyperparameters(num_samples=10, num_epochs=10, gpus_per_trial=0):
#     config = {
#         "sample_rate": tune.choice([2048, 4096]),
#         "fftlength": tune.choice([2, 4]),
#         "freq_low": tune.choice([55, 58]),
#         "freq_high": tune.choice([62, 65]),
#         # "asd": tune.choice([True, False]),
#         "inference_sampling_rate": tune.choice([64, 4]),
#         # "edge_pad": tune.uniform(0.1, 0.5),
#         # "filter_pad": tune.uniform(0.1, 0.5),
#         "patience": tune.choice([10, 20]),
#         "save_top_k_models": tune.choice([1, 3]),
#         "batch_size": tune.choice([32, 512]),
#         "max_epochs": tune.choice([20, 100])
#     }

#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=num_epochs,
#         grace_period=1,
#         reduction_factor=2)

#     analysis = tune.run(
#         tune.with_parameters(
#             train_deepclean,
#             data_dir="/home/shuwei.yeh/deepclean/data",
#             num_epochs=num_epochs,
#             num_gpus=gpus_per_trial),
#         resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
#         metric="loss",
#         mode="min",
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=tune.CLIReporter(parameter_columns=list(config.keys()))
#     )

#     print("Best hyperparameters found were: ", analysis.best_config)

# if __name__ == "__main__":
#     # Set the number of samples and epochs for tuning
#     tune_hyperparameters(num_samples=10, num_epochs=10, gpus_per_trial=1)



############# test 2 
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    # "batch_size": tune.choice([16, 32, 64, 128]),
    "sample_rate": tune.choice([2048, 4096]),
    "fftlength": tune.choice([2, 4]),
    "freq_low": tune.choice([55, 58]),
    "freq_high": tune.choice([62, 65]),
    # "asd": tune.choice([True, False]),
    "inference_sampling_rate": tune.choice([64, 4]),
    # "edge_pad": tune.uniform(0.1, 0.5),
    # "filter_pad": tune.uniform(0.1, 0.5),
    "patience": tune.choice([10, 20]),
    "save_top_k_models": tune.choice([1, 3]),
    "batch_size": tune.choice([32, 512]),
    "max_epochs": tune.choice([20, 100])
}

def train_tune(config, checkpoint_dir=None, data_dir=None):
    # This example assumes the use of a single GPU.
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join("checkpoints", tune.get_trial_name())),
            TuneReportCallback({"loss": "val_loss"}, on="validation_end")
        ]
    )

    model = DeepClean(
        arch=Architecture(),  # Make sure to adapt this part accordingly
        loss=PsdRatio(
            sample_rate=4096, 
            fftlength=4, 
            freq_low=[55], 
            freq_high=[65],
            lr=config["lr"]  # Example of using tuned hyperparameter
        ),
        metric=OnlinePsdRatio(
            inference_sampling_rate=64,
            edge_pad=0.25,
            filter_pad=0.5,
            sample_rate=4096,
            bandpass=[55, 65],
            y_scaler=torch.nn.Module()
        ),
        batch_size=config["batch_size"]  # Example of using tuned hyperparameter
    )

    # Load data as per your setup
    data_module = DeepCleanDataset(
        fname=os.path.join(data_dir, '/K-K1_lldata-1369291863-12288.hdf5'),
        channels=['K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ', 'K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ'],
        kernel_length=config["kernel_length"],
        freq_low=[config["freq_low"]],
        freq_high=[config["freq_high"]],
        batch_size=int(config["batch_size"]),
        train_duration=4096,
        test_duration=8192,
        valid_frac=0.33,
        train_stride=0.0625,
        inference_sampling_rate=64,
        start_offset=0,
        filt_order=8
    )
    trainer.fit(model, datamodule=data_module)

# from ray.tune.integration.pytorch_lightning import TuneReportCallback

analysis = tune.run(
    train_tune,
    resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
    config=config,
    num_samples=10,  # Number of times to sample from the hyperparameter space
    # data_dir = "/home/shuwei.yeh/deepclean/data",
    local_dir="./ray_results",
    # Add additional parameters as needed
)

print("Best hyperparameters found were: ", analysis.best_config)


############## test 3
# Initialize Ray. Important if you're running this script directly.
if not ray.is_initialized():
    ray.init()

def train_tune(config, num_epochs=10, data_dir=None):
    model = DeepClean(
        arch=Architecture(),  # Ensure this is correctly initialized
        loss=PsdRatio(
            sample_rate=config["sample_rate"],
            fftlength=config["fftlength"],
            freq_low=config["freq_low"],
            freq_high=config["freq_high"],
            asd=config.get("asd", False)  # Assuming 'asd' is a boolean
        ),
        metric=OnlinePsdRatio(
            inference_sampling_rate=config["inference_sampling_rate"],
            edge_pad=config["edge_pad"],
            filter_pad=config["filter_pad"],
            sample_rate=config["sample_rate"],
            bandpass=lambda x: x,  # Placeholder for actual bandpass function
            y_scaler=torch.nn.Identity()  # Placeholder for actual scaler
        ),
        patience=config["patience"],
        save_top_k_models=config["save_top_k_models"]
    )

    data_module = DeepCleanDataset(
        fname=os.path.join(data_dir, 'K-K1_lldata-1369291863-12288.hdf5'),
        channels=['K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ', 'K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ'],
        # Ensure the rest of the parameters are set correctly
    )

    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
            ModelCheckpoint(dirpath=os.path.join(tune.get_trial_dir(), "checkpoints"))
        ]
    )

    trainer.fit(model, datamodule=data_module)

def tune_hyperparameters(num_samples=10, num_epochs=10, gpus_per_trial=0):
    config = {
        "sample_rate": tune.choice([2048, 4096]),
        "fftlength": tune.choice([2, 4]),
        "freq_low": tune.choice([55, 58]),
        "freq_high": tune.choice([62, 65]),
        "inference_sampling_rate": tune.choice([64, 128]),
        "edge_pad": tune.choice([0.25, 0.5]),
        "filter_pad": tune.choice([0.25, 0.5]),
        "patience": tune.choice([10, 20]),
        "save_top_k_models": tune.choice([1, 3]),
        "batch_size": tune.choice([32, 64]),
        # Add other parameters as needed
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            data_dir="/home/shuwei.yeh/deepclean/data"  # Specify the correct path
        ),
        # resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=tune.CLIReporter(parameter_columns=list(config.keys()))
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    tune_hyperparameters(num_samples=10, num_epochs=10, gpus_per_trial=1)

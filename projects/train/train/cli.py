import os

import torch
from lightning.pytorch.cli import LightningCLI

from deepclean.logging import configure_logging
from train.data import DeepCleanDataset
from train.model import DeepClean


class AframeCLI(LightningCLI):
    def before_instantiate_classes(self):
        save_dir = self.config.trainer.logger[0].init_args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "train.log")
        configure_logging(log_file)

    def add_arguments_to_parser(self, parser):
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
            "data.valid_stride",
            "model.metric.init_args.stride",
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
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    main()

import law
import luigi

from deepclean.base import DeepCleanTask
from deepclean.config import deepclean as Config


class Train(DeepCleanTask):
    data_fname = luigi.Parameter()
    output_dir = luigi.Parameter()

    cfg = Config()

    @property
    def command(self) -> list[str]:
        channels = [self.strain_channel] + self.witnesses
        return [
            self.python,
            "/opt/deepclean/projects/train/train",
            "--config",
            "/opt/deepclean/projects/train/config.yaml",
            "--data.fname",
            self.data_fname,
            "--trainer.logger.save_dir",
            self.output().path,
            "--data.channels",
            "[" + ",".join(channels) + "]",
            "--data.freq_low",
            str(self.cfg.freq_low),
            "--data.freq_high",
            str(self.cfg.freq_high),
        ]

    def output(self):
        return law.LocalDirectoryTarget(self.output_dir)

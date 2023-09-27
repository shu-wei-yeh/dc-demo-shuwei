import os

import law
import luigi

from deepclean.base import DeepCleanTask, root

sandbox_type = os.getenv("SANDBOX_TYPE", "singularity")
image_root = os.getenv("DEEPCLEAN_IMAGES", f"{root}/images")


class Train(DeepCleanTask):
    data_fname = luigi.Parameter()
    output_dir = luigi.Parameter()
    sandbox = f"{sandbox_type}::{image_root}/train.sif"

    def get_args(self):
        return [
            "--config",
            "/opt/deepclean/projects/train/config.yaml",
            "--data.fname",
            self.data_fname,
            "--trainer.logger.save_dir",
            self.output_dir,
            "--trainer.max_epochs",
            "1",
        ]

    def run(self):
        from train.cli import main

        main(self.get_args())

    def output(self):
        return law.LocalDirectoryTarget(self.output_dir)


class TrainWithInsert(Train):
    def run(self):
        import sys

        sys.path.insert(0, "/usr/local/lib/python3.10/site-packages")
        super().run()


class TrainWithSubprocess(Train):
    def run(self):
        import shlex
        import subprocess

        cmd = ["/usr/local/bin/python", "/opt/deepclean/projects/train/train"]
        cmd += self.get_args()

        try:
            proc = subprocess.run(
                cmd, capture_output=True, check=True, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Command '{}' failed with return code {} "
                "and stderr:\n{}".format(
                    shlex.join(e.cmd), e.returncode, e.stderr
                )
            ) from None
        print(proc.stdout)

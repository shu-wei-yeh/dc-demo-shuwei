from collections.abc import Callable
from pathlib import Path

import law
import luigi
from law.contrib import singularity

root = Path(__file__).resolve().parent.parent


class DeepCleanSandbox(singularity.SingularitySandbox):
    sandbox_type = "deepclean"

    def _get_volumes(self):
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[root] = "/opt/deepclean"
        return volumes


class DeepCleanTask(law.SandboxTask):
    dev = luigi.BoolParameter(default=False)
    gpus = luigi.Parameter(default="")

    @property
    def singularity_forward_law(self) -> bool:
        return False

    @property
    def singularity_allow_binds(self) -> bool:
        return True

    @property
    def singularity_args(self) -> Callable:
        def arg_getter():
            if self.gpus:
                return ["--nv"]
            return []

        return arg_getter

    def sandbox_env(self, env):
        if self.gpus:
            return {"CUDA_VISIBLE_DEVICES": self.gpus}
        return {}


config_defaults = {
    "deepclean_sandbox": {},
    "deepclean_sandbox_env": {},
    "deepclean_sandbox_volumes": {},
}
law.util.merge_dicts(
    law.Config._default_config, config_defaults, deep=True, inplace=True
)

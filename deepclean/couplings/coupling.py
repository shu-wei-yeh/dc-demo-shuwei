from collections import defaultdict
from dataclasses import dataclass

subtraction_problems = defaultdict(dict)


@dataclass
class Coupling:
    freq_low: float
    freq_high: float
    witnesses: list[str]

    def __post_init__(self):
        self._ifo = None

    @property
    def ifo(self):
        return self._ifo

    @property
    def channels(self):
        return [f"{self.ifo}:{i}" for i in self.witnesses]


class SubtractionProblemMeta(type):
    def __new__(cls, name, bases, dct):
        subclass = super().__new__(cls, name, bases, dct)
        try:
            name = subclass.name
        except AttributeError:
            name = subclass.__name__.replace("Sub", "")

        for key, value in subclass.__dict__.items():
            if isinstance(value, Coupling):
                value._ifo = key
                subtraction_problems[name][key] = value
        return subclass


class SubtractionProblem(metaclass=SubtractionProblemMeta):
    pass

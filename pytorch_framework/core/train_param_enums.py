from enum import Enum, unique, auto

@unique
class LoadStrategy(Enum):
    NONE = 0
    MODEL = auto()
    MODEL_OPTIMIZER = auto()

@unique
class SaveStrategy(Enum):
    NONE = 0
    MODEL = auto()
    MODEL_OPTIMIZER = auto()
    BEST_MODEL = auto()
    BEST_MODEL_OPTIMIZER = auto()

@unique
class MetricType(Enum):
    METRIC = 0
    LOSS = auto()

@unique
class TypeLoadModel(Enum):
    METRIC = 0
    LOSS = auto()
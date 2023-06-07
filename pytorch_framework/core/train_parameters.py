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

class TrainParameters:
    need_train = None
    reg = None
    optimizer = None
    model = None
    model_name = None
    learning_rate = None
    scheduler_epoch = None
    scheduler_coefficient = None
    epoch_num = 5

    # Checkpoint
    load_strategy = LoadStrategy.MODEL_OPTIMIZER
    save_strategy = SaveStrategy.BEST_MODEL_OPTIMIZER
    metric_type = MetricType.METRIC
    metric_value_stop = 0.9
    checkpoint_frequency = 5


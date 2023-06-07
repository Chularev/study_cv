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


class TrainParameters:
    need_train = None
    reg = None
    optimizer = None
    model = None
    model_name = None
    learning_rate = None
    scheduler_epoch = None
    scheduler_coefficient = None
    epoch_num = None

    # Checkpoint
    load_strategy = None
    save_strategy = None
    checkpoint_frequency = None


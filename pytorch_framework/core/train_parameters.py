from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType
from metrics.metrics import MyMetric
class TrainParameters:
    need_train = None
    reg = None
    optimizer = None
    model = None
    model_name = None
    learning_rate = None
    scheduler_epoch = None
    scheduler_coefficient = None
    epoch_num = 2

    # Checkpoint
    load_strategy = LoadStrategy.MODEL_OPTIMIZER
    save_strategy = SaveStrategy.BEST_MODEL_OPTIMIZER
    metric_type = MetricType.METRIC
    metric = MyMetric
    metric_value_stop = 0.9
    checkpoint_frequency = 1

    # Loaders
    # Train Loader
    t_loader_batch_size = 16
    t_loader_num_workers = 2
    t_loader_pin_memory = True
    t_loader_shuffle = True
    t_loader_drop_last = True

    # Val loader
    v_loader_batch_size = 32
    v_loader_num_workers = 2
    v_loader_pin_memory = True
    v_loader_shuffle = True
    v_loader_drop_last = True


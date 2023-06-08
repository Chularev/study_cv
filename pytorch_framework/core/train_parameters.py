from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType
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


from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType
from metrics.metrics import MyMetric
import torch.optim as optim
from models.Yolov1 import Yolov1

class TrainParameters:
    need_train = True
    reg = 0.0001
    optimizer = optim.Adam
    model = Yolov1
    model_name = 'best_net'
    learning_rate = 1e-3
    scheduler_epoch = 100
    scheduler_coefficient = 0.1
    weight_decay = 0
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


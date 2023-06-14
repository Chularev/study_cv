'''

        This class must be used only with core classes

'''


class _TrainContext:
    device = None
    datasets = None
    logger = None
    scheduler = None
    optimizer = None
    model = None
    train_loader = None
    val_loader = None
    epoch_num = None
    metric = None

    # Checkpoints
    metric_checkpointer = None

    '''
    load_strategy = None
    save_strategy = None
    checkpoint_frequency = None
    metric_type = None
    metric_value_stop = None
    '''


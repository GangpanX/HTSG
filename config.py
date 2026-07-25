import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.0003, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (800, 'maximum number of epochs to train for'),
        'weight-decay': (0.0, 'l2 regularization strength'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam]'),
        'patience': (500, 'patience for early stopping'),
        'seed': (2023, 'seed for training'),
        'log-freq': (10, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq':(2, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'print-epoch': (True, ''),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'mem-bank-size': (64, 'memory bank size per class'),
        'mem-bank-momentum': (0.95, 'memory bank momentum update'),
        'mem-bank-loss-w': (0.1, 'memory bank loss weight'),
        'mem-bank-temp': (0.15, 'memory bank temperature'),
    },
    'model_config': {
        'model': ('HyboNet', 'HyboNet encoder'),
        'dim': (32, 'embedding dimension'),
        'manifold': ('Lorentz', 'Lorentz manifold'),
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'),
        'hidden_size': (512, 'hidden size'),
        'num-layers': (3, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision'),
        'use_att': (1, 'whether to use hyperbolic attention or not'),
        'local-agg': (1, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('pheme', 'which dataset to use'),
        'feature_dim': (768, 'feature dimension'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (12345, 'seed for data splits (train/test/val)'),
    },
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)

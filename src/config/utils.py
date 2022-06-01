from typing import Dict


def get_sweep_config() -> Dict:
    sweep_config = {'method': 'random',
                    'metric': {'goal': 'minimize', 'name': 'loss'},
                    'parameters': {'batch_size': {'distribution': 'q_log_uniform_values',
                                                  'max': 256,
                                                  'min': 32,
                                                  'q': 8},
                                   'dropout': {'values': [0.3, 0.4, 0.5]},
                                   'epochs': {'value': 2},
                                   'fc_layer_size': {'values': [128, 256, 512]},
                                   'learning_rate': {'distribution': 'uniform',
                                                     'max': 0.1,
                                                     'min': 0},
                                   'optimizer': {'values': ['adam', 'sgd']}}}
    return sweep_config

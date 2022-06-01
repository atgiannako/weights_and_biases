import os

import wandb

from src.config.utils import get_sweep_config
from src.dataset.utils import build_dataset
from src.training.trainer import Trainer

wandb.login(host='http://localhost:8080', key=os.environ.get('WANDB_API_KEY'))


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = Trainer.build_network(config.fc_layer_size, config.dropout)
        optimizer = Trainer.build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = Trainer.train_epoch(network, loader, optimizer)
            wandb.log({'loss': avg_loss, 'epoch': epoch})


if __name__ == '__main__':
    sweep_id = wandb.sweep(get_sweep_config(), project="wandb-tutorial")
    wandb.agent(sweep_id, train, count=2)

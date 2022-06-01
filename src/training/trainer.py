import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    @staticmethod
    def build_network(fc_layer_size, dropout):
        # fully-connected, single hidden layer
        network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, fc_layer_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_layer_size, 10),
            nn.LogSoftmax(dim=1))

        return network.to(device)

    @staticmethod
    def build_optimizer(network, optimizer, learning_rate):
        if optimizer == 'sgd':
            optimizer = optim.SGD(network.parameters(),
                                  lr=learning_rate, momentum=0.9)
        elif optimizer == 'adam':
            optimizer = optim.Adam(network.parameters(),
                                   lr=learning_rate)
        return optimizer

    @staticmethod
    def train_epoch(network, loader, optimizer):
        cumu_loss = 0
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # ➡ Forward pass
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ Backward pass + weight update
            loss.backward()
            optimizer.step()

            wandb.log({'batch loss': loss.item()})

        return cumu_loss / len(loader)

import wandb

import torch
import hydra

from omegaconf import DictConfig


@hydra.main(config_path=".../configs/training", config_name="default_train_config")
def train(config: DictConfig):
    init_wandb(config)


def init_wandb(config):
    wandb.init(project=config.wandb_log.name, config=dict(config))

if __name__ == '__main__':
    """
    Basic train loop, takes in hydra config to specify train parameters.
    """
    train()
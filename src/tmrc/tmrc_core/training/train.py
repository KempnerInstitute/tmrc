import wandb
import torch
import hydra
import os
import threading
import time
from torch.utils.data import DataLoader, Dataset, random_split
from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm

from tmrc.tmrc_core.models import gpt
from tmrc.tmrc_core.utils.platform import Platform
from tmrc.tmrc_core.models.components import OPTIMIZER_REGISTRY
from tmrc.tmrc_core.models import MODEL_REGISTRY
from tmrc.tmrc_core.training import data

def save_model_periodic(model: torch.nn.Module, 
                       save_dir: str, 
                       interval: int,
                       stop_event: threading.Event):
    """Thread function to periodically save model"""
    while not stop_event.is_set():
        time.sleep(interval)
        save_path = os.path.join(save_dir, f"model_checkpoint_{int(time.time())}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

@hydra.main(config_path="/n/home01/nikhilanand/tmrc/configs/training", config_name="default_train_config")
def train(config: DictConfig):
    # Initialize wandb
    init_wandb(config)
    print(config)
    
    # Initialize platform and model
    platform = Platform()
    model = MODEL_REGISTRY.get(config.model.name)(config, platform)
    
    if platform.is_gpu:
        platform.move_to_device(model, device_index=0)
    
    # Create datasets
    train_loader, val_loader = data.create_dataloaders(config)

    # Initialize optimizer
    optimizer = OPTIMIZER_REGISTRY.get(config.optimizer.name)(
        params=model.parameters(),
        lr=config.optimizer.lr,
        betas=(config.optimizer.betas[0], config.optimizer.betas[1]),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay
    )
    
    # Set up model saving thread that should keep going in the background
    # if config.training.save_model:
    #     os.makedirs(config.training.artifacts_path, exist_ok=True)
    #     stop_save_thread = threading.Event()
    #     save_thread = threading.Thread(
    #         target=save_model_periodic,
    #         args=(model, config.training.artifacts_path, config.training.save_every, stop_save_thread)
    #     )
    #     save_thread.start()
    
    # Training loop
    try:
        steps_done = 0
        for epoch in range(config.training.epochs):
            model.train()
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, (x, y) in enumerate(train_iterator):
                if platform.is_gpu:
                    x = platform.move_to_device(x, device_index=0)
                    y = platform.move_to_device(y, device_index=0)
                
                optimizer.zero_grad()
                
                with torch.autocast(device_type=platform.get_device_str(), 
                                  dtype=getattr(torch, config.model.autocast_precision)):
                    _, loss = model(x, y)
                
                

                loss.backward()
                optimizer.step()
                
                # Logging
                if batch_idx % config.training.log_interval == 0:
                    print(f"loss: {loss:.4f}")
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "step": steps_done
                    })
                
                steps_done += 1
                if steps_done >= config.training.train_steps:
                    break
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    if platform.is_gpu:
                        x = platform.move_to_device(x, device_index=0)
                        y = platform.move_to_device(y, device_index=0)
                    
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({
                "val_loss": avg_val_loss,
                "epoch": epoch
            })
            
            if steps_done >= config.training.train_steps:
                break
    
    finally:
        # Cleanup
        if config.training.save_model:
            #stop_save_thread.set()
            #save_thread.join()
            # Save final model
            final_save_path = os.path.join(config.training.artifacts_path, "model_final.pt")
            torch.save(model.state_dict(), final_save_path)

# def init_wandb(config):
#     wandb.init(project=config.wandb_log.name, config=dict(config))

def init_wandb(config):
    try:
        wandb.init(project=config.wandb_log.name, config=dict(config))
    except KeyError as e:
        if str(e) == "'Name'":
            os.environ['WANDB_DISABLE_CODE'] = 'true'
            wandb.init(project=config.wandb_log.name, config=dict(config))
        else:
            raise

if __name__ == '__main__':
    """
    Basic train loop, takes in hydra config to specify train parameters.
    """
    train()
import wandb
import torch
import hydra
import os
import threading
import time
from omegaconf import DictConfig, OmegaConf
import argparse

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

def print_model_info(model):
    total_params, non_emb_params = model.get_num_params(), model.get_num_params(non_embedding=True)
    print(f"Model parameters: {total_params}, non-embedding parameters: {non_emb_params}")
    wandb.log({"total_params": total_params, "non_emb_params": non_emb_params})
    return total_params, non_emb_params

@hydra.main(version_base=None)
def train(config: DictConfig):
    # Initialize wandb
    init_wandb(config)
    print(OmegaConf.to_yaml(config))
    
    # Initialize platform and model
    platform = Platform()
    model = MODEL_REGISTRY.get(config.model.name)(config, platform)
    print_model_info(model)
    
    if platform.is_gpu:
        platform.move_to_device(model, device_index=0)
    
    # Create datasets
    train_loader, val_loader = data.create_dataloaders(config)
    print(f"There are {len(train_loader)} batches in the training set")

    # Initialize optimizer
    optimizer = OPTIMIZER_REGISTRY.get(config.optimizer.name)(
        params=model.parameters(),
        lr=config.optimizer.lr,
        betas=(config.optimizer.betas[0], config.optimizer.betas[1]),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay
    )
    scaler = torch.amp.GradScaler(device=platform.get_device_str())

    # Training loop
    try:
        steps_done = 0
        start = None
        for epoch in range(config.training.epochs):
            model.train()
            
            for batch_idx, sample in enumerate(train_loader):
                tok_ids = sample.get("token_ids").long()
                doc_ids = sample.get("document_ids").long()

                y = torch.roll(tok_ids, shifts=-1, dims=1)
                y[:, -1] = -100 

                if platform.is_gpu:
                    x = platform.move_to_device(tok_ids, device_index=0)
                    y = platform.move_to_device(y, device_index=0)
                    doc_ids = platform.move_to_device(doc_ids, device_index=0)
                else:
                    x = tok_ids

                
                optimizer.zero_grad()
                
                with torch.autocast(device_type=platform.get_device_str(), 
                                  dtype=getattr(torch, config.model.autocast_precision)):
                      _, loss = model(x, y, doc_ids)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Logging
                if batch_idx % config.training.log_interval == 0:
                    if start is not None:
                        end = time.time()
                        print(f"Time to process: {(end - start):.2f} seconds")
                    start = time.time()
                    print(f"@ batch index {batch_idx}, train loss: {loss:.4f}")
                    wandb.log({
                        "train_loss": loss.item(),
                        "epoch": epoch,
                        "step": steps_done
                    })
                
                steps_done += 1
            
                # validation
                if steps_done % config.training.val_interval == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for sample in val_loader:
                            tok_ids = sample.get("token_ids").long()
                            y = torch.roll(tok_ids, shifts=-1, dims=1)
                            y[:, -1] = -100 

                            if platform.is_gpu:
                                x = platform.move_to_device(tok_ids, device_index=0)
                                y = platform.move_to_device(y, device_index=0)
                                doc_ids = platform.move_to_device(doc_ids, device_index=0)
                            else:
                                x = tok_ids
                            
                            _, loss = model(x, y, doc_ids)
                            val_losses.append(loss.item())
                            print(f"validation loss: {loss:.4f}")
                            wandb.log({
                                "val_loss": loss.item(),
                                "epoch": epoch,
                                "step": steps_done
                            })

                if steps_done >= config.training.train_steps:
                    break
            
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
            # create directory if it doesn't exist
            os.makedirs(config.training.artifacts_path, exist_ok=True)
            final_save_path = os.path.join(config.training.artifacts_path, "model_final.pt")
            torch.save(model.state_dict(), final_save_path)


def init_wandb(config):
    try:
        wandb.init(project=config.wandb_log.name, config=dict(config))
    except KeyError as e:
        if str(e) == "'Name'":
            os.environ['WANDB_DISABLE_CODE'] = 'true'
            wandb.init(project=config.wandb_log.name, config=dict(config))
        else:
            raise

def get_config_path():
    parser = argparse.ArgumentParser(description='Training script with customizable config path')
    parser.add_argument('--config-path', 
                       type=str,
                       default="../../../../configs/training",
                       help='Path to the config directory')
    parser.add_argument('--config-name',
                       type=str,
                       default="default_train_config",
                       help='Name of the config file (without .yaml extension)')
    args = parser.parse_args()
    return args.config_path, args.config_name

if __name__ == '__main__':
    """
    Basic train loop, takes in hydra config to specify train parameters.
    Config path and name can be customized via command line arguments.
    """
    config_path, config_name = get_config_path()
    hydra.initialize(version_base=None, config_path=config_path)
    config = hydra.compose(config_name=config_name)
    train(config)
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import numpy as np
from pathlib import Path
import os
from typing import Iterator, Optional, List, Tuple
import random
from itertools import cycle
from tatm.data import get_dataset, torch_collate_fn


def create_dataloaders(config):
    full_dataset = get_dataset(config.datasets.path, context_length = config.model.context_length)

    print(f"Dataset length: {len(full_dataset)}")

    #save last 5% for validation
    val_size = int(len(full_dataset) * 0.01)

    print(f"Validation size: {val_size}")
    #train_dataset = [full_dataset[d] for d in range(len(full_dataset)) if d < len(full_dataset)-val_size]
    val_dataset = [full_dataset[d] for d in range(10)]

    print("Creating dataloaders")
    train_loader = DataLoader(
        full_dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=torch_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=torch_collate_fn
    )
    
    return train_loader, val_loader
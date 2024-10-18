import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
import numpy as np
from pathlib import Path
import os
from typing import Iterator, Optional, List, Tuple
import random
from itertools import cycle

class StreamingBinaryDataset(IterableDataset):
    """This is a temporary implementation of a streaming binary dataset
    it will be depcrecated in the future in favor of `tatm` but exists
    right now for testing purposes until I sort out a `tatm` dependency issue!"""
    def __init__(self, 
                 data_dir: str, 
                 context_length: int,
                 chunk_size: int = 100_000,  # Number of tokens to load at once
                 buffer_size: int = 10_000):  # Size of shuffle buffer
        super().__init__()
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.files = sorted([f for f in self.data_dir.glob("*.bin")])
        
        if not self.files:
            raise ValueError(f"No .bin files found in {data_dir}")
            
        self.total_size = sum(os.path.getsize(f) // 2 for f in self.files)  # uint16 = 2 bytes
        
    def _load_chunk(self, file_path: Path, offset: int = 0) -> np.ndarray:
        """Load a chunk of data from a file starting at offset"""
        with open(file_path, 'rb') as f:
            f.seek(offset * 2)  # multiply by 2 because uint16
            data = np.fromfile(f, dtype=np.uint16, count=self.chunk_size)
        return data
    
    def _get_file_chunks(self, file_path: Path) -> Iterator[np.ndarray]:
        """Generate chunks from a single file"""
        file_size = os.path.getsize(file_path) // 2  # size in tokens
        for offset in range(0, file_size, self.chunk_size):
            chunk = self._load_chunk(file_path, offset)
            if len(chunk) < self.context_length + 1:
                continue
            yield chunk
            
    def _get_sequences(self, chunk: np.ndarray) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate sequences of context_length + 1 from a chunk"""
        for i in range(0, len(chunk) - self.context_length):
            sequence = chunk[i:i + self.context_length + 1]
            x = torch.from_numpy(sequence[:-1]).long()
            y = torch.from_numpy(sequence[1:]).long()
            yield x, y
            
    def _shuffle_buffer(self, iterator: Iterator, buffer_size: int) -> Iterator:
        """Shuffle stuff using a buffer"""
        buffer = []
        
        # Fill buffer
        try:
            for _ in range(buffer_size):
                item = next(iterator)
                buffer.append(item)
        except StopIteration:
            pass
        
        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            item = buffer.pop(idx)
            try:
                new_item = next(iterator)
                buffer.append(new_item)
            except StopIteration:
                pass
            yield item
            
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        
        if worker_info is not None:
            per_worker = int(np.ceil(len(files) / worker_info.num_workers))
            worker_id = worker_info.id
            files = files[worker_id * per_worker:(worker_id + 1) * per_worker]
        
        file_iter = cycle(files)
        
        def sequence_generator():
            while True:  # Continue indefinitely
                current_file = next(file_iter)
                for chunk in self._get_file_chunks(current_file):
                    yield from self._get_sequences(chunk)
                    
        iterator = sequence_generator()
        
        if self.buffer_size > 0:
            iterator = self._shuffle_buffer(iterator, self.buffer_size)
            
        return iterator

class ValidationBinaryDataset(Dataset):
    """Fixed-size dataset for validation"""
    def __init__(self, 
                 data_dir: str, 
                 context_length: int,
                 max_size: int = 10_000):  # Maximum number of sequences to use for validation
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.max_size = max_size
        
        # Load a fixed amount of data for validation
        self.data = self._load_validation_data()
        
    def _load_validation_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        files = sorted([f for f in self.data_dir.glob("*.bin")])
        if not files:
            raise ValueError(f"No .bin files found in {self.data_dir}")
            
        sequences = []
        tokens_needed = self.max_size * (self.context_length + 1)
        
        with open(files[0], 'rb') as f:  # Use first file for validation
            data = np.fromfile(f, dtype=np.uint16, count=tokens_needed)
            
        # Create sequences
        for i in range(0, len(data) - self.context_length):
            if len(sequences) >= self.max_size:
                break
                
            sequence = data[i:i + self.context_length + 1]
            x = torch.from_numpy(sequence[:-1]).long()
            y = torch.from_numpy(sequence[1:]).long()
            sequences.append((x, y))
            
        return sequences
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

def create_dataloaders(config):
    train_dataset = StreamingBinaryDataset(
        data_dir=config.datasets.path,
        context_length=config.model.context_length,
        chunk_size=100_000,
        buffer_size=10_000
    )
    
    val_dataset = ValidationBinaryDataset(
        data_dir=config.datasets.path,
        context_length=config.model.context_length,
        max_size=10_000 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
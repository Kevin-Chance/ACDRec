import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NpyEmbeddingDataset(Dataset):

    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        self.array = np.load(self.path, mmap_mode='r')
        if self.array.ndim != 2:
            raise ValueError(f'embedding array must be [N,D], got {self.array.shape}')
        self.length = len(self.array)
        self.dim = int(self.array.shape[1])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.array[index], dtype=np.float32, copy=True))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requested = torch.device(name)
    if requested.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False')
    return requested

@torch.no_grad()
def collect_indices(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        indices, _ = model.get_indices(batch.to(device, non_blocking=True), use_sk=False)
        chunks.append(indices.reshape(-1, indices.shape[-1]).cpu().numpy())
    return np.concatenate(chunks, axis=0)

def code_metrics(indices):
    n = len(indices)
    unique = len({tuple((int(v) for v in row)) for row in indices})
    return dict(num_items=n, unique_codes=unique, collision_rate=(n - unique) / max(n, 1))

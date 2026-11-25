import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

def duplicate_groups(data):
    table, groups = ({}, [])
    for i in range(len(data)):
        row = np.asarray(data[i], dtype=np.float32).copy()
        if not np.isfinite(row).all() or not np.any(row):
            raise ValueError(f'CCL requires finite nonzero inputs; invalid item {i}')
        row[row == 0] = 0
        key = row.tobytes()
        if key not in table:
            table[key] = len(table)
        groups.append(table[key])
    groups = np.asarray(groups, dtype=np.int64)
    return torch.from_numpy(groups)

class GroupedDataset(Dataset):

    def __init__(self, data, groups):
        if len(data) != len(groups):
            raise ValueError('Duplicate labels must cover every input')
        self.data, self.groups = (data, groups.tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (self.data[i], self.groups[i])

def grouped_loss(inputs, reconstructed, groups, group_size=128, temperature=0.1):
    if inputs.ndim != 2 or inputs.shape != reconstructed.shape or len(groups) != len(inputs):
        raise ValueError('Expected matching [B,D] tensors and B group labels')
    if len(inputs) == 0 or group_size < 2 or (not math.isfinite(temperature)) or (temperature <= 0):
        raise ValueError('Invalid CCL batch/group/temperature')
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        x = F.normalize(inputs.detach().float(), dim=-1)
        y = F.normalize(reconstructed.float(), dim=-1)
        total = y.sum() * 0
        full = len(x) // group_size * group_size

        def loss_sum(a, b, ids):
            scores = torch.bmm(a, b.transpose(1, 2)) / temperature
            positive = ids.unsqueeze(2).eq(ids.unsqueeze(1))
            return (scores.logsumexp(-1) - scores.masked_fill(~positive, -torch.inf).logsumexp(-1)).sum()
        if full:
            total = total + loss_sum(x[:full].reshape(-1, group_size, x.shape[-1]), y[:full].reshape(-1, group_size, y.shape[-1]), groups[:full].reshape(-1, group_size))
        if full < len(x):
            total = total + loss_sum(x[full:][None], y[full:][None], groups[full:][None])
        return total / len(x)

def loss_for_batch(model, batch, device, config):
    x, groups = batch
    x, groups = (x.to(device), groups.to(device))
    out, qloss, _, _ = model(x)
    original, recon = model.compute_loss(out, qloss, xs=x)
    auxiliary = grouped_loss(x, out, groups, config['group_size'], config['temperature'])
    loss = original + config['weight'] * auxiliary
    stats = dict(train_loss=loss.detach(), recon_loss=recon.detach(), quant_loss=qloss.detach(), ccl_loss=auxiliary.detach())
    return (loss, stats, len(x))

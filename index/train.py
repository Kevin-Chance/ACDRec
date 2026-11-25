import argparse
import json
import math
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from index.common import NpyEmbeddingDataset, set_seed, resolve_device, collect_indices, code_metrics
from index.models.adaptive import ACDRecTokenizer
from index.contrastive import duplicate_groups, GroupedDataset, loss_for_batch

def build_parser():
    p = argparse.ArgumentParser(description='Train an ACDRec modality tokenizer')
    p.add_argument('--data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--num_emb_list', type=int, nargs='+', default=[256] * 4)
    p.add_argument('--e_dim', type=int, default=32)
    p.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64])
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--tau_item', type=float, default=1.0)
    p.add_argument('--tau_code', type=float, default=0.1)
    p.add_argument('--eta', type=float, default=0.1)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--kmeans_iters', type=int, default=100)
    p.add_argument('--init_items', type=int, default=20000)
    p.add_argument('--ccl_weight', type=float, default=0.03)
    p.add_argument('--ccl_temperature', type=float, default=0.1)
    p.add_argument('--ccl_group_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=0.0001)
    p.add_argument('--eval_step', type=int, default=2)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    return p

def make_model(args, in_dim):
    keys = ['num_emb_list', 'e_dim', 'layers', 'beta', 'kmeans_iters', 'tau_item', 'tau_code', 'eta', 'iters']
    return ACDRecTokenizer(in_dim=in_dim, **{key: getattr(args, key) for key in keys})

def save_checkpoint(path, model, optimizer, scheduler, args, epoch, metrics):
    torch.save(dict(model_type='acdrec_tokenizer_v1', loss_reduction='squared_l2_layer_sum_batch_mean', model_config=model.model_config, state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), args=vars(args), epoch=epoch, metrics=metrics), path)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = code_metrics(collect_indices(model, loader, device))
    recon, count = (0.0, 0)
    for batch in loader:
        x = batch.to(device)
        out, qloss, _, _ = model(x, use_sk=False)
        _, rloss = model.compute_loss(out, qloss, xs=x)
        recon += float(rloss) * len(x)
        count += len(x)
    metrics['hard_recon_loss'] = recon / count
    return metrics

def run(args):
    if min(args.epochs, args.batch_size, args.eval_step, args.init_items, args.kmeans_iters) < 1:
        raise ValueError('Training sizes and intervals must be positive')
    if not math.isfinite(args.ccl_weight) or args.ccl_weight <= 0:
        raise ValueError('CCL weight must be finite and positive')
    if not math.isfinite(args.ccl_temperature) or args.ccl_temperature <= 0 or args.ccl_group_size < 2:
        raise ValueError('Invalid CCL temperature or group size')
    config = dict(weight=args.ccl_weight, temperature=args.ccl_temperature, group_size=args.ccl_group_size)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    set_seed(args.seed)
    device = resolve_device(args.device)
    data = NpyEmbeddingDataset(args.data_path)
    if len(data) < max(args.num_emb_list) or math.prod(args.num_emb_list) < len(data):
        raise ValueError('Codebook sizes do not fit the item catalog')
    output.mkdir(parents=True, exist_ok=True)
    groups = duplicate_groups(data)
    loader = DataLoader(GroupedDataset(data, groups), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = make_model(args, data.dim).to(device)
    count = min(max(args.init_items, max(args.num_emb_list)), len(data))
    ids = np.random.default_rng(args.seed).choice(len(data), count, replace=False)
    init_x = torch.stack([data[int(i)] for i in ids]).to(device)
    model.initialize_codebooks(init_x, seed=args.seed)
    del init_x
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    evaluate(model, eval_loader, device)
    best_loss, best_key, stale = (math.inf, (math.inf, math.inf), 0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, seen = ({}, 0)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats, n = loss_for_batch(model, batch, device, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf'), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value) * n
            seen += n
        row = {key: value / seen for key, value in totals.items()}
        row['epoch'] = epoch
        if epoch % args.eval_step == 0 or epoch == args.epochs:
            row.update(evaluate(model, eval_loader, device))
            key = (row['collision_rate'], row['hard_recon_loss'])
            if key < best_key:
                best_key, stale = (key, 0)
                save_checkpoint(output / 'best_collision_model.pth', model, optimizer, scheduler, args, epoch, row)
            else:
                stale += 1
            if row['hard_recon_loss'] < best_loss:
                best_loss = row['hard_recon_loss']
                save_checkpoint(output / 'best_loss_model.pth', model, optimizer, scheduler, args, epoch, row)
        save_checkpoint(output / 'latest_model.pth', model, optimizer, scheduler, args, epoch, row)
        print(json.dumps(row), flush=True)
        if args.patience > 0 and stale >= args.patience:
            break
if __name__ == '__main__':
    run(build_parser().parse_args())

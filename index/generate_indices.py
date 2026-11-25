import argparse
import json
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from index.common import NpyEmbeddingDataset, resolve_device
from index.indexing import prefix_tokens, resolve_collisions
from index.models.adaptive import ACDRecTokenizer

def build_model(checkpoint):
    if checkpoint.get('model_type') != 'acdrec_tokenizer_v1':
        raise ValueError('Expected an ACDRec tokenizer checkpoint')
    model = ACDRecTokenizer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description='Export collision-free ACDRec semantic IDs')
    p.add_argument('--data_path', required=True)
    p.add_argument('--ckpt_path', required=True)
    p.add_argument('--output_file', required=True)
    p.add_argument('--content', choices=['text', 'image'], required=True)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--candidate_width', type=int, default=32)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    output = Path(args.output_file).resolve()
    metadata_path = output.with_suffix(output.suffix + '.meta.json')
    if output.exists() or metadata_path.exists():
        raise FileExistsError(output)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint).to(device)
    data = NpyEmbeddingDataset(args.data_path)
    if data.dim != model.in_dim or not len(data):
        raise ValueError('Embedding dimensions do not match the tokenizer')
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    codes, distances = ([], [])
    for x in loader:
        idx, d = model.get_indices(x.to(device), use_sk=False)
        if not all((torch.isfinite(d[:, level, :size]).all() for level, size in enumerate(model.num_emb_list))):
            raise FloatingPointError('Non-finite raw index distances')
        codes.append(idx.cpu().numpy())
        distances.append(d.cpu().numpy())
    raw, distances = (np.concatenate(codes), np.concatenate(distances))
    final, _ = resolve_collisions(raw, distances, model.num_emb_list, args.candidate_width)
    prefixes = [prefix_tokens(args.content, len(model.num_emb_list), level) for level in range(len(model.num_emb_list))]
    mapping = {str(i): [prefixes[level].format(int(code)) for level, code in enumerate(row)] for i, row in enumerate(final)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2), encoding='utf8')
    metadata_path.write_text(json.dumps(dict(model_config=model.model_config, content=args.content, num_items=len(mapping)), indent=2), encoding='utf8')
if __name__ == '__main__':
    main()

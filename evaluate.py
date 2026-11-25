import argparse
import json
import math
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from ensemble import fuse_scores
from index.common import resolve_device, set_seed
from inference import generate_branch, load_catalog


def ranking_metrics(items, target):
    rank = next((i + 1 for i, item in enumerate(items) if item == target), None)
    metrics = {f'hit@{k}': float(rank is not None and rank <= k) for k in (1, 5, 10)}
    metrics.update({
        f'ndcg@{k}': 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
        for k in (5, 10)
    })
    return metrics


def ranked_items(items, scores):
    pairs = [(item, score if item != -1 else -1000) for item, score in zip(items, scores)]
    return [item for item, _ in sorted(pairs, key=lambda pair: pair[1], reverse=True)]


def evaluation_examples(interactions, catalog):
    if not isinstance(interactions, dict) or not interactions:
        raise ValueError('Interactions must be a nonempty user-to-item-sequence mapping')
    examples = []
    for user, items in interactions.items():
        if not isinstance(items, list) or len(items) < 2:
            raise ValueError(f'User {user} needs a history and a held-out target')
        if any(str(item) not in catalog for item in items):
            raise ValueError(f'User {user} contains an item outside the SID catalog')
        examples.append((items[:-1], int(items[-1])))
    return examples


@torch.no_grad()
def evaluate(args):
    if min(args.batch_size, args.num_beams) < 1 or args.num_beams < 10:
        raise ValueError('Batch size must be positive and at least 10 beams are required')
    output = Path(args.output_file).resolve()
    if output.exists():
        raise FileExistsError(output)
    set_seed(args.seed)
    device = resolve_device(args.device)
    folder = Path(args.data_path) / args.dataset
    text_catalog = load_catalog(folder / (args.dataset + args.index_file))
    image_catalog = load_catalog(folder / (args.dataset + args.image_index_file))
    if set(text_catalog) != set(image_catalog):
        raise ValueError('Text and image catalogs must contain the same items')
    interactions = json.loads((folder / (args.dataset + '.inter.json')).read_text(encoding='utf8'))
    examples = evaluation_examples(interactions, text_catalog)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).to(device).eval()
    totals = {branch: {metric: 0.0 for metric in ranking_metrics([], 0)}
              for branch in ('text', 'image', 'fused')}
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        histories = [history for history, _ in batch]
        text = generate_branch(
            model, tokenizer, histories, text_catalog, device, args.num_beams, args.max_his_len
        )
        image = generate_branch(
            model, tokenizer, histories, image_catalog, device, args.num_beams, args.max_his_len
        )
        for (_, target), (text_items, text_scores), (image_items, image_scores) in zip(batch, text, image):
            fused = fuse_scores(text_items, text_scores, image_items, image_scores)
            rankings = {
                'text': ranked_items(text_items, text_scores),
                'image': ranked_items(image_items, image_scores),
                'fused': [item for item, _ in fused],
            }
            for branch, items in rankings.items():
                for metric, value in ranking_metrics(items, target).items():
                    totals[branch][metric] += value
    results = {
        'dataset': args.dataset,
        'split': 'test',
        'num_users': len(examples),
        'num_beams': args.num_beams,
        **{branch: {metric: value / len(examples) for metric, value in values.items()}
           for branch, values in totals.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ACDRec on every held-out test target')
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--index_file', required=True)
    parser.add_argument('--image_index_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    evaluate(parser.parse_args())

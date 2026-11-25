import argparse
import json
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from index.common import resolve_device
from generation_trie import Trie, prefix_allowed_tokens_fn
from ensemble import fuse_scores

def load_catalog(path):
    mapping = json.loads(Path(path).read_text(encoding='utf8'))
    if not mapping or len({tuple(code) for code in mapping.values()}) != len(mapping):
        raise ValueError('SID catalogs must be nonempty and unique')
    if len({len(code) for code in mapping.values()}) != 1:
        raise ValueError('SID catalog contains inconsistent code lengths')
    return mapping

@torch.no_grad()
def generate_branch(model, tokenizer, histories, catalog, device, num_beams=20, max_his_len=20):
    reverse = {''.join(code): int(item) for item, code in catalog.items()}
    candidate_trie = Trie([[model.config.decoder_start_token_id] + tokenizer.encode(code) for code in reverse])
    texts = []
    for history in histories:
        history = history[-max_his_len:] if max_his_len > 0 else history
        texts.append(''.join((''.join(catalog[str(item)]) for item in history)))
    inputs = tokenizer(texts, return_tensors='pt', padding='longest', truncation=False, max_length=tokenizer.model_max_length, return_attention_mask=True)
    if inputs['input_ids'].shape[1] > tokenizer.model_max_length:
        raise ValueError('SID history exceeds the model maximum length')
    output = model.generate(**inputs.to(device), max_new_tokens=max(10, len(next(iter(catalog.values()))) + 1), prefix_allowed_tokens_fn=prefix_allowed_tokens_fn(candidate_trie), num_beams=num_beams, num_return_sequences=num_beams, output_scores=True, return_dict_in_generate=True, early_stopping=True)
    predictions = [text.strip().replace(' ', '') for text in tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)]
    scores = output['sequences_scores'].cpu().tolist()
    rows = []
    for row in range(len(histories)):
        start, end = (row * num_beams, (row + 1) * num_beams)
        rows.append(([reverse.get(code, -1) for code in predictions[start:end]], scores[start:end]))
    return rows

@torch.no_grad()
def recommend(args):
    device = resolve_device(args.device)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path).to(device).eval()
    text_catalog, image_catalog = (load_catalog(args.text_index), load_catalog(args.image_index))
    if set(text_catalog) != set(image_catalog):
        raise ValueError('Text and image catalogs must contain the same items')
    histories = json.loads(Path(args.history_file).read_text(encoding='utf8'))
    users = list(histories)
    results = {}
    for start in range(0, len(users), args.batch_size):
        batch_users = users[start:start + args.batch_size]
        batch = [histories[user] for user in batch_users]
        text = generate_branch(model, tokenizer, batch, text_catalog, device, args.num_beams, args.max_his_len)
        image = generate_branch(model, tokenizer, batch, image_catalog, device, args.num_beams, args.max_his_len)
        for user, (text_items, text_scores), (image_items, image_scores) in zip(batch_users, text, image):
            ranking = fuse_scores(text_items, text_scores, image_items, image_scores)
            results[user] = [dict(item=item, score=score) for item, score in ranking if item != -1][:args.top_k]
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf8')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ACDRec recommendations from observed item histories')
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--text_index', required=True)
    parser.add_argument('--image_index', required=True)
    parser.add_argument('--history_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if min(args.num_beams, args.top_k, args.batch_size) < 1:
        parser.error('Beam count, top-k and batch size must be positive')
    recommend(args)

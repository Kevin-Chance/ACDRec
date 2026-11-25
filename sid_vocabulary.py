import json
from pathlib import Path
from index.indexing import prefix_tokens

def catalog_structure(indices, model_config):
    import re
    sizes = model_config['num_emb_list']
    if not sizes or len(sizes) > 8 or (not indices):
        raise ValueError('Invalid or empty SID catalog')
    modality = None
    for row in indices.values():
        if len(row) != len(sizes):
            raise ValueError('SID length differs from tokenizer structure')
        for level, (token, size) in enumerate(zip(row, sizes)):
            match = re.fullmatch('<([a-hA-H])_(\\d+)>', token)
            if not match or not 0 <= int(match[2]) < size or match[1].lower() != 'abcdefgh'[level]:
                raise ValueError('SID symbol outside its level/codebook')
            current = 'text' if match[1].islower() else 'image'
            modality = current if modality is None else modality
            if modality != current:
                raise ValueError('Mixed SID modalities')
    return dict(sid_length=len(sizes), codebook_sizes=sizes, modality=modality)

def prepare(tokenizer, args):
    folder = Path(args.data_path) / args.dataset
    paths = {m: folder / (args.dataset + getattr(args, key)) for m, key in [('text', 'index_file'), ('image', 'image_index_file')]}
    metas = {m: p.with_suffix(p.suffix + '.meta.json') for m, p in paths.items()}
    if not any((p.exists() for p in metas.values())):
        return None
    if not all((p.exists() for p in metas.values())):
        raise ValueError('Both SID metadata files are required')
    metadata = {m: json.loads(p.read_text(encoding='utf8')) for m, p in metas.items()}
    indices = {m: json.loads(p.read_text(encoding='utf8')) for m, p in paths.items()}
    if set(indices['text']) != set(indices['image']):
        raise ValueError('SID catalogs differ')
    structures = {m: catalog_structure(indices[m], metadata[m]['model_config']) for m in paths}
    if any((s['modality'] != m for m, s in structures.items())):
        raise ValueError('SID modality mismatch')
    sizes = structures['text']['codebook_sizes']
    if sizes != structures['image']['codebook_sizes']:
        raise ValueError('ACDRec requires the same text/image codebook structure')
    if any((len({tuple(v) for v in ix.values()}) != len(ix) for ix in indices.values())):
        raise ValueError('Final SID catalogs must be unique')
    old_vocab = tokenizer.get_vocab().copy()
    tokens = [prefix_tokens(m, len(sizes), level).format(i) for m in ['text', 'image'] for level, size in enumerate(sizes) for i in range(size)]
    tokenizer.add_tokens(tokens)
    if any((tokenizer.convert_tokens_to_ids(t) != idx for t, idx in old_vocab.items())):
        raise ValueError('Old token IDs changed')
    for token in tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids != [tokenizer.convert_tokens_to_ids(token)] or ids[0] == tokenizer.unk_token_id:
            raise ValueError('SID symbol must be one known token: ' + token)
    for m in indices:
        code = ''.join((prefix_tokens(m, len(sizes), level).format(size - 1) for level, size in enumerate(sizes)))
        if len(tokenizer.encode(code)) != len(sizes) + 1:
            raise ValueError('SID sequence must contain L symbols followed by EOS')
    return dict(sid_length=len(sizes), codebook_sizes=sizes, vocabulary_size=len(tokenizer))

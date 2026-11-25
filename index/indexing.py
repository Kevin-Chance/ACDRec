import heapq
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np

def prefix_tokens(content: str, levels: int, index: int) -> str:
    text_prefix = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    upper_prefix = [p.upper() for p in text_prefix]
    prefix = upper_prefix if content == 'image' else text_prefix
    if index >= len(prefix):
        raise ValueError(f'support up to {len(prefix)} code levels, got {levels}')
    return f'<{prefix[index]}_{{}}>'

def candidate_codes(raw: Sequence[int], distances: np.ndarray, sizes: Sequence[int], width: int):
    levels = len(sizes)
    top_lists: List[np.ndarray] = []
    for level, size in enumerate(sizes):
        valid = np.asarray(distances[level, :size], dtype=np.float64)
        width_l = min(max(int(width), 1), size)
        top_lists.append(np.argsort(valid, kind='stable')[:width_l])
    start = tuple((0 for _ in range(levels)))

    def score(position: Tuple[int, ...]) -> float:
        return float(sum((distances[level, int(top_lists[level][position[level]])] for level in range(levels))))
    heap: List[Tuple[float, Tuple[int, ...]]] = [(score(start), start)]
    seen = {start}
    while heap:
        _, position = heapq.heappop(heap)
        code = tuple((int(top_lists[level][position[level]]) for level in range(levels)))
        yield code
        for level in range(levels):
            if position[level] + 1 >= len(top_lists[level]):
                continue
            nxt = list(position)
            nxt[level] += 1
            nxt_tuple = tuple(nxt)
            if nxt_tuple not in seen:
                seen.add(nxt_tuple)
                heapq.heappush(heap, (score(nxt_tuple), nxt_tuple))

def resolve_collisions(indices: np.ndarray, distances: np.ndarray, sizes: Sequence[int], width: int) -> Tuple[np.ndarray, Dict[str, int]]:
    final = [tuple((int(v) for v in row)) for row in indices]
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for item, code in enumerate(final):
        groups[code].append(item)
    occupied = set(groups.keys())
    collision_groups = [items for items in groups.values() if len(items) > 1]
    reassigned = 0
    max_group = max((len(g) for g in collision_groups), default=1)
    for items in sorted(collision_groups, key=lambda group: (-len(group), group[0])):
        items = sorted(items, key=lambda i: (float(sum((distances[i, l, final[i][l]] for l in range(len(sizes))))), i))
        raw_code = final[items[0]]
        occupied.discard(raw_code)
        occupied.update([final[items[0]]])
        for item in items[1:]:
            found = None
            for code in candidate_codes(final[item], distances[item], sizes, width):
                if code not in occupied:
                    found = code
                    break
            if found is None:
                raise RuntimeError(f'unable to resolve collision for item {item}; increase --candidate_width (current={width}) or change codebook sizes')
            final[item] = found
            occupied.add(found)
            reassigned += 1
    out = np.asarray(final, dtype=np.int64)
    if len({tuple(row) for row in out}) != len(out):
        raise AssertionError('collision resolution failed: codes are not unique')
    return (out, {'initial_collision_groups': len(collision_groups), 'max_initial_group': max_group, 'reassigned_items': reassigned})

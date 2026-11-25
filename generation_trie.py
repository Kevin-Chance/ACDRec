class Trie:

    def __init__(self, sequences):
        self.trie_dict = {}
        for sequence in sequences:
            self._add_to_trie(sequence, self.trie_dict)

    @staticmethod
    def _add_to_trie(sequence, trie_dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    def get(self, prefix_sequence):
        node = self.trie_dict
        for token in prefix_sequence:
            if token not in node:
                return []
            node = node[token]
        return list(node.keys())

def prefix_allowed_tokens_fn(candidate_trie):

    def prefix_allowed_tokens(batch_id, sentence):
        return candidate_trie.get(sentence.tolist())
    return prefix_allowed_tokens

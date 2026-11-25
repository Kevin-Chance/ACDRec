class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d['input_ids'] for d in batch]
        label_texts = [d['labels'] for d in batch]
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding='longest', max_length=self.tokenizer.model_max_length, truncation=not getattr(self.args, 'strict_sid_length', False), return_attention_mask=True)
        labels = self.tokenizer(label_texts, return_tensors='pt', padding='longest', max_length=self.tokenizer.model_max_length, truncation=not getattr(self.args, 'strict_sid_length', False), return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        if max(inputs['input_ids'].shape[1], inputs['labels'].shape[1]) > self.tokenizer.model_max_length:
            raise ValueError('SID sequence exceeds maximum length; refusing silent truncation')
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        return inputs

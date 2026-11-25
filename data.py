import json
import os
import random
from torch.utils.data import Dataset

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.max_his_len = args.max_his_len
        self.index_file = args.index_file
        self.image_index_file = args.image_index_file
        self.new_tokens = None
        self.image_indices = None

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        if self.image_indices is not None:
            for index in self.image_indices.values():
                for token in index:
                    self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens

class SeqRecDataset(BaseDataset):

    def __init__(self, args, task='seqrec', mode='train'):
        super().__init__(args)
        self.mode = mode
        self.task = task
        self._load_data()
        self._remap_items()
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + '.inter.json'), 'r') as f:
            self.inters = json.load(f)
        if self.task == 'seqrec':
            with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        elif self.task == 'seqimage':
            with open(os.path.join(self.data_path, self.dataset + self.args.image_index_file), 'r') as f:
                self.indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = [''.join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _process_train_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data['item'] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data['inters'] = history
                inter_data.append(one_data)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            one_data['item'] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            one_data['inters'] = history
            inter_data.append(one_data)
        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        input = ''.join(d['inters'])
        output = d['item']
        return dict(input_ids=input, labels=output, label=index)

class ItemImageDataset(BaseDataset):

    def __init__(self, args, task='item2image'):
        super().__init__(args)
        self.task = task.lower()
        self.soft_prompt = args.soft_prompts[self.task]
        self._load_data()
        self.data_pair = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _process_data(self):
        data_pair = []
        for item_id in self.indices:
            item_index = ''.join(self.indices[item_id])
            image_index = ''.join(self.image_indices[item_id])
            if self.task == 'item2image':
                data_pair.append([item_index, image_index])
            else:
                data_pair.append([image_index, item_index])
        return data_pair

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):
        d = self.data_pair[index]
        input = self.soft_prompt + d[0]
        output = d[1]
        return dict(input_ids=input, labels=output)

class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args):
        super().__init__(args)
        self.soft_prompt = args.soft_prompts['fusionseqrec']
        self._load_data()
        self.inter_data = self._process_train_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + '.inter.json'), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                one_data['inters'] = history
                one_data['item'] = item
                inter_data.append(one_data)
        return inter_data

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        if random.random() > 0.5:
            history = [''.join(self.indices[str(i)]) for i in d['inters']]
            input = self.soft_prompt + ''.join(history)
            output = ''.join(self.image_indices[str(d['item'])])
        else:
            history = [''.join(self.image_indices[str(i)]) for i in d['inters']]
            input = self.soft_prompt + ''.join(history)
            output = ''.join(self.indices[str(d['item'])])
        return dict(input_ids=input, labels=output)

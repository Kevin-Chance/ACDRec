import os
from pathlib import Path
from torch.utils.data import ConcatDataset
from data import SeqRecDataset, FusionSeqRecDataset, ItemImageDataset
from index.common import set_seed

def parse_global_args(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_model_name', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser

def parse_dataset_args(parser):
    parser.add_argument('--data_path', default=str(Path(__file__).resolve().parent.parent / 'experiments' / 'datasets'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--index_file', required=True)
    parser.add_argument('--image_index_file', required=True)
    parser.add_argument('--max_his_len', type=int, default=20)
    parser.add_argument('--prompt_num', type=int, default=4)
    return parser

def parse_train_args(parser):
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--per_device_batch_size', type=int, default=1024)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--logging_step', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--resume_from_checkpoint', default=None)
    return parser

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def load_datasets(args):
    train = [SeqRecDataset(args, task=task, mode='train') for task in ['seqrec', 'seqimage']]
    train.extend((ItemImageDataset(args, task=task) for task in ['item2image', 'image2item']))
    train.append(FusionSeqRecDataset(args))
    valid = ConcatDataset([SeqRecDataset(args, task='seqrec', mode='valid')])
    return (ConcatDataset(train), valid)

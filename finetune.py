import argparse
import os
import torch
import transformers
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.trainer_callback import EarlyStoppingCallback
from utils import parse_global_args, parse_dataset_args, parse_train_args, set_seed, ensure_dir, load_datasets
from collator import Collator
from sid_vocabulary import prepare as prepare_sid_vocabulary

def train(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    local_rank = int(os.environ.get('LOCAL_RANK') or 0)
    ddp = int(os.environ.get('WORLD_SIZE', 1)) != 1
    config = T5Config.from_pretrained(args.load_model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.load_model_name, model_max_length=512)
    tasks = ['seqrec', 'seqimage', 'item2image', 'image2item', 'fusionseqrec']
    args.soft_prompts = {}
    for i, task in enumerate(tasks):
        token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num))
        args.soft_prompts[task] = ''.join(tokenizer.convert_ids_to_tokens(token_ids))
    train_data, valid_data = load_datasets(args)
    sid_structure = prepare_sid_vocabulary(tokenizer, args)
    args.strict_sid_length = sid_structure is not None
    for dataset in train_data.datasets:
        tokenizer.add_tokens(dataset.get_new_tokens())
    model = T5ForConditionalGeneration.from_pretrained(args.load_model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    if sid_structure is not None:
        model.config.sid_structure = sid_structure
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(model=model, train_dataset=train_data, eval_dataset=valid_data, args=transformers.TrainingArguments(seed=args.seed, per_device_train_batch_size=args.per_device_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio, num_train_epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, lr_scheduler_type='cosine', fp16=args.fp16, logging_steps=args.logging_step, optim='adamw_torch', gradient_checkpointing=args.gradient_checkpointing, evaluation_strategy='epoch', save_strategy='epoch', output_dir=args.output_dir, save_total_limit=1, load_best_model_at_end=True, ddp_find_unused_parameters=False if ddp else None, report_to=None, eval_delay=1), tokenizer=tokenizer, data_collator=Collator(args, tokenizer), callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patient)])
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune the ACDRec recommender')
    parse_global_args(parser)
    parse_dataset_args(parser)
    parse_train_args(parser)
    train(parser.parse_args())

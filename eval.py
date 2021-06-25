import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging

from arguments import fetch_standard_evaluation_arguments
from data.dataset import fetch_standard_test_dataset
from models.language_refer import fetch_model
from utils.directory import fetch_pretrain_path_list
from utils.distilbert import fetch_tokenizer
from utils.logging import get_logger
from utils.random import seed_worker

logging.set_verbosity_error()
logger = get_logger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def compute_metrics(p) -> Dict[str, Any]:
    predictions, labels = p
    predictions = np.argmax(predictions.squeeze(), axis=-1)
    count = np.sum(predictions.squeeze(2) == labels)
    total = labels.size
    logger.info('accuracy: {:7.5f}'.format(count / total))
    return {'total': count / total}


def prepare_model(pretrain_path, device, args, tokenizer):
    # prepare the model
    args.pretrain_path = pretrain_path
    model = fetch_model(args=args, tokenizer=tokenizer)
    model.to(device)
    model.is_train = False
    model.eval()
    return model


def eval_custom() -> None:
    args = fetch_standard_evaluation_arguments(verbose=False)
    tokenizer = fetch_tokenizer()
    dataset = fetch_standard_test_dataset(args=args, tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        worker_init_fn=seed_worker)

    device = 'cpu' if args.no_cuda else 'cuda:0'
    model = prepare_model(args.pretrain_path, device, args, tokenizer)

    out_matched = []
    out_assignment_id = []
    count = 0
    for batch in tqdm(data_loader):
        with torch.no_grad():
            refined_batch = model.prepare_batch(batch, device)
            matched = model.eval_forward(batch=refined_batch)
            out_matched.append(matched)
            out_assignment_id.append(batch['assignment_ids'])
            count += 1

    matched = torch.cat(out_matched).detach().cpu()
    assignment_id = torch.cat(out_assignment_id).detach().cpu()
    matched_dict = {a.item(): m.item() for m, a in zip(matched, assignment_id)}
    print('Accuracy {:7.5f}'.format(sum(1 for v in matched_dict.values() if v) / len(matched_dict.values()) * 100))
    with open(str(Path(args.output_dir) / 'eval.json'), 'w') as file:
        json.dump(matched_dict, file, indent=4)
    logger.info('wrote an evaluation file: {}'.format(Path(args.output_dir) / 'eval.json'))


if __name__ == '__main__':
    eval_custom()

import json
import os
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from data.instance_storage import fetch_index_by_instance_class
from utils.directory import mkdir_if_not_exists
from utils.distilbert import fetch_tokenizer
from utils.eval import AverageMeter

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from arguments import fetch_standard_training_arguments
from data.dataset import fetch_dataset
from models.language_refer import fetch_model
from utils.logging import get_logger

logger = get_logger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def run_evaluation(args, model, test_dl, device, global_num_iter):
    out_matched = []
    out_assignment_id = []
    count = 0
    with torch.no_grad():
        for local_eval_num_iter, batch in enumerate(test_dl):
            refined_batch = model.prepare_batch(batch, device)
            matched = model.eval_forward(batch=refined_batch)
            out_matched.append(matched)
            out_assignment_id.append(batch['assignment_ids'])
            count += 1

    matched = torch.cat(out_matched).detach().cpu()
    assignment_id = torch.cat(out_assignment_id).detach().cpu()
    matched_dict = {a.item(): m.item() for m, a in zip(matched, assignment_id)}
    print('Iteration {}, Accuracy {:7.5f}'.format(
        global_num_iter, sum(1 for v in matched_dict.values() if v) / len(matched_dict.values()) * 100))
    with open(str(Path(args.output_dir) / 'eval{:06d}.json'.format(global_num_iter)), 'w') as file:
        json.dump(matched_dict, file, indent=4)
    logger.info('wrote an evaluation file: {}'.format(Path(args.output_dir) / 'eval{:06d}.json'.format(global_num_iter)))


def modify_args_train(args):
    args.use_target_mask = False
    args.target_mask_k = 1


def modify_args_test(args):
    args.use_target_mask = True
    args.target_mask_k = 4


def train():
    args = fetch_standard_training_arguments()
    device = 'cuda:0'

    tokenizer = fetch_tokenizer()
    modify_args_train(args)
    train_dataset = fetch_dataset(
        args=args,
        split_name='train',
        tokenizer=tokenizer)

    modify_args_test(args)
    test_dataset = fetch_dataset(
        args=args,
        split_name='test',
        tokenizer=tokenizer)

    modify_args_train(args)
    model = fetch_model(args=args, tokenizer=tokenizer)
    model.train()
    model = model.to(device)

    index_by_instance_class = fetch_index_by_instance_class(
        label_type=args.label_type,
        dataset_name=args.dataset_name)

    default_task_name = 'viewpoint' if args.use_bbox_annotation_only else 'ref'
    task_names = {default_task_name}

    criterion_dict = {
        default_task_name: CrossEntropyLoss().to(device),
    }
    weight_dict = {
        default_task_name: args.weight_ref,
    }

    if args.use_clf_loss:
        ignore_class = 'otherprop' if args.label_type == 'revised' else 'pad'
        task_names.add('cls')
        criterion_dict['cls'] = CrossEntropyLoss(ignore_index=index_by_instance_class[ignore_class]).to(device)
        weight_dict['cls'] = args.weight_clf

    if args.use_tar_loss:
        task_names.add('tar')
        criterion_dict['tar'] = CrossEntropyLoss().to(device)
        weight_dict['tar'] = args.weight_tar

    if args.use_pos_loss:
        task_names.add('pos')
        criterion_dict['pos'] = MSELoss().to(device)
        weight_dict['pos'] = args.weight_pos

    if args.use_mask_loss:
        task_names.add('mask')
        criterion_dict['mask'] = CrossEntropyLoss(ignore_index=-100).to(device)
        weight_dict['mask'] = args.weight_mask

    meter_dict: Dict[str, AverageMeter] = {key: AverageMeter(key) for key in task_names}

    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        args.num_train_epochs * len(train_dataset) // args.per_device_train_batch_size)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True)
    test_dl = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True)

    global_num_iter = 0
    for num_batch in range(args.num_train_epochs):
        tqdm_iter = tqdm(enumerate(train_dl))
        for local_num_iter, batch in tqdm_iter:
            global_num_iter += 1

            refined_batch = model.prepare_batch(batch, device)
            optimizer.zero_grad()
            with autocast():
                logits_dict, gt_dict = model(**refined_batch)
                loss_dict = {
                    name: criterion_dict[name](logits_dict[name], gt_dict[name].to(dtype=torch.long, device=device))
                    for name in task_names}
                for key, value in loss_dict.items():
                    meter_dict[key].update(value, args.logging_steps)
                loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])

            if global_num_iter % args.logging_steps == 0:
                for key in meter_dict.keys():
                    meter_dict[key].reset()

            tqdm_iter.set_description('loss: {:7.5f}'.format(loss))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_num_iter % args.save_steps == 0:
                checkpoint_dir = mkdir_if_not_exists(
                    Path(args.output_dir) / 'checkpoint-{:06d}'.format(global_num_iter))
                model_path = checkpoint_dir / 'model.pt'
                optimizer_path = checkpoint_dir / 'optimizer.pt'
                state_path = checkpoint_dir / 'state.pt'
                torch.save(model.state_dict(), str(model_path))
                torch.save(optimizer.state_dict(), str(optimizer_path))
                torch.save({
                    'num_batch': num_batch,
                    'local_num_iter': local_num_iter,
                    'global_num_iter': global_num_iter,
                }, str(state_path))

                modify_args_test(args)
                run_evaluation(args, model, test_dl, device, global_num_iter)
                modify_args_train(args)


if __name__ == '__main__':
    train()

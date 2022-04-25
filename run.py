import os
from argparse import Namespace
from typing import Dict, Any

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from arguments import fetch_arguments
from data.dataset import ReferIt3DDataset
from data.instance_storage import fetch_index_by_instance_class
from models.language_refer import fetch_model
from utils.distilbert import fetch_tokenizer
from utils.eval import AverageMeter

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class LanguageRefer(pl.LightningModule):
    def __init__(self, args: Namespace):
        pl.LightningModule.__init__(self)
        self.args = args
        self.tokenizer = fetch_tokenizer()
        self.model = fetch_model(args, self.tokenizer)

        self.index_by_instance_class = fetch_index_by_instance_class(
            label_type=args.label_type,
            dataset_name=args.dataset_name)

        default_task_name = 'viewpoint' if args.use_bbox_annotation_only else 'ref'
        self.task_names = {default_task_name, 'cls'}

        ignore_class = 'otherprop' if args.label_type == 'revised' else 'pad'
        self.criterion_dict = {
            default_task_name: CrossEntropyLoss(),
            'cls': CrossEntropyLoss(ignore_index=self.index_by_instance_class[ignore_class]),
        }
        self.weight_dict = {
            default_task_name: args.weight_ref,
            'cls': args.weight_clf,
        }
        self.meter_dict: Dict[str, AverageMeter] = {key: AverageMeter(key) for key in self.task_names}

    def forward(self,
                input_ids,
                attention_mask,
                ref_mask,
                cls_mask,
                bboxs,
                target_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          ref_mask=ref_mask,
                          cls_mask=cls_mask,
                          bboxs=bboxs)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_training_steps,
            last_epoch=self.trainer.current_epoch - 1)
        return [optimizer], [scheduler]

    def _single_step(self, batch, mode: str):
        item_dict, gt_dict, assignment_ids = batch
        logits_dict = self.model(**item_dict)
        gt_dict = {k: v.view(-1) for k, v in gt_dict.items()}
        loss_dict = {name: self.criterion_dict[name](logits_dict[name], gt_dict[name]) for name in self.task_names}
        loss = sum([self.weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        for k, v in loss_dict.items():
            self.log('{}_{}_loss'.format(mode, k), v.item())
        self.log('{}_loss'.format(mode), loss.item())
        return loss, logits_dict, assignment_ids, gt_dict

    def _single_step_with_matched(self, batch, mode: str):
        _, logits_dict, assignment_ids, gt_dict = self._single_step(batch=batch, mode=mode)
        indices = torch.argmax(logits_dict['ref'], dim=1)
        matched = indices == gt_dict['ref']
        return matched.detach().cpu(), assignment_ids.detach().cpu()

    def training_step(self, train_batch, batch_idx):
        loss, _, _, _ = self._single_step(batch=train_batch, mode='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        return self._single_step_with_matched(batch=val_batch, mode='val')

    def test_step(self, test_batch, batch_idx):
        return self._single_step_with_matched(batch=test_batch, mode='test')

    def _single_epoch(self, outputs, mode: str):
        matched, assignment_id = zip(*outputs)
        matched = torch.cat(list(matched))
        assignment_id = torch.cat(list(assignment_id))
        matched_dict = {str(a.item()): float(m.item()) for m, a in zip(matched, assignment_id)}
        accuracy = sum(1 for v in matched_dict.values() if v) / len(matched_dict.values()) * 100
        self.log('{}_accuracy'.format(mode), accuracy, on_step=False, on_epoch=True)
        df = pd.DataFrame(sorted(matched_dict.items()), columns=['index', 'correct'])
        self.logger.log_text(key='{}_matched'.format(mode), dataframe=df)

    def validation_epoch_end(self, outputs):
        self._single_epoch(outputs=outputs, mode='val')

    def test_epoch_end(self, outputs):
        self._single_epoch(outputs=outputs, mode='test')


def fetch_data_loaders(args: Namespace):
    train_dataset = ReferIt3DDataset(
        args=args,
        split='train',
        use_target_mask=False,
        target_mask_k=1)
    test_dataset = ReferIt3DDataset(
        args=args,
        split='test',
        use_target_mask=True,
        target_mask_k=args.target_mask_k)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=args.pin_memory)
    test_dl = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=args.pin_memory)
    return train_dl, test_dl


def main():
    args = fetch_arguments()
    wandb_logger = WandbLogger(project='lr')
    train_dl, test_dl = fetch_data_loaders(args)

    args.total_training_steps = int(args.total_training_epochs * (len(train_dl) / args.gpus))
    print('Set the total training steps: {}'.format(args.total_training_steps))

    if args.resume:
        model = LanguageRefer.load_from_checkpoint(args.resume, args=args)
    else:
        model = LanguageRefer(args=args)

    filename_fmt = '{}-'.format(args.experiment_tag) + '{epoch:02d}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints/',
        filename=filename_fmt,
        save_top_k=3,
        mode='max')
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        strategy='ddp',
        precision=16,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor_callback, ])

    if args.mode == 'train':
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    else:
        trainer.test(model, dataloaders=test_dl)


if __name__ == '__main__':
    main()

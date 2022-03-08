import os
from argparse import Namespace
from typing import Dict

import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from arguments import fetch_standard_training_arguments
from data.dataset import fetch_dataset
from data.instance_storage import fetch_index_by_instance_class
from models.language_refer import fetch_model
from utils.distilbert import fetch_tokenizer
from utils.eval import AverageMeter

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class LanguageReferWrapper(pl.LightningModule):
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
                utterance_attention_mask,
                object_attention_mask,
                bboxs,
                target_mask):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          utterance_attention_mask=utterance_attention_mask,
                          object_attention_mask=object_attention_mask,
                          bboxs=bboxs,
                          target_mask=target_mask)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.args.warmup_steps,
            200000)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        item_dict, gt_dict = train_batch
        logits_dict = self.model(**item_dict)
        gt_dict = {
            'ref': gt_dict['wheres'].view(-1),  # (bsize * seq_len, )
            'cls': gt_dict['binary_labels'].view(-1),  # (bsize * seq_len, )
            'tar': gt_dict['target_labels']
        }

        loss_dict = {name: self.criterion_dict[name](logits_dict[name], gt_dict[name]) for name in self.task_names}
        loss = sum([self.weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        for k, v in loss_dict.items():
            self.log('train_{}_loss'.format(k), v.item())
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        # BATCH_KEYS = {'input_ids', 'attention_mask', 'utterance_attention_mask',
        #               'object_attention_mask', 'bboxs', 'wheres', 'binary_labels',
        #               'target_mask', 'target_labels', 'gt_input_ids'}
        #
        # print(BATCH_KEYS)
        # for k in val_batch.keys():
        #     print(k, k in BATCH_KEYS)
        item_dict, gt_dict = val_batch
        logits_dict = self.model(**item_dict)
        gt_dict = {
            'ref': gt_dict['wheres'].view(-1),  # (bsize * seq_len, )
            'cls': gt_dict['binary_labels'].view(-1),  # (bsize * seq_len, )
            'tar': gt_dict['target_labels']
        }

        loss_dict = {name: self.criterion_dict[name](logits_dict[name], gt_dict[name]) for name in self.task_names}
        loss = sum([self.weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        for k, v in loss_dict.items():
            self.log('val_{}_loss'.format(k), v.item())
        self.log('val_loss', loss.item())
        return loss


def train():
    args = fetch_standard_training_arguments()
    tokenizer = fetch_tokenizer()

    train_dataset = fetch_dataset(
        args=args,
        split_name='train',
        tokenizer=tokenizer)
    test_dataset = fetch_dataset(
        args=args,
        split_name='test',
        tokenizer=tokenizer)

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

    model = LanguageReferWrapper(args)
    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model, train_dl)


if __name__ == '__main__':
    train()

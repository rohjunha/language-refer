from argparse import Namespace
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from transformers import DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizerFast
from transformers.models.distilbert import DistilBertConfig

from utils.directory import fetch_target_class_list_path
from utils.distilbert import fetch_pretrained_bert_model
from utils.positional_encoding import pe_from_tensor

BATCH_KEYS = {'input_ids', 'attention_mask', 'utterance_attention_mask',
              'object_attention_mask', 'bboxs', 'wheres', 'binary_labels',
              'target_mask', 'target_labels', 'gt_input_ids'}


class LanguageRefer(DistilBertPreTrainedModel):
    def __init__(
            self,
            config: DistilBertConfig,
            num_target_classes: int,
            use_target_mask: bool,
            use_clf_loss: bool,
            use_tar_loss: bool,
            use_mask_loss: bool,
            use_pos_loss: bool):
        DistilBertPreTrainedModel.__init__(
            self,
            config)

        self.config = config
        self.use_target_mask = use_target_mask
        self.use_clf_loss = use_clf_loss
        self.use_tar_loss = use_tar_loss
        self.use_mask_loss = use_mask_loss
        self.use_pos_loss = use_pos_loss
        self.num_target_classes = num_target_classes
        self.vocab_size = 28996

        self.bert_reference = DistilBertModel(config)
        self.dropout = Dropout(0.1)
        self.ref_classifier = Linear(self.config.hidden_size, 1)
        if self.use_clf_loss:
            self.cls_classifier = Linear(self.config.hidden_size, 2)
        if self.use_tar_loss:
            self.tar_classifier = Linear(self.config.hidden_size, num_target_classes)
        if self.use_mask_loss:
            self.mask_classifier = Linear(self.config.hidden_size, self.vocab_size)
        if self.use_pos_loss:
            self.pos_regressor = Linear(self.config.hidden_size, 3)

        self.is_train = None

    def forward(
            self,
            input_ids,
            attention_mask,
            utterance_attention_mask,
            object_attention_mask,
            bboxs,
            wheres,
            binary_labels,
            target_mask,
            target_labels,
            gt_input_ids=None,
    ):
        assert attention_mask is not None
        assert utterance_attention_mask is not None
        assert object_attention_mask is not None

        if self.use_mask_loss:
            assert gt_input_ids is not None
        else:
            assert gt_input_ids is None

        input_embeddings = self.bert_reference.get_input_embeddings()  # .to(device=self.device)
        inputs_embeds = input_embeddings(input_ids)
        bboxs = pe_from_tensor(bboxs, 128)
        inputs_embeds += bboxs

        outputs = self.bert_reference(
            None,
            attention_mask=attention_mask,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_outputs = outputs[0]  # last hidden state (bsize, seq_len, hidden_size)
        sequence_outputs = self.dropout(sequence_outputs)
        ref_logits = self.ref_classifier(sequence_outputs).squeeze(dim=2)  # (bsize, seq_len)

        ref_mask = object_attention_mask == 1
        if self.use_target_mask:
            ref_mask = torch.logical_and(ref_mask, target_mask)
        ref_logits[~ref_mask] = -1e4

        logits_dict = {
            'ref': ref_logits,  # (bsize, seq_len)
        }
        gt_dict = {
            'ref': wheres.view(-1),  # (bsize * seq_len, )
        }

        if self.use_clf_loss:
            cls_logits = self.cls_classifier(sequence_outputs)  # (bsize, seq_len, 2)
            cls_mask = object_attention_mask == 1
            cls_logits[~cls_mask, :] = -1e4
            logits_dict['cls'] = cls_logits.view(-1, 2)  # (bsize * seq_len, 2)
            gt_dict['cls'] = binary_labels.view(-1)  # (bsize * seq_len, )

        if self.use_tar_loss:
            tar_logits = self.tar_classifier(sequence_outputs[:, 0, :].squeeze(dim=1))  # (bsize, num_target_classes)
            logits_dict['tar'] = tar_logits
            gt_dict['tar'] = target_labels

        if self.use_pos_loss:
            pos_logits = self.pos_regressor(sequence_outputs[:, 0, :].squeeze())  # (bsize, 3)
            tar_positions = torch.stack([torch.index_select(bboxs[i, ...], 0, wheres[i]).squeeze()
                                         for i in range(wheres.shape[0])])[:, :3]  # (bsize, 3)
            logits_dict['pos'] = pos_logits
            gt_dict['pos'] = tar_positions

        if gt_input_ids is not None:
            mask_mask = utterance_attention_mask == 1
            mask_logits = self.mask_classifier(sequence_outputs)  # (bsize, seq_len, vocab_size)
            mask_logits[~mask_mask, :] = -1e4
            logits_dict['mask'] = mask_logits.view(-1, self.vocab_size)  # (bsize * seq_len, vocab_size)
            gt_dict['mask'] = gt_input_ids.view(-1)  # (bsize * seq_len)

        return logits_dict, gt_dict

    def prepare_batch(self, raw_batch, device):
        batch = dict()
        for key, value in raw_batch.items():
            if key in BATCH_KEYS:
                if isinstance(value, Tensor):
                    batch[key] = value.to(device=device)
                else:
                    batch[key] = value
        return batch

    def eval_forward(self, batch) -> Tensor:
        logits_dict, gt_dict = self.forward(**batch)
        indices = torch.argmax(logits_dict['ref'], dim=1)
        matched = indices == gt_dict['ref']
        return matched


def fetch_index_by_target_class_dict(label_type: str) -> Dict[str, int]:
    target_class_list_path = fetch_target_class_list_path(label_type=label_type)
    with open(str(target_class_list_path), 'r') as file:
        words = file.read().splitlines()
    return {w: i for i, w in enumerate(words)}


def fetch_model(
        args: Namespace,
        tokenizer: DistilBertTokenizerFast) -> Module:
    num_target_class = len(fetch_index_by_target_class_dict(args.label_type))
    model_ = fetch_pretrained_bert_model()
    config = model_.config

    model = LanguageRefer(
        config=config,
        use_target_mask=args.use_target_mask,
        use_clf_loss=args.use_clf_loss,
        use_tar_loss=args.use_tar_loss,
        use_mask_loss=args.use_mask_loss,
        use_pos_loss=args.use_pos_loss,
        num_target_classes=num_target_class)

    if args.pretrain_path is None:
        model.bert_reference = model_
    else:
        checkpoint_dir = Path(args.pretrain_path)
        model_path = checkpoint_dir / 'model.pt'

        state_dict = torch.load(str(model_path))
        new_state_dict = dict()
        for k, v in state_dict.items():
            if not args.use_mask_loss and k.startswith('mask_classifier'):
                continue
            if not args.use_tar_loss and k.startswith('tar_classifier'):
                continue
            if not args.use_pos_loss and k.startswith('pos_regressor'):
                continue
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.tokenizer = tokenizer

    return model

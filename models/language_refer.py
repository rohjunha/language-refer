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
            use_target_mask: bool):
        DistilBertPreTrainedModel.__init__(
            self,
            config)

        self.config = config
        self.use_target_mask = use_target_mask
        self.num_target_classes = num_target_classes
        self.vocab_size = 28996

        self.bert_reference = DistilBertModel(config)
        self.dropout = Dropout(0.1)
        self.ref_classifier = Linear(self.config.hidden_size, 1)
        self.cls_classifier = Linear(self.config.hidden_size, 2)

        self.is_train = None

    def forward(
            self,
            input_ids,
            attention_mask,
            utterance_attention_mask,
            object_attention_mask,
            bboxs,
            target_mask
    ):
        assert attention_mask is not None
        assert utterance_attention_mask is not None
        assert object_attention_mask is not None

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

        # update ref_logits with mask
        ref_mask = object_attention_mask == 1
        if self.use_target_mask:
            ref_mask = torch.logical_and(ref_mask, target_mask)
        ref_logits[~ref_mask] = -1e4

        cls_logits = self.cls_classifier(sequence_outputs)  # (bsize, seq_len, 2)
        cls_mask = object_attention_mask == 1
        cls_logits[~cls_mask, :] = -1e4

        return {
            'ref': ref_logits,  # (bsize, seq_len)
            'cls': cls_logits.view(-1, 2),  # (bsize * seq_len, 2)
        }

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
        num_target_classes=num_target_class)

    if args.pretrain_path is None:
        model.bert_reference = model_
    else:
        checkpoint_dir = Path(args.pretrain_path)
        model_path = checkpoint_dir / 'model.pt'

        state_dict = torch.load(str(model_path))
        new_state_dict = dict()
        cls_prefixs = ['mask_classifier', 'tar_classifier', 'pos_regressor']
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in cls_prefixs):
                continue
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model.tokenizer = tokenizer

    return model

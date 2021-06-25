from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import torch
from pandas import DataFrame, Series
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

from data.instance_storage import InstanceStorage, fetch_instance_storage
from data.unified_data_frame import fetch_unified_data_frame, fetch_custom_data_frame
from utils.directory import fetch_instance_class_list_by_scene_id_path, \
    fetch_predicted_target_class_by_assignment_id_path, fetch_data_root_dir, fetch_target_class_list_path, \
    fetch_predicted_instance_class_list_by_hash_path, fetch_nr3d_eval_assignment_id_list_path
from utils.hash import generate_hash


def fetch_index_by_target_class_dict(label_type: str) -> Dict[str, int]:
    with open(str(fetch_target_class_list_path(label_type)), 'r') as file:
        words = file.read().splitlines()
    return {w: i for i, w in enumerate(words)}


def get_sep_index(word_ids: List[int]) -> int:
    """
    Extracts the index of the first token right after the first [SEP] token.
    This indicates the starting position of the second sentence.
    :param word_ids: word indices
    :return: integer index
    """
    return np.where(list(map(lambda x: x is None, word_ids)))[0][1] + 1


def create_sep_word_ids(
        word_ids: List[int],
        tokens: List[str]) -> Tuple[List[int], int]:
    sep_index = get_sep_index(word_ids)
    sub_word_ids = np.array(word_ids[sep_index:])
    sub_tokens = np.array(tokens[sep_index:])
    sub_word_ids[sub_tokens == '[SEP]'] = -1
    sub_word_ids[sub_tokens == '[PAD]'] = -1
    # if use_point_embedding_separate:
    #     sub_word_ids[sub_tokens == '[POINT]'] = -2
    sub_word_ids[
        np.logical_and(sub_word_ids != -1, sub_word_ids != -2)] = 0  # set 0 to tokens from words, -1 (or -2) to escapes
    num_item_per_groups = [len(list(g)) for k, g in groupby(sub_word_ids)]  # count the number of 0s or -1s (or -2s)
    num_types = 2  # 3 if use_point_embedding_separate else 2
    num_elements = num_item_per_groups[::num_types]  # count the numbers of tokens other than [SEP]s or [PAD]s
    num_pads = num_item_per_groups[-1] - 1  # count the numbers of pads
    new_word_ids = []
    for i, n in enumerate(num_elements):
        new_word_ids.extend([i] * n)
        # if use_point_embedding_separate:
        #     new_word_ids.append(None)
        new_word_ids.append(None)
    new_word_ids = word_ids[:sep_index] + new_word_ids + [None] * num_pads
    assert len(new_word_ids) == len(tokens)

    return new_word_ids, sep_index


def replace_tokens_with_mask_and_noun_indices(
        tokenizer: DistilBertTokenizerFast,
        noun_word_indices: List[int],
        tokens: List[str],
        word_ids: List[int],
        input_ids: List[int],
        rng):
    num_tokens = len(input_ids)

    noun_token_indices = [i for i, j in enumerate(word_ids) if j in noun_word_indices]

    gt_input_ids = [-100] * num_tokens
    modified_input_ids = deepcopy(input_ids)
    for i, j in enumerate(noun_token_indices):
        k = rng.uniform()
        if k < 0.15:
            gt_input_ids[j] = input_ids[j]
            l = rng.uniform()
            if l < 0.8:
                modified_input_ids[j] = 103
            elif l < 0.9:
                modified_input_ids[j] = rng.integers(106, 28996, 1)[0]

    return {
        'gt_input_ids': gt_input_ids,  # List[int], (num_tokens, ), -100 if not replaced
        'original_bert_tokens': tokens,  # List[str], (num_tokens, )
        'modified_input_ids': modified_input_ids,  # List[int], (num_tokens, ), replaced input ids
        'modified_bert_tokens': tokenizer.convert_ids_to_tokens(modified_input_ids),
        # List[str], (num_tokens, ), list of (modified) tokens
    }


def encode_single_label_with_sep_without_point(
        word_ids: List[Union[int, None]],
        tokens: List[str],
        labels: np.ndarray,
        hashs: np.ndarray,
        target_mask: str) -> Dict[str, Any]:
    """
    Generates tags for token for each referral data point (with [SEP] added)
    :param word_ids: List[Union[int, None]], (max_tokens, ), integer indices to map tokens to words
    :param tokens: List[str], (max_tokens, ), tokens after tokenizer
    :param labels: np.int64, (max_context, ), integer labels
    :param bboxs: np.float32, (max_context, 6 or 768), bounding box values
    :param hashs: np.int64, (max_context, ), hash codes
    :param target_mask: str, ' '.joined indices of (predicted) target candidates
    :return:
        'labels': np.int64, (max_encoded, ), encoded labels
        'binary_labels': np.int64, (max_encoded, ), binary labels of indicating the target class
        'wheres': int, encoded position of the target instance
        'hashs': int64, (max_encoded, ), encoded hash codes
        'utterance_attention_mask': np.int64, (max_encoded, ), attention mask on utterance
        'object_attention_mask': np.int64, (max_encoded, ), attention mask only on object instances
        'target_mask': np.bool, (max_encoded, ), True if it is belong to the target class
    """
    tmp_word_ids, sep_index = create_sep_word_ids(word_ids, tokens)
    num_context = len(tmp_word_ids)
    new_word_ids = tmp_word_ids[sep_index:]
    new_tokens = tokens[sep_index:]

    new_hashs = np.ones((num_context,), dtype=np.int64) * -1
    utterance_attention_mask = np.zeros((num_context,), dtype=np.int64)
    object_attention_mask = np.zeros((num_context,), dtype=np.int64)
    utterance_attention_mask[1:sep_index] = 1

    final_indices: List[int] = list(map(int, target_mask.split()))
    encoded_final_mask = (np.zeros if final_indices else np.ones)((num_context,), dtype=np.bool)

    encoded_labels = [-100] * sep_index

    previous_word_idx = None
    count_pad = 0
    eff_idx = sep_index
    label_where = -1
    for idx, (word_idx, token) in enumerate(zip(new_word_ids, new_tokens)):
        # Special tokens have a word id that is None.
        # We set the label to -100 so they are automatically ignored in the loss function.
        label_value = -100
        if word_idx is not None and word_idx != previous_word_idx:
            label_value = labels[word_idx]
            new_hashs[eff_idx] = hashs[word_idx]
            object_attention_mask[eff_idx] = 1
            if word_idx in final_indices:
                encoded_final_mask[eff_idx] = True
        if label_value == 1:
            label_where = len(encoded_labels)
        previous_word_idx = word_idx
        # if remove_sep and token == '[SEP]':
        #     count_pad += 1
        #     continue
        encoded_labels.append(label_value)
        eff_idx += 1
    if count_pad > 0:
        encoded_labels.extend([-100] * count_pad)

    encoded_labels = np.array(encoded_labels)
    binary_labels = np.zeros_like(encoded_labels)
    binary_labels[encoded_labels == 1] = 1
    binary_labels[encoded_labels == 2] = 1

    return {
        'labels': encoded_labels,
        'binary_labels': binary_labels,
        'wheres': label_where,
        'hashs': new_hashs,
        'object_attention_mask': object_attention_mask,
        'utterance_attention_mask': utterance_attention_mask,
        'target_mask': encoded_final_mask,
    }


def encode_data_with_tokenizer(
        tokenizer: DistilBertTokenizerFast,
        rng,
        use_mask_loss: bool,
        dataset_names: List[str],
        assignment_ids: List[int],
        utterances: List[str],
        instance_types: List[List[str]],
        view_dependent: List[bool],
        view_dependent_explicit: List[bool],
        labels: List[np.ndarray],
        hashs: List[np.ndarray],
        target_mask: List[str],
        target_labels: List[int],
        noun_word_indices: List[str]):
    """
    Create an encoded data with tokens by a DistilBertTokenizer
    :param tokenizer: DistilBertTokenizerFast
    :param tagger: SequenceTagger
    :param dataset_names: List[str], (num_entries, ), list of dataset names in {nr3d, sr3d}
    :param assignment_ids: List[int], (num_entries, ), list of unique ids (-1 in sr3d)
    :param utterances: List[str], (num_entries, ), list of utterances
    :param instance_types: List[List[str]], (num_entries, num_instances), class types of instances (e.g. table, ...)
    :param view_dependent: List[bool], (num_entries, ), list of view dependent indicators
    :param view_dependent_explicit: List[bool], (num_entries, ), list of view dependent-explicit indicators
    :param labels: List[np.ndarray], (num_entries, max_context), class labels of instances (e.g. 1: target, ...)
    :param hashs: List[np.ndarray], (num_entries, max_context), hash values
    :param target_mask: List[str], (num_entries, )
    :param target_labels: List[int], (num_entries, ), target indices (out of 76)
    :param noun_word_indices: List[str], (num_entries, ), ' '.joined noun word indices
    :return:
        dataset_names: List[str], (num_entries, ), list of dataset names in {nr3d, sr3d}; this is bypassed
        assignment_ids: List[int], (num_entries, ), list of unique ids (-1 in sr3d); this is bypassed
        view_dependent: List[bool], (num_entries, ), bypassed
        view_dependent_explicit: List[bool], (num_entries, ), bypassed
        labels: np.int64, (num_entries, num_encoded), integer labels
        wheres: np.int64, (num_entries, ), integer positions in encoded values
        hashs: np.int64, (num_entries, num_encoded), hash codes
        target_mask: np.bool, (num_entries, num_encoded), boolean target indicator
        target_labels: List[int], (num_entries, ), bypassed
        input_ids
        gt_input_ids
    """
    join_char = '[SEP]'
    joined_instance_types: List[str] = [join_char.join(l) for l in instance_types]
    encodings = tokenizer(
        utterances,
        joined_instance_types,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True)

    encoded_value_list_dict = defaultdict(list)
    for i in range(len(labels)):
        tokens = encodings[i].tokens
        word_ids = encodings.word_ids(batch_index=i)
        input_ids = encodings['input_ids'][i]

        if use_mask_loss:
            noun_word_str_list = noun_word_indices[i].split(' ')
            if noun_word_str_list and noun_word_str_list[0] != '':
                noun_indices = list(map(int, noun_word_str_list))
            else:
                noun_indices = []
            replaced_token_info = replace_tokens_with_mask_and_noun_indices(
                tokenizer, noun_indices, tokens, word_ids, input_ids, rng)
            gt_input_ids = replaced_token_info['gt_input_ids']
            input_ids = replaced_token_info['modified_input_ids']
        else:
            gt_input_ids = None
        encoded_value_list_dict['input_ids'].append(input_ids)
        encoded_value_list_dict['gt_input_ids'].append(gt_input_ids)

        # print(word_ids)
        # print(tokens)
        # print(labels[i])
        # print(hashs[i])

        encoded_item = encode_single_label_with_sep_without_point(
            word_ids=word_ids,
            tokens=tokens,
            labels=labels[i],
            hashs=hashs[i],
            target_mask=target_mask[i])
        for k, v in encoded_item.items():
            encoded_value_list_dict[k].append(v)

        # print(encoded_item['labels'])
        # print(encoded_item['hashs'])
        # print(encoded_item['target_mask'])
        # print(encoded_item['wheres'])
        # input()
    encoded_value_dict = dict()
    for k, v in encoded_value_list_dict.items():
        encoded_value_dict[k] = None if v[0] is None else np.stack(v)

    encoded_value_dict['dataset_names'] = dataset_names
    encoded_value_dict['assignment_ids'] = assignment_ids
    encoded_value_dict['view_dependent'] = view_dependent
    encoded_value_dict['view_dependent_explicit'] = view_dependent_explicit
    encoded_value_dict['target_labels'] = target_labels
    encoded_value_dict['utterances'] = utterances
    encoded_value_dict['attention_mask'] = np.stack(encodings['attention_mask'])
    return encoded_value_dict


def fetch_instance_class_list_by_scene_id(label_type: str) -> Dict[str, List[str]]:
    return torch.load(str(fetch_instance_class_list_by_scene_id_path(label_type)))

    # dict_path =
    # if dict_path.exists():
    #     return torch.load(str(dict_path))
    # else:
    #     dataset_names = {'nr3d', 'sr3d'}
    #     instance_class_list_by_scene_id = dict()
    #     for dataset_name in dataset_names:
    #         scannet_data: ScannetData = read_raw_scannet_data(dataset_name)
    #         for scene_id, scan in scannet_data.scan_dict.items():
    #             instance_class_list_by_scene_id[scene_id] = [o.instance_label for o in scan.three_d_objects]
    #     torch.save(instance_class_list_by_scene_id, str(dict_path))
    #     return instance_class_list_by_scene_id


def fetch_predicted_target_class_by_assignment_id(label_type: str) -> Dict[int, str]:
    return torch.load(str(fetch_predicted_target_class_by_assignment_id_path(label_type=label_type)))


def fetch_predicted_instance_class_list_by_hash(label_type: str) -> Dict[int, List[str]]:
    return torch.load(str(fetch_predicted_instance_class_list_by_hash_path(label_type=label_type)))


def fetch_noun_word_indices_by_assignment_id() -> Dict[int, List[int]]:
    # return torch.load(str(fetch_data_root_dir() / 'backup/noun_word_indices_by_assignment_id_nr3d.pt'))
    return torch.load(str(fetch_data_root_dir() / 'noun_word_indices_by_assignment_id.pt'))


class InstanceSampler:
    def __init__(
            self,
            dataset_name: str,
            label_type: str,
            split_name: str,
            storage: InstanceStorage,
            max_distractors: int,
            use_predicted_class: bool,
            target_mask_k: int,
            num_points: int = 1000):
        self.dataset_name = dataset_name
        self.label_type = label_type
        self.split_name = split_name

        self.storage = storage
        self.max_distractors = max_distractors
        self.num_points = num_points
        self.target_mask_k = target_mask_k
        self.use_predicted_class = use_predicted_class

        self.instance_class_list_by_scene_id = fetch_instance_class_list_by_scene_id(label_type=label_type)
        self.predicted_instance_class_list_by_hash = fetch_predicted_instance_class_list_by_hash(label_type=label_type)

        self.index_by_target_class = fetch_index_by_target_class_dict(label_type=label_type)
        self.predicted_target_class_by_assignment_id = fetch_predicted_target_class_by_assignment_id(
            label_type=label_type)
        self.noun_word_indices_by_assignment_id = fetch_noun_word_indices_by_assignment_id()

    @property
    def max_context(self) -> int:
        return self.max_distractors + 1

    def fetch_instances(
            self,
            df: DataFrame) -> Dict[str, List[Any]]:
        item_dict = defaultdict(list)
        for i, series in df.iterrows():
            for k, v in self.fetch_sampled_instances(series).items():
                item_dict[k].append(v)
        return item_dict

    def fetch_sampled_instances(
            self,
            series: Series) -> Dict[str, Any]:
        """
        Draw samples from the scene and generate instance information
        :param series:
        :return: dictionary of instance information
            dataset_names: str, name of the dataset
            instance_types: List[str], (num_instances, ), class types of instances (e.g. table, chair, ...)
            target_class: str, target class
            target_mask: str, indices of target instances (target and same-class distractors)
            target_labels: int, index of target class (out of 76)
            labels: np.int64 (max_context, ), class labels of instances (e.g. 1: target, 2: distractors, ...)
            assignment_ids: int, unique id for nr3d utterances
            utterances: str, utterance GENERATED from tokens
            hashs: np.int64 (max_context, ), hash values
            noun_word_indices: str, ' '.joined noun word indices
        """
        scene_id = series['scan_id']
        dataset_name = series['dataset']
        target_id = series['target_id']
        distractor_ids = series['distractor_ids']
        anchor_ids = series['anchor_ids']
        is_train = series['is_train']
        assignment_id = series['assignmentid']
        tokens = series['tokens']
        view_dependent = series['view_dependent']
        view_dependent_explicit = series['view_dependent_explicit']

        # all the instance class in the scene
        gt_instance_class_list = self.instance_class_list_by_scene_id[scene_id]
        gt_target_class = gt_instance_class_list[target_id]
        pred_target_class = self.predicted_target_class_by_assignment_id[assignment_id]

        if not distractor_ids:
            distractor_ids = [i for i, c in enumerate(gt_instance_class_list)
                              if c == gt_target_class and i != target_id]

        anchor_hash_list = list(map(lambda x: generate_hash(scene_id, x), anchor_ids))
        anchor_class_set = set([self.storage.get_class(h) for h in anchor_hash_list])
        clutter_ids = [i for i in range(len(gt_instance_class_list))
                       if i != target_id and i not in distractor_ids and i not in anchor_ids]
        np.random.shuffle(clutter_ids)

        # ids from the scene in the sorted order
        instance_ids = [target_id] + distractor_ids + anchor_ids + clutter_ids
        instance_ids = instance_ids[:self.max_context]
        instance_hash_list = [generate_hash(scene_id, i) for i in instance_ids]
        gt_instance_class_list = [self.storage.get_class(h) for h in instance_hash_list]
        pred_instance_class_list = [self.predicted_instance_class_list_by_hash[h] for h in instance_hash_list]

        anchor_id_set = set(range(len(distractor_ids) + 1, len(distractor_ids) + len(anchor_ids) + 1))
        # list of instance_ids in which belongs to the anchor class but not in the anchor_id_set
        false_anchor_ids = [i for i, c in enumerate(gt_instance_class_list)
                            if c in anchor_class_set and i not in anchor_id_set]

        # print('assignment_id: {}, instance_class_list: {}'.format(assignment_id, gt_instance_class_list))

        # randomly shuffle local indices
        new_instance_local_ids = np.random.permutation(len(instance_ids))

        # converts the old local index to a new local index
        new_index_dict = {j: i for i, j in enumerate(new_instance_local_ids)}
        new_target_id = new_index_dict[0]
        new_distractor_ids = [new_index_dict[i] for i in range(1, len(distractor_ids) + 1)]
        new_anchor_ids = [new_index_dict[i] for i in anchor_id_set]
        new_false_anchor_ids = [new_index_dict[i] for i in false_anchor_ids]

        new_gt_instance_classes = [gt_instance_class_list[i] for i in new_instance_local_ids]
        new_pred_instance_classes = [pred_instance_class_list[i] for i in new_instance_local_ids]
        new_instance_hashs = np.ones((self.max_context,), dtype=np.int64) * -1
        new_instance_hashs[:len(new_instance_local_ids)] = [instance_hash_list[i] for i in new_instance_local_ids]

        # labels
        # 0: irrelevant items
        # 1: true target
        # 2: false target (distractors)
        # 3: true anchor(s)
        # 4: false anchor(s)
        new_instance_labels = np.zeros((self.max_context,), dtype=np.int64)
        new_instance_labels[new_target_id] = 1
        new_instance_labels[new_distractor_ids] = 2
        new_instance_labels[new_anchor_ids] = 3
        new_instance_labels[new_false_anchor_ids] = 4

        noun_word_indices = self.noun_word_indices_by_assignment_id[assignment_id]

        if self.use_predicted_class:
            final_instance_classes = [cl[0] for cl in new_pred_instance_classes]
            mask_target_cls = pred_target_class
            target_mask = [str(i) for i, cl in enumerate(new_pred_instance_classes)
                           if mask_target_cls in cl[:self.target_mask_k]]
        else:
            final_instance_classes = new_gt_instance_classes
            mask_target_cls = gt_target_class
            target_mask = [str(i) for i, c in enumerate(final_instance_classes) if c == mask_target_cls]

        item_dict = {
            'dataset_names': dataset_name,
            'instance_types': final_instance_classes,
            'target_class': gt_target_class,
            'target_mask': ' '.join(target_mask),
            'target_labels': self.index_by_target_class[gt_target_class],
            'labels': new_instance_labels,
            'assignment_ids': assignment_id,
            'utterances': ' '.join(tokens),
            'hashs': new_instance_hashs,
            'view_dependent': view_dependent,
            'view_dependent_explicit': view_dependent_explicit,
            'noun_word_indices': ' '.join(list(map(str, noun_word_indices))),
        }
        return item_dict


def fetch_instance_items_and_storage_dict(
        args: Namespace,
        split_name: str) -> Tuple[Dict[str, List[Any]], InstanceStorage]:
    storage = fetch_instance_storage(dataset_name=args.dataset_name)

    if args.use_custom_df:
        df = fetch_custom_data_frame(
            dataset_name=args.dataset_name,
            df_path=Path(args.custom_df_path))
    else:
        df = fetch_unified_data_frame(
            dataset_name=args.dataset_name,
            split_name=split_name,
            use_view_independent=args.use_view_independent,
            use_view_dependent_explicit=args.use_view_dependent_explicit,
            use_view_dependent_implicit=args.use_view_dependent_implicit)

    if args.dataset_name == 'nr3d' and split_name == 'test' and not args.use_custom_df:
        assignment_ids = torch.load(str(fetch_nr3d_eval_assignment_id_list_path()))
        df = df.loc[df.assignmentid.isin(assignment_ids)]

    sampler = InstanceSampler(
        dataset_name=args.dataset_name,
        label_type=args.label_type,
        split_name=split_name,
        storage=storage,
        max_distractors=args.max_distractors,
        use_predicted_class=args.use_predicted_class,
        target_mask_k=args.target_mask_k,
        num_points=args.num_points)
    if args.debug:
        df = df.iloc[:100]
    instance_items = sampler.fetch_instances(df)

    return instance_items, storage


class NewDataset(Dataset):
    def __init__(
            self,
            args: Namespace,
            is_train: bool,
            use_mask_loss: bool,
            storage: InstanceStorage,
            encoded_value_dict: Dict[str, List[Any]]):
        Dataset.__init__(self)
        self.args = args
        self.is_train = is_train
        self.use_mask_loss = use_mask_loss
        self.storage = storage
        self.encoded_value_dict = encoded_value_dict

        self.DATASET_KEYS = {'labels', 'wheres', 'assignment_ids', 'binary_labels', 'object_attention_mask',
                             'utterance_attention_mask', 'target_mask', 'target_labels', 'input_ids',
                             'hashs', 'attention_mask'}
        if self.use_mask_loss:
            self.DATASET_KEYS.add('gt_input_ids')

    def __len__(self):
        return len(self.encoded_value_dict['labels'])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Fetch utterance-related information and returns it as a dictionary
        :param index: int, index
        :return:
            input_ids: torch.int64, (num_encoded, ), input word ids from encodings
            attention_mask: torch.bool, (num_encoded, ), mask information from encodings
            wheres: torch.int64/int, integer (encoded) position of the target instance
            labels: torch.int64, (num_encoded, ), integer labels of instances
            instance_ids: torch.int64/int, (???, ), instance_classes of objects
            assignment_ids: torch.int64/int, unique id of utterance (-1 in sr3d)
            bboxs: torch.float32, (num_encoded, 6), bounding-box information
            points: Optional[torch.float32], (num_encoded, num_points, 6), points
            rotation_index: float32, index of applied rotation to the bounding box
            viewpoint_annotation: int64, ground-truth viewpoint annotation (only if view-dependent-explicit)
            target_mask: torch.bool, (num_encoded, ), bool (predicted) target class indicator
            target_labels: torch.int64, index of target class (out of 76)
            gt_input_ids: Optional[torch.int64]
        """
        item = dict()

        with torch.no_grad():
            # copy values from the encoded value dict
            for key in self.DATASET_KEYS:
                item[key] = torch.tensor(self.encoded_value_dict[key][index])

            # prepare hash indices and values
            encoded_hashs = self.encoded_value_dict['hashs'][index]
            hash_indices = np.where(encoded_hashs >= 0)[0]
            hash_values = encoded_hashs[hash_indices]

            # fetch bbox/point values from storage w.r.t hash information
            encoded_bboxs = np.zeros((*encoded_hashs.shape, 6), dtype=np.float32)
            for i, h in zip(hash_indices, hash_values):
                encoded_bboxs[i, ...] = self.storage.get_bbox(h)

            # apply rotation w.r.t the arg options
            assignment_id = self.encoded_value_dict['assignment_ids'][index]
            item['bboxs'] = torch.tensor(encoded_bboxs)

        return item


BATCH_KEYS = [
    'dataset_names', 'assignment_ids', 'utterances', 'instance_types', 'view_dependent',
    'view_dependent_explicit', 'labels', 'hashs', 'target_mask', 'target_labels', 'noun_word_indices']


def fetch_dataset(
        args: Namespace,
        split_name: str,
        tokenizer: DistilBertTokenizerFast) -> Dataset:
    instance_items_dict, storage = fetch_instance_items_and_storage_dict(args, split_name)

    rng = np.random.default_rng(0)
    encoded_value_dict = encode_data_with_tokenizer(
        tokenizer=tokenizer,
        rng=rng,
        use_mask_loss=args.use_mask_loss,
        **{k: instance_items_dict[k] for k in BATCH_KEYS})
    dataset = NewDataset(
        args=args,
        is_train=split_name == 'train',
        use_mask_loss=args.use_mask_loss,
        storage=storage,
        encoded_value_dict=encoded_value_dict)
    return dataset


def fetch_standard_test_dataset(
        args: Namespace,
        tokenizer: DistilBertTokenizerFast) -> Dataset:
    assert args.use_standard_test
    assert args.use_mentions_target_class_only
    assert args.use_correct_guess_only
    assert not args.use_bbox_random_rotation_independent
    assert not args.use_bbox_random_rotation_dependent_explicit
    assert not args.use_bbox_random_rotation_dependent_implicit

    instance_items_dict, storage = fetch_instance_items_and_storage_dict(args=args, split_name='test')
    rng = np.random.default_rng(0)

    encoded_value_dict = encode_data_with_tokenizer(
        tokenizer=tokenizer,
        rng=rng,
        use_mask_loss=args.use_mask_loss,
        **{k: instance_items_dict[k] for k in BATCH_KEYS})
    dataset = NewDataset(
        args=args,
        is_train=False,
        use_mask_loss=args.use_mask_loss,
        storage=storage,
        encoded_value_dict=encoded_value_dict)
    return dataset

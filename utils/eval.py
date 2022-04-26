from typing import Tuple

import pandas as pd
import torch

from data.unified_data_frame import fetch_unified_data_frame
from utils.directory import fetch_nr3d_eval_assignment_id_list_path


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def hardness_from_stimulus_string(s: str) -> bool:
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractor_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractor_ids = \
            s.split('-', maxsplit=4)

    # instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    # target_id = int(target_id)
    distractor_ids = [int(i) for i in distractor_ids.split('-') if i != '']
    assert len(distractor_ids) == n_objects - 1
    return n_objects > 2


def load_evaluation_df() -> pd.DataFrame:
    df = fetch_unified_data_frame(
        dataset_name='nr3d',
        split='test',
        use_view_independent=True,
        use_view_dependent_explicit=True,
        use_view_dependent_implicit=True)
    assignment_ids = torch.load(str(fetch_nr3d_eval_assignment_id_list_path()))
    df = df.loc[df.assignmentid.isin(assignment_ids)]
    df['hard'] = df.stimulus_id.apply(hardness_from_stimulus_string)
    return df


def compute_accuracy(matched, mask) -> Tuple[float, int, int]:
    df = matched.loc[mask] if mask is not None else matched
    count = sum(df)
    total = len(df)
    if total == 0:
        return 0, 0, 1
    acc = count / total * 100
    return acc, count, total


def compute_stat_from_eval_dict(val_data, eval_dict) -> pd.DataFrame:
    matched = val_data['assignmentid'].map(eval_dict)
    easy = ~val_data.hard
    hard = val_data.hard
    vd = val_data.view_dependent
    vi = ~vd
    mask_info_list = [
        ('overall', None),
        ('easy', easy),
        ('hard', hard),
        ('vd', vd),
        ('vi', vi),
    ]
    value_dict = dict()
    for name, mask in mask_info_list:
        acc, count, total = compute_accuracy(matched, mask)
        value_dict[name] = acc
    return pd.DataFrame({k: ['{:4.1f}'.format(v)] for k, v in value_dict.items()})

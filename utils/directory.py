import re
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple


def mkdir_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
    return path


def fetch_data_root_dir() -> Path:
    return Path.cwd() / 'resources'


def check_valid_dataset_name(dataset_name: str) -> bool:
    return dataset_name in {'nr3d', 'sr3d'}


def fetch_nr3d_eval_assignment_id_list_path() -> Path:
    path = fetch_data_root_dir() / 'eval_nr3d_assignment_ids.pt'
    assert path.exists()
    return path


def fetch_index_by_instance_class_path(
        label_type: str,
        dataset_name: str) -> Path:
    path = fetch_labels_dir(label_type) / 'index_by_instance_class/{}.json'.format(dataset_name)
    assert path.exists()
    return path


def fetch_instance_item_dict_storage_path(dataset_name: str) -> Path:
    assert check_valid_dataset_name(dataset_name)
    return fetch_data_root_dir() / 'storage_{}.pt'.format(dataset_name)


def fetch_target_class_list_path(label_type: str) -> Path:
    return fetch_labels_dir(label_type=label_type) / 'target_class.txt'


def fetch_labels_dir(label_type: str) -> Path:
    return fetch_data_root_dir() / 'labels/{}'.format(label_type)


def fetch_predicted_target_class_by_assignment_id_path(label_type: str):
    return fetch_labels_dir(label_type) / 'predicted_target_class_by_assignment_id.pt'


def fetch_instance_class_list_by_scene_id_path(label_type: str) -> Path:
    return fetch_labels_dir(label_type) / 'instance_class_list_dict.pt'


def fetch_predicted_instance_class_list_by_hash_path(label_type: str) -> Path:
    return fetch_labels_dir(label_type) / 'predicted_instance_class_list_by_hash.pt'


def fetch_unified_data_frames_dir() -> Path:
    return fetch_data_root_dir() / 'unified_data_frames'


def fetch_unified_data_frames_path(dataset_name: str) -> Path:
    return fetch_unified_data_frames_dir() / '{}.csv'.format(dataset_name)


def fetch_pretrain_path_list(args: Namespace) -> List[Tuple[int, Path]]:
    if args.pretrain_path is None:
        raise FileNotFoundError('pretrain_path was not found: {}'.format(args.pretrain_path))

    in_dir = Path(args.pretrain_path)
    if (in_dir / 'pytorch_model.bin').exists() and (in_dir / 'optimizer.pt').exists():
        in_dir = in_dir.parent

    checkpoint_dir_list = list(filter(lambda x: x.is_dir(), in_dir.glob('*')))
    valid_num_iter_list = []
    valid_checkpoint_dir_list = []
    for dir_path in checkpoint_dir_list:
        resp = re.findall(r'checkpoint-([\d]+)', dir_path.name)
        if resp:
            num_iter = int(resp[0])
            valid_num_iter_list.append(num_iter)
            valid_checkpoint_dir_list.append(dir_path)
    num_dir_list = sorted(zip(valid_num_iter_list, valid_checkpoint_dir_list))
    if args.eval_reverse:
        num_dir_list = num_dir_list[::-1]
    if args.eval_single_only:
        num_dir_list = [num_dir_list[0]]
    return num_dir_list

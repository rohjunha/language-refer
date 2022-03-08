import json
from pathlib import Path
from typing import Dict

import torch

from utils.directory import fetch_instance_item_dict_storage_path, fetch_index_by_instance_class_path
from utils.logging import get_logger

logger = get_logger(__name__)


def hash_str(hash: int) -> str:
    return '{:16d}'.format(hash)


def fetch_index_by_instance_class(
        label_type: str,
        dataset_name: str) -> Dict[str, int]:
    with open(str(fetch_index_by_instance_class_path(label_type=label_type, dataset_name=dataset_name)), 'r') as file:
        index_by_instance_class = json.load(file)
    return index_by_instance_class


class InstanceStorage:
    def __init__(
            self,
            label_type: str,
            dataset_name: str,
            storage_path: Path):
        assert storage_path.exists()
        self.label_type = label_type
        self.dataset_name = dataset_name
        self.item_by_hash = {int(k): v for k, v in torch.load(str(storage_path)).items()}
        self.index_by_instance_class: Dict[str, int] = fetch_index_by_instance_class(
            label_type=label_type,
            dataset_name=dataset_name)

    def get_scan_id(self, h: int):
        return self.item_by_hash[h]['scene_id']

    def get_class(self, h: int):
        return self.item_by_hash[h]['class']

    def get_bbox(self, h: int):
        return self.item_by_hash[h]['bbox']

    def get_instance_class_index(self, h: int) -> int:
        cls = self.get_class(h)
        return self.index_by_instance_class[cls]


def fetch_instance_storage(dataset_name: str) -> InstanceStorage:
    instance_item_dict_path = fetch_instance_item_dict_storage_path(dataset_name)
    return InstanceStorage(label_type='revised', dataset_name=dataset_name, storage_path=instance_item_dict_path)

from pathlib import Path

import torch

from utils.directory import fetch_instance_item_dict_storage_path
from utils.logging import get_logger

logger = get_logger(__name__)


def hash_str(hash: int) -> str:
    return '{:16d}'.format(hash)


class InstanceStorage:
    def __init__(self, storage_path: Path):
        assert storage_path.exists()
        self.item_by_hash = {int(k): v for k, v in torch.load(str(storage_path)).items()}

    def get_scan_id(self, h: int):
        return self.item_by_hash[h]['scene_id']

    def get_class(self, h: int):
        return self.item_by_hash[h]['class']

    def get_bbox(self, h: int):
        return self.item_by_hash[h]['bbox']


def fetch_instance_storage(dataset_name: str) -> InstanceStorage:
    instance_item_dict_path = fetch_instance_item_dict_storage_path(dataset_name)
    return InstanceStorage(instance_item_dict_path)

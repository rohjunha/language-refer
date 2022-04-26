import json
import math
import random
from typing import List, Tuple, Set, Dict

import numpy as np

from utils.directory import fetch_viewpoint_annotation_path, fetch_assignment_id_by_utterance_id_path

THETA_FROM_ANNOTATION_INDEX = {
    0: 1.5 * math.pi,
    1: math.pi,
    2: 0.5 * math.pi,
    3: 0
}


def fetch_annotation_by_assignment_id() -> Dict[int, List[int]]:
    annotation_path = fetch_viewpoint_annotation_path()
    assert annotation_path.exists()
    with open(str(annotation_path), 'r') as file:
        annotation_by_utterance_id = json.load(file)

    abyu_path = fetch_assignment_id_by_utterance_id_path()
    assert abyu_path.exists()
    with open(str(abyu_path), 'r') as file:
        assignment_id_by_utterance_id = json.load(file)

    annotation_by_assignment_id = dict()
    for scan_id, annotation_list_by_utterance_id in annotation_by_utterance_id.items():
        for utterance_id, annotation_list in annotation_list_by_utterance_id.items():
            assignment_id = assignment_id_by_utterance_id[utterance_id]
            annotation_by_assignment_id[assignment_id] = annotation_list
    return annotation_by_assignment_id


def _normalize_bbox(bbox: np.ndarray) -> np.ndarray:
    """
    Apply normalization on 3D positions of the bbox
    :param bbox: (max_context, 6), np.float32
    :return: normalized bbox (max_context, 6), np.float32
    """
    new_bbox = np.array(bbox)
    mask = (np.abs(bbox) > 1e-5).any(axis=1)
    new_bbox[mask, :3] -= np.mean(bbox[mask, :3], axis=0)
    return new_bbox


def _rotate_bbox(bbox: np.ndarray, th: float):
    """
    Apply a 2D xy-plane rotation on x, y positions of the bbox, given theta
    :param bbox: (max_context, 6), np.float32
    :param th: float
    :return: rotated bbox (max_context, 6), np.float32
    """
    new_bbox = np.array(bbox)
    cc, ss = math.cos(th), math.sin(th)
    rm = np.array([[cc, ss], [-ss, cc]])
    new_bbox[:, :2] = new_bbox[:, :2] @ rm
    return new_bbox


def _randomly_rotate_bbox(bbox: np.ndarray):
    """
    Apply a random 2D xy-plane rotation on x, y positions of the bbox
    :param bbox: (max_context, 6), np.float32
    :return: rotated bbox (max_context, 6), np.float32
    """
    th = random.uniform(-np.pi, np.pi)
    return _rotate_bbox(bbox, th)


def _fetch_rotated_bbox(
        bbox: np.ndarray,
        annotation_indices: List[int]) -> Tuple[int, np.ndarray]:
    if not annotation_indices:
        return -1, bbox
    annotation_index = random.choice(annotation_indices)
    theta = THETA_FROM_ANNOTATION_INDEX[annotation_index]
    return annotation_index, _rotate_bbox(bbox, theta)


def _randomly_rotate_box(bbox: np.ndarray) -> Tuple[int, np.ndarray]:
    return _fetch_rotated_bbox(bbox, [0, 1, 2, 3])


def _randomly_valid_rotated_bbox(bbox: np.ndarray, indices: List[int]) -> Tuple[np.ndarray, bool]:
    if len(indices) == 4:
        _, bbox = _randomly_rotate_box(bbox)
        return bbox, True
    elif not indices:
        _, bbox = _randomly_rotate_box(bbox)
        return bbox, False
    else:
        invalid_indices = [i for i in range(4) if i not in indices]
        binary_indices = [random.choice(indices), random.choice(invalid_indices)]
        index = random.choice(binary_indices)
        theta = THETA_FROM_ANNOTATION_INDEX[binary_indices[index]]
        return _rotate_bbox(bbox, theta), index == 0


class BoundingBoxHandler:
    def __init__(
            self,
            is_train: bool,
            use_valid_classification: bool,
            use_bbox_random_rotation_independent: bool,
            use_bbox_random_rotation_dependent_explicit: bool,
            use_bbox_random_rotation_dependent_implicit: bool):
        self.is_train = is_train
        self.use_valid_classification = use_valid_classification
        self.use_bbox_random_rotation_independent = use_bbox_random_rotation_independent
        self.use_bbox_random_rotation_dependent_explicit = use_bbox_random_rotation_dependent_explicit
        self.use_bbox_random_rotation_dependent_implicit = use_bbox_random_rotation_dependent_implicit
        self.annotation_by_assignment_id = fetch_annotation_by_assignment_id()

    def fetch_annotation(self, assignment_id: int) -> int:
        """
        fetch annotation indices by assignment id
        :param assignment_id: unique utterance id (-1 in sr3d)
        :return: annotation indices or -1 if no annotation is available
        """
        if assignment_id in self.annotation_by_assignment_id:
            annotations = self.annotation_by_assignment_id[assignment_id]
            if len(annotations) == 1:
                return annotations[0]
        return -1

    def fetch_assignment_id_set_with_annotations(self) -> Set[int]:
        """
        Returns a set of assignment ids that have annotations
        :return: a set of assignment ids that have annotations
        """
        return {k for k, v in self.annotation_by_assignment_id.items() if v and len(v) == 1}

    def update_bbox_from_annotation(
            self,
            bbox: np.ndarray,
            assignment_id: int,
            view_dependent: bool,
            view_dependent_explicit: bool) -> Tuple[np.ndarray, bool]:
        view_independent = not view_dependent
        view_dependent_implicit = view_dependent and not view_dependent_explicit
        indices = self.annotation_by_assignment_id[assignment_id]

        # In evaluation, no rotation is applied and just check whether the instruction is valid in the current view.
        if not self.is_train:
            valid = True if (view_independent or (0 in indices)) else False
            return bbox, valid

        # In training,
        else:
            if view_independent:
                if self.use_bbox_random_rotation_independent:
                    _, bbox = _fetch_rotated_bbox(bbox, [0, 1, 2, 3])
                return bbox, True
            else:
                return _randomly_valid_rotated_bbox(bbox, indices)

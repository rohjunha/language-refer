import re
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.directory import fetch_unified_data_frames_path


def parse_int_list_from_str(query: str) -> List[int]:
    return list(map(int, re.findall(r'([\d]+)', query)))


def parse_str_list_from_str(query: str) -> List[str]:
    return re.findall(r"\'([\w| ]+)\'", query)


def parse_point(query) -> Union[None, np.ndarray]:
    if isinstance(query, float) and np.isnan(query):
        return None
    else:
        return np.array(list(map(lambda x: list(map(float, x.split())), re.findall(r'\[([-\d. ]+)\]', query))))


def encode_bbox(value: np.ndarray) -> str:
    return ','.join(['{:.8f}'.format(f) for f in value.reshape(-1, )])


def decode_bbox(query: str) -> np.ndarray:
    return np.array(list(map(float, query.split(',')))).reshape(-1, 6)


def update_column_int(df: DataFrame, column_name: str):
    if column_name in df:
        df[column_name] = df[column_name].apply(int)


def read_refer_data(refer_path: Path):
    df = pd.read_csv(str(refer_path))
    update_column_int(df, 'target_id')
    update_column_int(df, 'assignment_id')

    if 'anchor_ids' in df:
        df['anchor_ids'] = df['anchor_ids'].apply(parse_int_list_from_str)
        if 'anchors_types' in df:
            df['anchor_types'] = df['anchors_types'].apply(parse_str_list_from_str)
        if 'anchor_types' in df and isinstance(df['anchor_types'][0], str):
            df['anchor_types'] = df['anchor_types'].apply(parse_str_list_from_str)
        df['distractor_ids'] = df['distractor_ids'].apply(parse_int_list_from_str)
    if 'instance_text_labels' in df:
        df['instance_text_labels'] = df['instance_text_labels'].apply(parse_str_list_from_str)
        df['instance_tags'] = df['instance_tags'].apply(parse_int_list_from_str)
        df['instance_points'] = df['instance_points'].apply(parse_point)
        df['instance_bboxs'] = df['instance_bboxs'].apply(decode_bbox)
    if 'tokens' in df:
        df['tokens'] = df['tokens'].apply(parse_str_list_from_str)
    if 'instance_distractor_ids' in df:
        df['instance_distractor_ids'] = df['instance_distractor_ids'].apply(parse_int_list_from_str)
    if 'instance_orientations' in df:
        df['instance_orientations'] = df['instance_orientations'].apply(parse_int_list_from_str)

    return df


def contains_explicit_viewpoint_description(x):
    query_single_words = ['facing', 'rotate', 'face', 'if', 'when']
    query_double_words = [('looking', 'at')]
    for w in query_single_words:
        if w in x:
            return True
    for w1, w2 in query_double_words:
        if w1 in x and w2 in x and np.where(np.array(x) == w1)[0][0] + 1 == np.where(np.array(x) == w2)[0][0]:
            return True
    return False


def fetch_unified_data_frame(
        dataset_name: str,
        split_name: str,
        use_view_independent: bool = True,
        use_view_dependent_explicit: bool = True,
        use_view_dependent_implicit: bool = True) -> DataFrame:
    """
    Creates a unified data frame from raw data frames
    :return: df with columns
        scan_id: str (e.g. scene0668_00)
        stimulus_id: str, contains scan_id, target_class, number of target instances and ids
            (e.g. scene0668_00-pillow-6-23-16-21-22-29-34)
        assignmentid: int, unique integer id for nr3d (-1 for sr3d)
        target_id: int, target instance id in the scene
        target_class: str, target class text
        distractor_ids: List[int], distractor ids (does not include the target id)
        anchor_ids: List[int], only sr3d has this information
        anchor_types: List[str], anchor classes, only sr3d has this information
        dataset: str, dataset name
        other columns: bool, flags
    """
    assert dataset_name in {'nr3d', 'sr3d', 'nr3d+sr3d'}
    assert split_name in {'train', 'test'}
    assert use_view_independent or use_view_dependent_explicit or use_view_dependent_implicit

    if dataset_name != 'nr3d+sr3d':
        unified_df_path = fetch_unified_data_frames_path(dataset_name)
        df = read_refer_data(unified_df_path)
    else:
        df_nr3d = read_refer_data(fetch_unified_data_frames_path('nr3d'))
        df_sr3d = read_refer_data(fetch_unified_data_frames_path('sr3d'))
        df = pd.concat([df_nr3d, df_sr3d])

    df = df.loc[df['mentions_target_class'] == True]
    df = df.loc[df['correct_guess'] == True]
    vi_mask = df.view_dependent.apply(lambda x: not x) if use_view_independent else False
    vd_mask = df.view_dependent.apply(lambda x: x)
    vde_mask = df.view_dependent_explicit if use_view_dependent_explicit else False
    vdi_mask = vd_mask & ~vde_mask if use_view_dependent_implicit else False
    df = df[vi_mask | vde_mask | vdi_mask]

    split_mask = df.is_train == (True if split_name == 'train' else False)
    df = df[split_mask]

    df.reset_index(drop=True, inplace=True)
    return df


def fetch_custom_data_frame(
        dataset_name: str,
        df_path: Path):
    assert dataset_name in {'nr3d', 'sr3d', 'nr3d+sr3d'}
    df = read_refer_data(df_path)
    return df

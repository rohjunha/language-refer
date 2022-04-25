import json
import os
import pprint
from argparse import ArgumentTypeError, ArgumentParser, Namespace
from datetime import datetime
from os import path as osp
from pathlib import Path
from typing import Optional, List

from utils.random import set_random_seed


def str2bool(v):
    """
    Boolean values for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)


def create_dir(dir_path):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def mkdir(dir_path: Path) -> Path:
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    return dir_path


def fetch_parser() -> ArgumentParser:
    """
    Returns ArgumentParser with common options
    :return: ArgumentParser with common options
    """
    parser = ArgumentParser(description='ReferIt3D Nets + Ablations')

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--experiment-tag', type=str, default='')
    parser.add_argument('--dataset-name', type=str, default='nr3d', help='the name of the dataset {nr3d, sr3d}')
    parser.add_argument('--extra-dataset-name', type=str, default=None)
    parser.add_argument('--label-type', type=str, default='revised')
    parser.add_argument('--random-seed', type=int, default=2022)

    parser.add_argument('--max-distractors', type=int, default=51)
    parser.add_argument('--max-test-objects', type=int, default=87)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--use-custom-df', action='store_true', default=False)
    parser.add_argument('--custom-df-path', type=str, default='')
    parser.add_argument('--use-target-mask', action='store_true', default=False)
    parser.add_argument('--use-predicted-class', action='store_true', default=True)
    parser.add_argument('--target-mask-k', type=int, default=4)
    parser.add_argument('--normalize-bbox', action='store_true', default=False,
                        help='Normalize the 3D position of bboxs')

    parser.add_argument('--use-view-independent', type=str2bool, default=True, help='Use view independent utterances')
    parser.add_argument('--use-view-dependent-explicit', type=str2bool, default=True,
                        help='Use view dependent (explicit) utterances')
    parser.add_argument('--use-view-dependent-implicit', type=str2bool, default=True,
                        help='Use view dependent (implicit) utterances')

    parser.add_argument('--use-bbox-annotation-only', action='store_true', default=False,
                        help='Flag whether the model is doing a viewpoint prediction or a referring task.')
    parser.add_argument('--use-bbox-random-rotation-independent', type=str2bool, default=True)
    parser.add_argument('--use-bbox-random-rotation-dependent-explicit', type=str2bool, default=False)
    parser.add_argument('--use-bbox-random-rotation-dependent-implicit', type=str2bool, default=False)
    parser.add_argument('--bbox-fixed-rotation-independent-index', type=int, default=-1)
    parser.add_argument('--bbox-fixed-rotation-dependent-explicit-index', type=int, default=-1)
    parser.add_argument('--bbox-fixed-rotation-dependent-implicit-index', type=int, default=-1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--weight-ref', type=float, default=1.0, help='Weight on the referring loss or viewpoint pred.')
    parser.add_argument('--weight-clf', type=float, default=0.5, help='Weight on the object classification loss')

    parser.add_argument('--output-dir-prefix', type=str, default='results')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--pretrain-path', type=str, default=None)
    parser.add_argument('--save-args', type=str2bool, default=True)
    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--save-steps', type=int, default=2000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pin-memory', action='store_true', default=False)

    # epoch parameters
    parser.add_argument('--no-cuda', type=str2bool, default=False)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)

    # training arguments.
    parser.add_argument('--train-custom', type=str2bool, default=False)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--total-training-epochs', type=int, default=80)

    # evaluation arguments.
    parser.add_argument('--eval-reverse', type=str2bool, default=True)
    parser.add_argument('--eval-single-only', type=str2bool, default=True)

    return parser


def parse_argument_parser(
        parser: ArgumentParser,
        notebook_options: Optional[List[str]] = None) -> Namespace:
    """
    Parse the arguments from an ArgumentParser instance
    :param parser: ArgumentParser
    :param notebook_options: List[str], options from notebook (e.g. ['--max-distractors', '100'])
    :return: Namespace, parsed arguments
    """
    if notebook_options is not None:
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()
    return args


def post_process_arguments(
        args: Namespace,
        is_train: bool,
        verbose: bool = False):
    """
    Update the arguments
    :param args: Namespace, parsed arguments
    :param is_train: bool, if it is training or evaluation
    :param verbose: bool, option to print out arguments
    :return: Namespace, updated arguments
    """
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    if args.output_dir_prefix:
        output_dir = mkdir(Path.cwd() / args.output_dir_prefix)
        if args.pretrain_path is not None:
            args.pretrain_path = output_dir / args.pretrain_path
            assert args.pretrain_path.exists()
            if not is_train and not args.experiment_tag:
                args.experiment_tag = 'eval-{}'.format(str(args.pretrain_path.parent).split('/')[-1])
                print('Automatically set the experiment tag: {}'.format(args.experiment_tag))
            args.pretrain_path = str(args.pretrain_path)
        args.output_dir = str(mkdir(output_dir / args.experiment_tag / timestamp))
    assert args.experiment_tag

    if not is_train:
        args.max_distractors = args.max_test_objects

    # turn off fp16 mode if cuda is not used
    if args.no_cuda:
        args.fp16 = False

    if args.random_seed >= 0:
        set_random_seed(args.random_seed)

    if args.save_args:
        out = osp.join(args.output_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    assert args.dataset_name in {'nr3d', 'sr3d'}
    eff_dataset_name = args.dataset_name
    if args.extra_dataset_name is not None:
        assert args.extra_dataset_name in {'nr3d', 'sr3d'}
        assert args.extra_dataset_name != args.dataset_name
        eff_dataset_name = 'nr3d+sr3d'
    args.dataset_name = eff_dataset_name
    assert args.dataset_name in {'nr3d', 'sr3d', 'nr3d+sr3d'}

    if verbose:
        print(pprint.pformat(vars(args)))

    step_size = args.batch_size * args.gpus
    _warmup_steps = args.warmup_steps
    args.warmup_steps = args.warmup_steps // step_size
    args.total_training_steps = args.total_training_epochs // step_size
    print('Altered the warmup steps from {} to {}'.format(_warmup_steps, args.warmup_steps))
    print('Set the total training steps: {}'.format(args.total_training_steps))

    return args


def fetch_arguments(
        notebook_options=None,
        verbose: bool = True) -> Namespace:
    parser = fetch_parser()

    args = parse_argument_parser(
        parser=parser,
        notebook_options=notebook_options)

    is_train = args.mode == 'train'
    args = post_process_arguments(
        args=args,
        is_train=is_train,
        verbose=verbose)
    return args

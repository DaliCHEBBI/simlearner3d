"""Generation of a toy dataset for testing purposes."""

import os
import os.path as osp
import sys
import argparse
from enum import Enum
import typing
import hydra
from omegaconf import DictConfig
import csv 
import shutil
import secrets

# to use from CLI.
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))

from simlearner3d.processing.dataset.utils import (
    read_images_and_create_full_data_obj
)
from simlearner3d.processing.dataset.hdf5 import HDF5Dataset  # noqa


TOY_IMAGE_DATA = ("tests/data/DMTrain_ENSH-0021764_1_0021765_1_0000_Im1.tif",
                  "tests/data/DMTrain_ENSH-0021764_1_0021765_1_0000_Im2.tif",
                  "tests/data/DensifyPx_DMTrain_ENSH-0021764_1_0021765_1_0000_Im1.tif",
                  "tests/data/Nocc_clean_DensifyPx_DMTrain_ENSH-0021764_1_0021765_1_0000_Im1.tif")

TOY_DATASET_HDF5_PATH = "tests/data/toy_dataset.hdf5"

class TASK_NAMES(Enum):
    MAKE_HDF5_DATASET = "make_hdf5"
    PREPARE_DATASET = "prepare_dataset"

DEFAULT_TASK = TASK_NAMES.MAKE_HDF5_DATASET.value
TASK_NAME_DETECTION_STRING = "task.task_name="

def read_list_from_file(f_):
    with open(f_, 'r') as h:
        filelist = h.read().splitlines()
        filelist = [x.split()[0] for x in filelist]
    return filelist

def make_arg_parser():
    prepare_dataset_parser = argparse.ArgumentParser()
    prepare_dataset_parser.add_argument(
        "--left",
        "-l",
        type=str,
        help="txt file containing full names of left images",
        required=True,
    )
    prepare_dataset_parser.add_argument(
        "--right",
        "-r",
        type=str,
        help="txt file containing full names of right images",
        required=True,
    )
    prepare_dataset_parser.add_argument(
        "--disp",
        "-d",
        type=str,
        help="txt file containing full names of disparity images",
        required=True,
    )
    prepare_dataset_parser.add_argument(
        "--masq",
        "-m",
        type=str,
        help="txt file containing full names of occlusion masqs",
        required=True,
    )
    prepare_dataset_parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        help="output directory for arrangin dataset",
        required=True,
    )
    return prepare_dataset_parser

def make_toy_dataset_from_test_file():
    """Prepare a toy dataset from a single, small LAS file.

    The file is first duplicated to get 2 LAS in each split (train/val/test),
    and then each file is splitted into .data files, resulting in a training-ready
    dataset loacted in td_prepared

    Args:
        src_las_path (str): input, small LAS file to generate toy dataset from
        split_csv (str): Path to csv with a `basename` (e.g. '123_456.las') and
        a `split` (train/val/test) columns specifying the dataset split.
        prepared_data_dir (str): where to copy files (`raw` subfolder) and to prepare
        dataset files (`prepared` subfolder)

    Returns:
        str: path to directory containing prepared dataset.

    """
    if os.path.isfile(TOY_DATASET_HDF5_PATH):
        os.remove(TOY_DATASET_HDF5_PATH)

    # TODO: update transforms ? or use a config ?
    HDF5Dataset(
        TOY_DATASET_HDF5_PATH,
        image_paths_by_split_dict={
            "train": [TOY_IMAGE_DATA],
            "val": [TOY_IMAGE_DATA],
            "test": [TOY_IMAGE_DATA],
        },
        images_pre_transform=read_images_and_create_full_data_obj,
        tile_width=1024,
        tile_height=1024,
        patch_size=768,
        subtile_width=50,
        sign_disp_multiplier=-1,
        masq_divider=255,
        train_transform=None,
        eval_transform=None,
    )
    return TOY_DATASET_HDF5_PATH

def populate_dataset(nb_train,
                     nb_val,
                     all_set_images,
                     _output_dir,
                     writer):
    while nb_train:
        a_set=(l,r,d,m)=secrets.choice(all_set_images)
        # put a_set in train folder and fill csv_file
        shutil.copy(l,_output_dir + '/train/' )
        shutil.copy(r,_output_dir + '/train/' )
        shutil.copy(d,_output_dir + '/train/' )
        shutil.copy(m,_output_dir + '/train/' )
        # fill csv
        writer.writerow([osp.basename(l),
                    osp.basename(r),
                    osp.basename(d),
                    osp.basename(m),
                    "train"])
        nb_train-=1
        all_set_images.remove(a_set)        
    while nb_val:
        a_set=(l,r,d,m)=secrets.choice(all_set_images)
        # put a_set in train folder and fill csv_file
        shutil.copy(l,_output_dir + '/val/' )
        shutil.copy(r,_output_dir + '/val/' )
        shutil.copy(d,_output_dir + '/val/' )
        shutil.copy(m,_output_dir + '/val/' )
        # fill csv
        writer.writerow([osp.basename(l),
                    osp.basename(r),
                    osp.basename(d),
                    osp.basename(m),
                    "val"])
        nb_val-=1
        all_set_images.remove(a_set)

    # the remaining a set as test dataset
    for (l,r,d,m) in all_set_images:
        # put a_set in train folder and fill csv_file
        shutil.copy(l,_output_dir + '/test/' )
        shutil.copy(r,_output_dir + '/test/' )
        shutil.copy(d,_output_dir + '/test/' )
        shutil.copy(m,_output_dir + '/test/' )
        # fill csv
        writer.writerow([osp.basename(l),
                    osp.basename(r),
                    osp.basename(d),
                    osp.basename(m),
                    "test"])

@hydra.main(config_path="../../../configs/dataset/", config_name="default.yaml")
def prepare_dataset(config: DictConfig):#, args: typing.Sequence[str]):
    #parser=make_arg_parser()
    #args_prepare=parser.parse_args(args)
    _left_names=read_list_from_file(config.prepare_dataset.get('left'))
    _right_names=read_list_from_file(config.prepare_dataset.get('right'))
    _disp_names=read_list_from_file(config.prepare_dataset.get('disp'))
    _masq_names=read_list_from_file(config.prepare_dataset.get('masq'))
    _output_dir=config.prepare_dataset.get('out_dir')

    os.makedirs(_output_dir,exist_ok=True)
    os.makedirs(_output_dir + '/train', exist_ok=True)
    os.makedirs(_output_dir + '/val',exist_ok=True)
    os.makedirs(_output_dir + '/test',exist_ok=True)

    """ split dataset into train val and test subsets """
    split_train= config.data_split.get('train')/100
    split_val  = config.data_split.get('val')/100
    split_test = config.data_split.get('test')/100

    all_set_images=list(zip(_left_names,_right_names,_disp_names,_masq_names))
    # train 
    nb_train=round(split_train*len(all_set_images))
    nb_val=round(split_val)*len(all_set_images)

    if os.path.isfile(_output_dir + '/split.csv'):
        with open(_output_dir + '/split.csv', 'a', newline='') as csv_split:
            writer=csv.writer(csv_split)
            populate_dataset(nb_train,
                             nb_val,
                             all_set_images,
                             _output_dir,
                             writer)
    else:
        with open(_output_dir + '/split.csv', 'w', newline='') as csv_split:
            writer=csv.writer(csv_split)  
            writer.writerow(["basename_l","basename_r","disparity","masq","split"])
            populate_dataset(nb_train,
                             nb_val,
                             all_set_images,
                             _output_dir,
                             writer)

if __name__ == "__main__":
    task_name = "make_hdf5"
    for arg in sys.argv:
        if TASK_NAME_DETECTION_STRING in arg:
            _, task_name = arg.split("=")
            break
    if task_name==TASK_NAMES.PREPARE_DATASET.value:
        prepare_dataset()

    if task_name==TASK_NAMES.MAKE_HDF5_DATASET.value:
        make_toy_dataset_from_test_file()

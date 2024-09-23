import glob
import json
from pathlib import Path
import subprocess as sp
from numbers import Number
from typing import Dict, List, Literal, Union,Tuple

import numpy as np
import pandas as pd
#from scipy.spatial import cKDTree
from simlearner3d.utils.utils import Data
import tifffile as tf 



SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
IMAGE_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[Tuple[str]]]


def find_file_in_dir(data_dir: str, basename: str) -> str:
    """Query files matching a basename in input_data_dir and its subdirectories.
    Args:
        input_data_dir (str): data directory
    Returns:
        [str]: first file path matching the query.
    """
    query = f"{data_dir}/**/{basename}"
    files = glob.glob(query, recursive=True)
    return files[0]


def read_images_and_create_full_data_obj(
        image_paths,
):
    return Data(
        _left=tf.imread(image_paths[0]),
        _right=tf.imread(image_paths[1]),
        _disp=tf.imread(image_paths[2]),
        _masq=tf.imread(image_paths[3]),
    )


def get_image_paths_by_split_dict(
    data_dir: str, split_csv_path: str
) -> IMAGE_PATHS_BY_SPLIT_DICT_TYPE:
    image_paths_by_split_dict: IMAGE_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames_l = split_df[split_df.split == phase].basename_l.tolist()
        basenames_r = split_df[split_df.split == phase].basename_r.tolist()
        basenames_d = split_df[split_df.split == phase].disparity.tolist()
        basenames_m = split_df[split_df.split == phase].masq.tolist()
        basenames=list(zip(basenames_l,basenames_r,basenames_d,basenames_m))
        # Reminder: an explicit data structure with ./val, ./train, ./test subfolder is required.
        image_paths_by_split_dict[phase] = [(str(Path(data_dir) / phase / b[0]),
                                             str(Path(data_dir) / phase / b[1]),
                                             str(Path(data_dir) / phase / b[2]),
                                             str(Path(data_dir) / phase / b[3])) for b in basenames]

    if not image_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return image_paths_by_split_dict

import copy
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional

import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from simlearner3d.utils import utils
import numpy as np

from simlearner3d.utils.utils import Data

from simlearner3d.processing.dataset.utils import (
    IMAGE_PATHS_BY_SPLIT_DICT_TYPE,
    SPLIT_TYPE,
    read_images_and_create_full_data_obj
)

log = utils.get_logger(__name__)

class HDF5Dataset(Dataset):
    """Single-file HDF5 dataset for collections of large LAS tiles."""

    def __init__(
        self,
        hdf5_file_path: str,
        image_paths_by_split_dict: IMAGE_PATHS_BY_SPLIT_DICT_TYPE,
        images_pre_transform: Callable = read_images_and_create_full_data_obj,
        tile_width: Number = 1024,
        tile_height: Number= 1024,
        patch_size: Number = 768,
        subtile_width: Number = 50,
        sign_disp_multiplier: Number = 1,
        masq_divider: Number = 1,
        subtile_overlap_train: Number = 0,
        train_transform: List[Callable] = None,
        eval_transform: List[Callable] = None,
    ):
        """Initialization, taking care of HDF5 dataset preparation if needed, and indexation of its content.

        Args:
        image_paths_by_split_dict ([IMAGE_PATHS_BY_SPLIT_DICT_TYPE]): should look like
                    image_paths_by_split_dict = {'train': [('dir/left.tif','dir/right.tif','dir/disp1.tif','dir/msq1.tif'),.....],
                    'test': [...]},
            hdf5_file_path (str): path to HDF5 dataset
            images_pre_transform (Callable): Function to turn images to Data Object.
            tile_width (Number, optional) : width of a IMAGE Defaults to 1024.
            tile_height (Number, optional): width of a IMAGE Defaults to 1024.
            patch_size: (Number, optional): width of a IMAGE patch for training. Defaults to 768.
            subtile_width (Number, optional): effective width of a subtile (i.e. receptive field). Defaults to 50.
            train_transform (List[Callable], optional): Transforms to apply to a sample for training. Defaults to None.
            eval_transform (List[Callable], optional): Transforms to apply to a sample for evaluation (test/val sets). Defaults to None.
        """

        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.tile_width = tile_width
        self.tile_height=tile_height
        self.patch_size=patch_size
        self.sign_disp_multiplier=sign_disp_multiplier
        self.masq_divider=masq_divider

        self.subtile_width = subtile_width
        self.subtile_overlap_train = subtile_overlap_train

        self.hdf5_file_path = hdf5_file_path

        # Instantiates these to null;
        # They are loaded within __getitem__ to support multi-processing training.
        self.dataset = None
        self._samples_hdf5_paths = None

        if not image_paths_by_split_dict:
            log.warning(
                "No image_paths_by_split_dict given, pre-computed HDF5 dataset is therefore used."
            )
            return
        # Add data for all IMAGE Files into a single hdf5 file.
        create_hdf5(
            image_paths_by_split_dict,
            hdf5_file_path,
            tile_width,
            tile_height,
            patch_size,
            subtile_width,
            subtile_overlap_train,
            images_pre_transform,
        )


        # Use property once to be sure that samples are all indexed into the hdf5 file.
        self.samples_hdf5_paths

    def __getitem__(self, idx: int) -> Optional[Data]:
        sample_hdf5_path = self.samples_hdf5_paths[idx]
        data = self._get_data(sample_hdf5_path)

        #----------------------------------------------------------------------------#
        transform = self.train_transform
        if sample_hdf5_path.startswith("val") or sample_hdf5_path.startswith("test"):
            transform = self.eval_transform
        if transform:
            data = transform(data)
        return data._left.unsqueeze(0), data._right.unsqueeze(0),data._disp.unsqueeze(0),data._masq.unsqueeze(0),data._xupl
        #----------------------------------------------------------------------------#


    def _get_data(self, sample_hdf5_path: str) -> Data:
        """Loads a Data object from the HDF5 dataset.

        Opening the file has a high cost so we do it only once and store the opened files as a singleton
        for each process within __get_item__ and not in __init__ to support for Multi-GPU.

        See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?u=piojanu.

        """
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5_file_path, "r")
        split,basename=osp.dirname(sample_hdf5_path),osp.basename(sample_hdf5_path)

        grp = self.dataset[split][basename]#[sample_hdf5_path]

        return Data(
            _left=torch.from_numpy(grp["l"][...]),
            _right=torch.from_numpy(grp["r"][...]),
            _disp=torch.from_numpy(grp["d"][...]).mul(self.sign_disp_multiplier), # do it once for all 
            _masq=torch.from_numpy(grp["m"][...]).div(self.masq_divider), # do it once for all
        )
    

    def __len__(self):
        return len(self.samples_hdf5_paths)

    @property
    def traindata(self):
        return self._get_split_subset("train")

    @property
    def valdata(self):
        return self._get_split_subset("val")

    @property
    def testdata(self):
        return self._get_split_subset("test")

    def _get_split_subset(self, split: SPLIT_TYPE):
        """Get a sub-dataset of a specific (train/val/test) split."""
        indices = [idx for idx, p in enumerate(self.samples_hdf5_paths) if p.startswith(split)]
        return torch.utils.data.Subset(self, indices)
    
    @property
    def samples_hdf5_paths(self):
        """Index all samples in the dataset, if not already done before."""
        # Use existing if already loaded as variable.
        if self._samples_hdf5_paths:
            return self._samples_hdf5_paths

        # Load as variable if already indexed in hdf5 file. Need to decode b-string.
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if "samples_hdf5_paths" in hdf5_file:
                self._samples_hdf5_paths = [
                    sample_path.decode("utf-8") for sample_path in hdf5_file["samples_hdf5_paths"]
                ]
                return self._samples_hdf5_paths

        # Otherwise, index samples, and add the index as an attribute to the HDF5 file.
        self._samples_hdf5_paths = []
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            for split in hdf5_file.keys():
                if split not in ["train", "val", "test"]:
                    continue
                for basename in hdf5_file[split].keys():
                    self._samples_hdf5_paths.append(osp.join(split, basename))

        with h5py.File(self.hdf5_file_path, "a") as hdf5_file:
            # special type to avoid silent string truncation in hdf5 datasets.
            variable_lenght_str_datatype = h5py.special_dtype(vlen=str)
            hdf5_file.create_dataset(
                "samples_hdf5_paths",
                (len(self.samples_hdf5_paths),),
                dtype=variable_lenght_str_datatype,
                data=self._samples_hdf5_paths,
            )
        return self._samples_hdf5_paths


def create_hdf5(
    image_paths_by_split_dict: dict,
    hdf5_file_path: str,
    tile_width:  Number = 1024,
    tile_height: Number = 1024,
    patch_size:  Number = 768,
    subtile_width: Number = 50,
    subtile_overlap_train: Number = 0,
    images_pre_transform: Callable = read_images_and_create_full_data_obj,
):
    """Create a HDF5 dataset file from left, right , disparities and masqs.

    Args:
    image_paths_by_split_dict ([IMAGE_PATHS_BY_SPLIT_DICT_TYPE]): should look like
                image_paths_by_split_dict = {'train': [('dir/left.tif','dir/right.tif','dir/disp1.tif','dir/msq1.tif'),.....],
                'test': [...]},
        hdf5_file_path (str): path to HDF5 dataset,
        tile_width: (Number, optional): width of an image tile. 1024 by default,
        tile_height: (Number, optional): height of an image tile. 1024 by default,
        patch_size: (Number, optional): considered subtile size for training 
        subtile_width: (Number, optional): effective width of a subtile (i.e. receptive field). 50 by default,
        pre_filter: Function to filter out specific subtiles. "pre_filter_below_n_points" by default,
        subtile_overlap_train (Number, optional): Overlap for data augmentation of train set. 0 by default,
        images_pre_transform (Callable): Function to load images and GT and create one Data Object.
    """
    os.makedirs(os.path.dirname(hdf5_file_path), exist_ok=True)
    for split, image_paths in image_paths_by_split_dict.items():
        print(split,image_paths)
        with h5py.File(hdf5_file_path, "a") as f:
            if split not in f:
                f.create_group(split)
        for image_gt_masq_set in tqdm(image_paths, desc=f"Preparing {split} set..."):
            basename_left  = os.path.basename(image_gt_masq_set[0]) # left image
            #basename_right = os.path.basename(image_gt_masq_set[1]) # right image
            #basename_disp  = os.path.basename(image_gt_masq_set[2]) # disparity
            #basename_masq  = os.path.basename(image_gt_masq_set[3]) # masq_nocc
            with h5py.File(hdf5_file_path, "a") as hdf5_file:
                if (
                    basename_left in hdf5_file[split]
                    and "is_complete" not in hdf5_file[split][basename_left].attrs
                ):
                    del hdf5_file[split][basename_left]
                    # Parse and add subtiles to split group.
            with h5py.File(hdf5_file_path, "a") as hdf5_file:
                if basename_left in hdf5_file[split]:
                    continue
                if not images_pre_transform:
                    continue
                data = images_pre_transform(image_gt_masq_set)
                print(data._left.shape)

                
                if basename_left not in hdf5_file[split]:
                     hdf5_file[split].create_group(basename_left)

                hdf5_file[split][basename_left].create_dataset(
                    "l",
                    data._left.shape,
                    dtype="f",
                    data=data._left,
                )
                hdf5_file[split][basename_left].create_dataset(
                    "r",
                    data._right.shape,
                    dtype="f",
                    data=data._right,
                )
                hdf5_file[split][basename_left].create_dataset(
                    "d",
                    data._disp.shape,
                    dtype="f",
                    data=data._disp,
                )
                hdf5_file[split][basename_left].create_dataset(
                    "m",
                    data._masq.shape,
                    dtype="f",
                    data=data._masq,
                )
                # A termination flag to report that all samples for this point cloud were included in the df5 file.
                # Group may not have been created if source cloud had no patch passing the pre_filter step, hence the "if" here.
                if basename_left in hdf5_file[split]:
                    hdf5_file[split][basename_left].attrs["is_complete"] = True

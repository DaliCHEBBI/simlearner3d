from numbers import Number
from typing import Callable, Dict, List, Optional

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

from simlearner3d.processing.transforms.compose import CustomCompose
from simlearner3d.processing.dataset.hdf5 import HDF5Dataset
from simlearner3d.processing.dataset.utils import (
    get_image_paths_by_split_dict
)
from simlearner3d.utils import utils
from simlearner3d.utils.utils import Data



log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class HDF5StereoDataModule(LightningDataModule):
    """Datamodule to feed train and validation data to the model."""

    def __init__(
        self,
        data_dir: str,
        split_csv_path: str,
        hdf5_file_path: str,
        tile_width: Number = 1024,
        tile_height: Number= 1024,
        patch_size: Number = 768,
        sign_disp_multiplier: Number = 1,
        masq_divider: Number = 1,
        subtile_overlap_train: Number = 0,
        subtile_overlap_predict: Number = 0,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
        **kwargs,
    ):
        super().__init__()

        self.split_csv_path = split_csv_path
        self.data_dir = data_dir
        self.hdf5_file_path = hdf5_file_path
        self._dataset = None  # will be set by self.dataset property
        self.image_paths_by_split_dict = {}  # Will be set from split_csv

        self.tile_width = tile_width
        self.tile_height=tile_height
        self.patch_size=patch_size
        self.sign_disp_multiplier=sign_disp_multiplier
        self.masq_divider=masq_divider

        self.subtile_overlap_train = subtile_overlap_train
        self.subtile_overlap_predict = subtile_overlap_predict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        t = transforms
        #self.preparation_train_transform: TRANSFORMS_LIST = t.get("preparations_train_list", [])
        #self.preparation_eval_transform: TRANSFORMS_LIST = t.get("preparations_eval_list", [])
        #self.preparation_predict_transform: TRANSFORMS_LIST = t.get(
        #    "preparations_predict_list", []
        #)
        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def train_transform(self) -> CustomCompose:
        return CustomCompose(
            self.normalization_transform
            + self.augmentation_transform
        )

    @property
    def eval_transform(self) -> CustomCompose:
        return CustomCompose(self.normalization_transform)

    @property
    def predict_transform(self) -> CustomCompose:
        return CustomCompose(self.normalization_transform)

    def prepare_data(self, stage: Optional[str] = None):
        """Prepare dataset containing train, val, test data."""

        if stage in ["fit", "test"] or stage is None:
            if self.split_csv_path and self.data_dir:
                image_paths_by_split_dict = get_image_paths_by_split_dict(
                    self.data_dir, self.split_csv_path
                )
            else:
                log.warning(
                    "cfg.data_dir and cfg.split_csv_path are both null. Precomputed HDF5 dataset is used."
                )
                image_paths_by_split_dict = None
        # Create the dataset in prepare_data, so that it is done one a single GPU.
        self.image_paths_by_split_dict = image_paths_by_split_dict
        self.dataset

    # TODO: not needed ?
    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate the (already prepared) dataset (called on all GPUs)."""
        self.dataset

    @property
    def dataset(self) -> HDF5Dataset:
        """Abstraction to ease HDF5 dataset instantiation.

        Args:
            image_paths_by_split_dict (IMAGE_PATHS_BY_SPLIT_DICT_TYPE, optional): Maps split (val/train/test) to file path.
                If specified, the hdf5 file is created at dataset initialization time.
                Otherwise,a precomputed HDF5 file is used directly without I/O to the HDF5 file.
                This is usefule for multi-GPU training, where data creation is performed in prepare_data method, and the dataset
                is then loaded again in each GPU in setup method.
                Defaults to None.

        Returns:
            HDF5Dataset: the dataset with train, val, and test data.

        """
        if self._dataset:
            return self._dataset

        self._dataset = HDF5Dataset(
            self.hdf5_file_path,
            self.image_paths_by_split_dict,
            tile_width=self.tile_width,
            tile_height=self.tile_height,
            patch_size=self.patch_size,
            sign_disp_multiplier=self.sign_disp_multiplier,
            masq_divider=self.masq_divider,
            subtile_overlap_train=self.subtile_overlap_train,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._dataset

    def train_dataloader(self):
        return DataLoader(self.dataset.traindata,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          drop_last=True,
                          )
    def val_dataloader(self):
        return DataLoader(self.dataset.valdata,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor,
                          drop_last=True,
                          )

    def test_dataloader(self):
        return DataLoader(self.dataset.testdata,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=1,
                          prefetch_factor=self.prefetch_factor,
                          )

    """def _set_predict_data(self, las_file_to_predict):
        self.predict_dataset = InferenceDataset(
            las_file_to_predict,
            self.epsg,
            points_pre_transform=self.points_pre_transform,
            pre_filter=self.pre_filter,
            transform=self.predict_transform,
            tile_width=self.tile_width,
            subtile_overlap=self.subtile_overlap_predict,
        )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.batch_size,
                          num_workers=1,
                          prefetch_factor=self.prefetch_factor,
                          )"""
    

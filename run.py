import os, sys
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from enum import Enum
import dotenv


from simlearner3d.utils import utils
from simlearner3d.processing.dataset.hdf5 import create_hdf5
from simlearner3d.processing.dataset.utils import get_image_paths_by_split_dict

class TASK_NAMES(Enum):
    FIT = "fit"
    TEST = "test"
    FINETUNE = "finetune"
    PREDICT = "predict"
    HDF5 = "create_hdf5"
    EXTRACT ="extract_pt"
    QUALIFY ="qualify"

DEFAULT_TASK = TASK_NAMES.FIT.value
TASK_NAME_DETECTION_STRING = "task.task_name="
DEFAULT_DIRECTORY = "./"
DEFAULT_CONFIG_FILE = ".yaml" # to change
DEFAULT_CHECKPOINT = ".ckpt" # to change
DEFAULT_ENV = ".env"

log = utils.get_logger(__name__)



@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_train(
    config: DictConfig,
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    """Training, evaluation, testing, or finetuning of a neural network."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from simlearner3d.train import train

    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)
    return train(config)


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_extract(
    config:DictConfig
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    """Training, evaluation, testing, or finetuning of a neural network."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from simlearner3d.extract import extract
    utils.extras(config)
     # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)
    return extract(config)   
    


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_hdf5(config: DictConfig):
    """Build an HDF5 file from a directory with pairs of images, gt disparities and masks."""

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    image_paths_by_split_dict = get_image_paths_by_split_dict(
        config.datamodule.get("data_dir"), config.datamodule.get("split_csv_path")
    )
    create_hdf5(
        image_paths_by_split_dict=image_paths_by_split_dict,
        hdf5_file_path=config.datamodule.get("hdf5_file_path"),
        tile_width=config.datamodule.get("tile_width"),
        tile_height=config.datamodule.get("tile_height"),
        patch_size=config.datamodule.get("patch_size"),
        subtile_width=config.datamodule.get("subtile_width"),
        subtile_overlap_train=config.datamodule.get("subtile_overlap_train"),
        images_pre_transform=hydra.utils.instantiate(
            config.datamodule.get("images_pre_transform")
        )
    )


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_qualify(
    config:DictConfig
):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    """Training, evaluation, testing, or finetuning of a neural network."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from simlearner3d.qualify import qualify
    utils.extras(config)
     # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)
    return qualify(config)   
    


if __name__=="__main__":
    task_name = "fit"
    for arg in sys.argv:
        if TASK_NAME_DETECTION_STRING in arg:
            _, task_name = arg.split("=")
            break

    log.info(f"Task: {task_name}")
    if task_name in [TASK_NAMES.FIT.value, TASK_NAMES.FINETUNE.value, TASK_NAMES.TEST.value]:
        # load environment variables from `.env` file if it exists
        # recursively searches for `.env` in all folders starting from work dir
        dotenv.load_dotenv(override=True)
        launch_train()

    elif task_name == TASK_NAMES.HDF5.value:
        dotenv.load_dotenv(os.path.join(DEFAULT_DIRECTORY, DEFAULT_ENV))
        launch_hdf5()

    elif task_name == TASK_NAMES.EXTRACT.value:
        dotenv.load_dotenv(override=True)
        launch_extract()

    elif task_name == TASK_NAMES.QUALIFY.value:
        dotenv.load_dotenv(override=True)
        launch_qualify()

    else:
        choices = ", ".join(task.value for task in TASK_NAMES)
        raise ValueError(
            f"Task '{task_name}' is not known. Specify a valid task name via task.task_name. Valid choices are: {choices})"
        )





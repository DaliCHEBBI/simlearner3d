
import logging
import time
import warnings
from typing import List, Sequence

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from typing import Callable, List, Optional

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.debug("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)

@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "task",
        "seed",
        "logger",
        "trainer",
        "model",
        "datamodule",
        "dataset_description",
        "callbacks",
        "predict",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.logging.Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def eval_time(method):
    """Decorator to log the duration of the decorated method"""

    def timed(*args, **kwargs):
        log = get_logger()
        time_start = time.time()
        result = method(*args, **kwargs)
        time_elapsed = round(time.time() - time_start, 2)

        log.info(f"Runtime of {method.__name__}: {time_elapsed}s")
        return result

    return timed


def define_device_from_config_param(gpus_param):
    """
    Param can be an in specifying a number of GPU to use (0 or 1) or an int in
    a list specifying which GPU to use (cuda:0, cuda:1, etc.)
    """
    device = (
        torch.device("cpu")
        if gpus_param == 0
        else (torch.device("cuda") if gpus_param == 1 else f"cuda:{int(gpus_param[0])}")
    )
    return device


class Data:
    def __init__(self,
                _left:Optional[torch.Tensor]  = None,
                _right:Optional[torch.Tensor] = None,
                _disp:Optional[torch.Tensor]  = None, 
                _masq:Optional[torch.Tensor]  = None,
                _xupl:Optional[int]=None,
                     ):
        if _left is not None:
            self._left=_left
        if _right is not None:
            self._right=_right
        if _disp is not None:
            self._disp=_disp
        if _masq is not None:
            self._masq=_masq
        if _xupl is not None:
            self._xupl=_xupl

    def __call__(self):
        return self
    



def get_grid_coords_2d(y, x, coord_dim=-1):
    y, x = torch.meshgrid(y, x)
    coords = torch.stack([x, y], dim=coord_dim)
    return coords

def get_grid_coords_3d(z, y, x, coord_dim=-1):
    z, y, x = torch.meshgrid(z, y, x)
    coords = torch.stack([x, y, z], dim=coord_dim)
    return coords

def signed_to_unsigned(array):
    return (array + 1) / 2


def unsigned_to_signed(array):
    return (array - 0.5) / 0.5

def pytorch_to_numpy(array, is_batch=True, flip=True):
    array = array.detach().cpu().numpy()

    if flip:
        source = 1 if is_batch else 0
        dest = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    return array

def numpy_to_pytorch(array, is_batch=False, flip=True):
    if flip:
        dest = 1 if is_batch else 0
        source = array.ndim - 1
        array = np.moveaxis(array, source, dest)

    array = torch.from_numpy(array)
    array = array.float()

    return array

def convert_to_int(array):
    array *= 255
    array[array > 255] = 255.0

    if type(array).__module__ == 'numpy':
        return array.astype(np.uint8)

    elif type(array).__module__ == 'torch':
        return array.byte()
    else:
        raise NotImplementedError


def convert_to_float(array):
    max_value = np.iinfo(array.dtype).max
    array[array > max_value] = max_value

    if type(array).__module__ == 'numpy':
        return array.astype(np.float32) / max_value

    elif type(array).__module__ == 'torch':
        return array.float() / max_value
    else:
        raise NotImplementedError

def metric_mse(output, target):
    return torch.nn.functional.mse_loss(output, target).mean().item()



def dict_to_keyvalue(params, prefix=''):
    hparams = {}

    for key, value in params.items():
        if isinstance(value, dict):
            if not prefix == '':
                new_prefix = '{}.{}'.format(prefix, key)
            else:
                new_prefix = key
            hparams.update(dict_to_keyvalue(value, prefix=new_prefix))
        else:
            if not prefix == '':
                key = '{}.{}'.format(prefix, key)
            hparams[key] = value

    return hparams

def dict_mean(dict_list):

    mean_dict = {}
    dict_item = dict_list[0]

    for key in dict_list[0].keys():
        if isinstance(dict_item[key], dict):
            for key2 in dict_item[key].keys():
                if not mean_dict.__contains__(key):
                    mean_dict[key] = {}
                mean_dict[key][key2] = sum(d[key][key2] for d in dict_list) / len(dict_list)
        else:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from utils.io import load_config_train
import utils.utils as utils
from pathlib import Path
from models.MSNETPl import MSNETWithDecisionNetwork_Dense_LM_N_2, MSNETWithDecisionNetwork_Dense_LM_SubPix
from datasets.CubeDataset import StereoTrAerialDatasetDenseN, StereoValAerialDatasetDenseN 
from utils.logger import Logger
import torch
import tifffile as tff
import numpy as np
import os 
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
def MemStatus(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))

"""def collate_func(batch):
    return torch.cat(batch,0)"""

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    print('IS CUDA AVAILABLE ',torch.cuda.is_available())
    torch.backends.cudnn.benchmark=True
    experiment_name = 'MSNET_DECISION_AERIAL_DENSE_NORM_SUBPIX'
    print("actual training path,   ",Path().cwd())
    root_dir = Path().cwd() / 'trained_models'
    param = load_config_train(root_dir, experiment_name,config_path=
                              "/work/scratch/data/alichem/Data/Code_train_Model/configs/MSNET_DECISION_AERIAL_DENSE_NORM_SUBPIX/config_default.yaml")
    param.version=2
    logger = Logger(param)
    paramConfig=utils.dict_to_keyvalue(param)
    #MSAFFMlpDenseSubPix=MSNETWithDecisionNetwork_Dense_LM_SubPix(32,0,0.5,4,0.0) #
    MSAFFMlpDenseSubPix=MSNETWithDecisionNetwork_Dense_LM_SubPix.load_from_checkpoint("/work/scratch/data/alichem/Data/Code_train_Model/trained_models/MSNET_DECISION_AERIAL_DENSE_NORM_SUBPIX/version_0/logs/epoch=5-step=23292.ckpt")
    """MSAFFMlpDense_Legacy=MSNETWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint('/home/ad/alichem/trained_models/MSNET_DECISION_AERIAL_DENSE_NORM/version_2/logs/epoch=5-step=11646.ckpt')
    MSAFFMlpDenseSubPix.feature.load_state_dict(MSAFFMlpDense_Legacy.feature.state_dict())
    MSAFFMlpDenseSubPix.decisionNet.load_state_dict(MSAFFMlpDense_Legacy.decisionNet.state_dict())"""
    # RESET THE F_ALSE EXAMPLES RANDOM SELECTION INTERVAL TO TIGHTER CONSTRAINTS 
    MSAFFMlpDenseSubPix=MSAFFMlpDenseSubPix.cuda()
    # Create Datasets 
    #0.4357159999015069], 'RGB_STD': 0.19518538611431166 # DUBLIN + ENSHEDE
    # DUBLIN STATS 
    #0.4353755468,0.19367880
    # ALL AERIAL STATS
    #0.4345832367221748], 'RGB_STD': 0.19487172550026804
    Train_dataset=StereoTrAerialDatasetDenseN(paramConfig['dataset.path_train'],0.434583236,0.1948717255,paramConfig['dataset.name'])
    #Train_dataset.patch_size=512
    MemStatus("training dataset")
    # The dataset declaration lacks more information on augmentation
    Val_dataset=StereoValAerialDatasetDenseN(paramConfig['dataset.path_val'],0.434583236,0.1948717255,paramConfig['dataset.name'])
    
    checkpointVal = ModelCheckpoint(
        dirpath=logger.log_files_dir,
        save_top_k=5,
        verbose=True,
        monitor='val_loss',
        mode='min'
        #prefix=param.experiment_name
    )
    checkpointTrain = ModelCheckpoint(
        dirpath=logger.log_files_dir,
        save_top_k=5,
        verbose=True,
        monitor='training_loss',
        mode='min'
    )
    profiler=SimpleProfiler()
    trainer = pl.Trainer(
        #profiler=profiler,
        enable_model_summary=True,
        logger=logger,
        max_epochs=param.train.epochs,
        callbacks=[checkpointTrain,checkpointVal, LearningRateMonitor("epoch")],
        check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
        val_check_interval=0.5,
        accumulate_grad_batches=param.train.accumulate_grad_batches,
        track_grad_norm=2,
        strategy='ddp',
        gpus=param.n_gpus,
        precision=16,
        #auto_lr_find=True,
    )
    #trainer.logger._log_graph=True
    pl.seed_everything(42)
    MemStatus("before training loader")
    train_loader=torch.utils.data.DataLoader(Train_dataset, batch_size=param.train.bs, shuffle=True,drop_last=True,pin_memory=True, num_workers=param.train.num_workers)
    MemStatus("after training loader")
    val_loader=torch.utils.data.DataLoader(Val_dataset, batch_size=param.val.bs, shuffle=False,drop_last=True,pin_memory=True, num_workers=param.val.num_workers)
    MemStatus("after validation loader")
    
    #trainer.tune(UnetMlpDense,train_loader,val_loader)
    
    trainer.fit(MSAFFMlpDenseSubPix, train_loader, val_loader)

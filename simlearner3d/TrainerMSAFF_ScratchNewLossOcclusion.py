import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from utils.io import load_config_train
import utils.utils as utils
from pathlib import Path
#from models.MulScaleFeatures import feature_extractor_dfc
#from models.UNetDecisionEBM import UNETWithDecisionNetwork_Dense_LM_N,UNETWithDecisionNetwork_Dense_LM_N_2
#from models.unet import UNETWithDecisionNetwork_LM
from models.MSNETPl import MSNETWithDecisionNetwork_Dense_LM_N_2
from datasets.CubeDataset import StereoTrAerialDatasetDenseN,StereoTrAerial4SatDatasetDenseN, StereoValAerialDatasetDenseN
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
    print('IS CUDA AVAILABLE ',torch.cuda.is_available())
    torch.backends.cudnn.benchmark=True
    experiment_name = 'MSAFF_MLP_NEWLOSS_OCCLUSION'
    print("actual training path,   ",Path().cwd())
    root_dir = Path().cwd() / 'trained_models'
    param = load_config_train(root_dir, experiment_name,config_path=
                              "/work/scratch/data/alichem/Data/Code_train_Model/configs/MSAFF_MLP_NEWLOSS_OCCLUSION/config_default.yaml")
    param.version=14
    logger = Logger(param)
    paramConfig=utils.dict_to_keyvalue(param)
    
    MsaffMlpDense=MSNETWithDecisionNetwork_Dense_LM_N_2(32,0,1,4,0.0) # true -1,0,1 false 2 --> 6
    MsaffMlpDense.learning_rate=0.001
    MsaffMlpDense=MsaffMlpDense.cuda()
    
    Train_dataset=StereoTrAerial4SatDatasetDenseN(paramConfig['dataset.path_train'],
                                                  0.434583236,
                                                  0.1948717255,
                                                  paramConfig['dataset.name'])
    Train_dataset.patch_size=640
    
    MemStatus("training dataset")
    # The dataset declaration lacks more information on augmentation
    Val_dataset=StereoValAerialDatasetDenseN(paramConfig['dataset.path_val'],
                                             0.434583236,
                                             0.1948717255,
                                             paramConfig['dataset.name'])
    
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
        enable_model_summary=True,
        logger=logger,
        max_epochs=param.train.epochs,
        callbacks=[checkpointTrain,checkpointVal, LearningRateMonitor("epoch")],
        check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
        val_check_interval=0.5,
        accumulate_grad_batches=param.train.accumulate_grad_batches,
        strategy='ddp',
        devices=param.n_gpus,
        precision="16-mixed",
    )
    #trainer.logger._log_graph=True
    pl.seed_everything(42)
    MemStatus("before training loader")
    train_loader=torch.utils.data.DataLoader(Train_dataset, 
                                             batch_size=param.train.bs, 
                                             shuffle=True,
                                             drop_last=True,
                                             pin_memory=True, 
                                             num_workers=param.train.num_workers)
    MemStatus("after training loader")
    val_loader=torch.utils.data.DataLoader(Val_dataset, 
                                           batch_size=param.val.bs, 
                                           shuffle=False,
                                           drop_last=True,
                                           pin_memory=True, 
                                           num_workers=param.val.num_workers)
    MemStatus("after validation loader")
    
    trainer.fit(MsaffMlpDense, 
                train_loader, 
                val_loader)

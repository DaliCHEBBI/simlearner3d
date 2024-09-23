import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from utils.io import load_config_train
import utils.utils as utils
from pathlib import Path
from models.MSNETPl import MSNETWithDecisionNetwork_2D_SubPix

from datasets.CubeDataset import DDeformDataset, DDeformDatasetVal

from utils.logger import Logger
import torch
import os


def MemStatus(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))



if __name__ == '__main__':
    
    #torch.set_float32_matmul_precision('high')
    print('IS CUDA AVAILABLE ?',torch.cuda.is_available())
    
    torch.backends.cudnn.benchmark=True
    experiment_name = 'MSAFF_SENTINEL_DEFORM_2D'
    
    print("Actual training path,   ",Path().cwd())
    
    
    root_dir = Path().cwd() / 'trained_models'
    param = load_config_train(root_dir, 
                              experiment_name,
                              config_path="/gpfs/scratch/rupnik/deepsim/DeformAnalysis/configs/MSAFF_SENTINEL_DEFORM_2D/config_default.yaml",
                              )
    
    param.version=1
    logger = Logger(param)
    paramConfig=utils.dict_to_keyvalue(param)
    
    
    MSAFFMlpDenseSubPix=MSNETWithDecisionNetwork_2D_SubPix(32,
                                                           0,
                                                           0.5,
                                                           2.0,
                                                           0.0
                                                           )
    MSAFFMlpDenseSubPix=MSAFFMlpDenseSubPix.cuda()
    
    
    Train_dataset=DDeformDataset(paramConfig['dataset.path_train'],
                                              0.434583236,
                                              0.1948717255,
                                              paramConfig['dataset.name'],
                                )


    Val_dataset=DDeformDatasetVal(paramConfig['dataset.path_val'],
                                             0.434583236,
                                             0.1948717255,
                                             paramConfig['dataset.name'],
                                )
    
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
        strategy='ddp',
        devices=param.n_gpus,
        precision="16-mixed",
        #auto_lr_find=True,
    )
    
    #trainer.logger._log_graph=True
    pl.seed_everything(42)
    
    MemStatus("Before training loader")
    train_loader=torch.utils.data.DataLoader(Train_dataset, batch_size=param.train.bs, 
                                             shuffle=True,drop_last=True,pin_memory=True, 
                                             num_workers=param.train.num_workers)
    MemStatus("After training loader")
    val_loader=torch.utils.data.DataLoader(Val_dataset, batch_size=param.val.bs, 
                                           shuffle=False,drop_last=True,pin_memory=True,
                                           num_workers=param.val.num_workers)
    MemStatus("After validation loader")
    
    trainer.fit(MSAFFMlpDenseSubPix, train_loader, val_loader)

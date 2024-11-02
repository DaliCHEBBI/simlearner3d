import copy
import os
import hydra
import argparse
from pathlib import Path
import typing
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc

from omegaconf import DictConfig
from pytorch_lightning import (
    LightningModule,
    LightningDataModule,
)
import torch
from torch import nn
from simlearner3d.models.generic_model import Model

from simlearner3d.models.modules.msaff import MSNet,MSNETInferenceGatedAttention
from simlearner3d.models.modules.unet import UNet,UNetInference
from simlearner3d.models.modules.unetgatedattention import UNetGatedAttention, UNetInferenceGatedAttention
from simlearner3d.models.modules.decision_net import DecisionNetworkOnCube
from simlearner3d.utils import utils
from tqdm import tqdm

log = utils.get_logger(__name__)

NEURAL_NET_ARCHITECTURE_CONFIG_GROUP = "neural_net"

MODEL_ZOO = [MSNet,UNet,UNetGatedAttention]
MODEL_INFERENCE_ZOO=[MSNETInferenceGatedAttention,UNetInference,UNetInferenceGatedAttention]

DEFAULT_MODE="feature"

def PlotJointDistribution(Simsplus,
                          Simsmoins,
                          model,
                          model_ckpt_path,
                          output_folder,
                          ):
    import pandas as pd
    import matplotlib.patches as  mpatches
    import matplotlib.cm as cm
    import seaborn as sns
    Simsplus=np.expand_dims(Simsplus, axis=1)
    Simsmoins=np.expand_dims(Simsmoins, axis=1)
    Simall=np.concatenate((Simsplus,Simsmoins),axis=1)
    print(Simall.shape)
    JDistrib=pd.DataFrame(Simall,columns=['+','-'])
    sns.color_palette("RdBu", 20)
    g = sns.jointplot(data = JDistrib,
                    x = "-",
                    y = "+",
                    xlim=(-1.0,1.0) if model.mode == DEFAULT_MODE else (0.0,1.0),
                    ylim=(-1.0,1.0) if model.mode == DEFAULT_MODE else (0.0,1.0),
                    cmap=cm.jet, 
                    kind="hist", 
                    marginal_kws={"color":"r", "alpha":.4, "bins":200, "stat":'percent'}, 
                    joint_kws={"bins":(200,200)},
                    stat='percent',
                    marginal_ticks=True,
                    #n_levels=50,
                    cbar=True,
                    #cbar_kws={"use_gridspec":False, "location":"top"},
                    label='Module KDE'
                    )
    plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    g.fig.axes[-1].set_position([.85, pos_joint_ax.y0, .07, pos_joint_ax.height])
    #facecolor = g.ax_marg_x.collections[0].get_facecolor()
    #g.ax_marg_x.lines[0].set_color('orange')
    g.ax_marg_y.spines['right'].set_visible(False)
    g.ax_marg_y.spines['left'].set_visible(True)
    g.ax_marg_y.spines['bottom'].set_visible(True)
    g.ax_marg_y.spines['top'].set_visible(False)
    
    #g.ax_marg_y.spines['right'].set_linewidth(0.5)
    g.ax_marg_y.spines['left'].set_linewidth(0.5)
    g.ax_marg_y.spines['bottom'].set_linewidth(0.5)
    #g.ax_marg_y.spines['top'].set_linewidth(0.5)
    g.ax_marg_y.tick_params(axis='both', which='major', labelsize=10)
    g.ax_marg_y.tick_params(axis='both', which='minor', labelsize=10)
    #g.ax_marg_y.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    g.set_axis_labels(xlabel='Non-matching similarity', ylabel='Matching Similarity', size=12)
    g.ax_joint.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_y.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_x.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_y.invert_yaxis()
    #g.ax_joint.xaxis.tick_top()
    values= g.ax_joint.collections[0].get_array()
    values=np.reshape(values, (200,200))
    SUM_GOOD=0.0
    for j in range(200):
        for i in range(j+1):
            if (values[j,i]!='--'):
                print(values)
                SUM_GOOD+=values[j,i]   
    pourcent=str("%.2f" % SUM_GOOD)+" %"       

    _NAME_OUT=model.feature.__class__.__name__
    if model.mode!=DEFAULT_MODE:
        _NAME_OUT+="_MLP"
    NAME_EPOCH=os.path.basename(model_ckpt_path)[:-5]
    handles = [mpatches.Patch(facecolor=plt.cm.jet(255), label='{} : {}'.format(_NAME_OUT,pourcent))]
    g.ax_joint.legend(handles=handles,loc=4)
    plt.savefig("{}/{}_{}".format(output_folder,
                                  _NAME_OUT,
                                 NAME_EPOCH)+".svg")


def ComputeAreabetweencurves(CurveSet1,CurveSet2):
    SurfaceIntersection=[]
    SurfaceUnion=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfaceIntersection.append(np.maximum(Surface2,Surface1)-np.abs(Surface2-Surface1))
         SurfaceUnion.append(np.maximum(Surface2,Surface1))
    # likelihood
    #IOU=[a/b for a,b in zip(SurfaceIntersection,SurfaceUnion)]
    return np.sum(SurfaceIntersection)/np.sum(SurfaceUnion)

def ComputeAreaRationPositiveNegative(CurveSet1,CurveSet2,BIN):
    SurfacePositive=[]
    SurfaceNegative=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfacePositive.append(Surface1)
         SurfaceNegative.append(Surface2)
    # likelihood
    # Binning Factor every n values compute ratio
    likelihoodRatio=[]
    steps=[]
    for i in range(0,len(SurfacePositive),BIN):
        # Compute ratio 
        ratio=sum(SurfacePositive[i:i+BIN])/float(sum(SurfaceNegative[i:i+BIN]))
        likelihoodRatio.append(ratio)
        # Save steps 
        steps.append(CurveSet1[i][0])
    steps.append(CurveSet1[-1][0])
    # Get steps at midle locations 
    return steps,likelihoodRatio

def testing_step_decision_DFC(batch,modulems,device,NANS=-999.0):
    true1=1
    false1=1
    false2=8
    x0,x1,dispnoc0=batch
    dispnoc0=dispnoc0.to(device)
    Mask0=(dispnoc0!=NANS).float().to(device)  # NAN=-999.0
    dispnoc0[dispnoc0==NANS]=0.0 # Set Nans to 0.0
    # Forward
    FeatsL=modulems.feature(x0.to(device)) 
    FeatsR=modulems.feature(x1.to(device))
    Offset_pos=(-2*true1) * torch.rand(dispnoc0.size(),device=device) + true1 #[-true1,true1]
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size(),device=device) + false2)
    RandSens=torch.rand(dispnoc0.size(),device=device)
    RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(device)
    Offset_neg=Offset_neg*RandSens
    #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
    D_pos=dispnoc0+Offset_pos
    D_neg=dispnoc0+Offset_neg
    Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
    Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 
    # Clean Indexes so there is no overhead 
    MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(device) 
    MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(device)
    # Cleaned Offp and Offn
    Offp=(Offp*MaskOffPositive).to(torch.int64)
    Offn=(Offn*MaskOffNegative).to(torch.int64)
    # Need to repeat interleave 
    Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
    Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
    # Get Examples positive and negative 
    FeatsR_plus=torch.gather(FeatsR,-1,Offp)
    # Test gather operator 
    FeatsR_minus=torch.gather(FeatsR,-1,Offn)
    # Mask Global = Mask des batiments + Mask des offsets bien definis 
    MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
    ref_pos=modulems.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
    ref_neg=modulems.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
    ref_pos=F.sigmoid(ref_pos)
    ref_neg=F.sigmoid(ref_neg)
    simplus=torch.masked_select(ref_pos, MaskGlob.bool())
    simmins=torch.masked_select(ref_neg, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()


def ComputeSurfacesIntersection(CurveSet1,CurveSet2):
    SurfaceIntersection=[]
    SurfaceUnion=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfaceIntersection.append(np.maximum(Surface2,Surface1)-np.abs(Surface2-Surface1))
         SurfaceUnion.append(np.maximum(Surface2,Surface1))
    # likelihood
    IOU=np.asarray(SurfaceIntersection)/np.sum(SurfaceUnion)
    IOU.tofile("./IOU_UNETATT.bin")

def ROCCurveAuc(Simplus,Simminus):
    from sklearn import metrics
    from scipy.special import kl_div
    print("KL DIVERGENCE UNETATT ==> ",np.sum(kl_div(Simplus,Simminus))+np.sum(kl_div(Simminus,Simplus)))
    AllSims=np.concatenate((Simplus,Simminus),axis=0)
    Labels=np.concatenate((np.ones(len(Simplus)),np.zeros(len(Simminus))))
    fpr,tpr,_=metrics.roc_curve(Labels,AllSims)
    AUC=metrics.roc_auc_score(Labels,AllSims)
    # RETURN ALL RESULTS 
    """print("AREA UNDER CURVE  ==>  ",AUC)
    print("FALSE POSITVE RATE ==> ",fpr)
    print("TRUE POSITVE RATE  ==> ",tpr)"""
    fpr.tofile('./FPR_UNETATT.bin')
    print("SHAPE OF TPR UNETATT  ====>  ",tpr.shape)
    tpr.tofile('./TPR_UNETATT.bin')
    print("SHAPE OF FPR UNETATT  ====>  ",fpr.shape)
    print("AUC UNETATT     =====>  ",AUC)


def testing_step_dense(batch,
                       modulems,
                       device, 
                       nans=0.0, 
                       false1=2, 
                       false2=40
                       ):
        
    x0,x1,dispnoc0,Mask0,x_offset=batch

    MaskDef=(dispnoc0!=nans)
    
    FeatsL=modulems.feature(x0.to(device)) 
    
    FeatsR=modulems.feature(x1.to(device))
    
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size()) + false2)
    
    RandSens=torch.rand(dispnoc0.size())
    
    RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0))
    
    Offset_neg=Offset_neg*RandSens
    
    #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
    D_pos=dispnoc0
    D_neg=dispnoc0+Offset_neg
    
    Index_X=torch.arange(0,dispnoc0.size()[-1])
    Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)

    Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
    
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 

    MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float()
    MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float()

    Offp=(Offp*MaskOffPositive).to(torch.int64)
    Offn=(Offn*MaskOffNegative).to(torch.int64)

    Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
    Offn=Offn.repeat_interleave(FeatsR.size()[1],1)

    FeatsR_plus=torch.gather(FeatsR,-1,Offp)

    FeatsR_minus=torch.gather(FeatsR,-1,Offn)

    MaskGlob=Mask0*MaskDef*MaskOffPositive

    if modulems.mode==DEFAULT_MODE:
        simplus=F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1)
        simmins=F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1)
        simplus=torch.masked_select(simplus, MaskGlob.bool())
        simmins=torch.masked_select(simmins, MaskGlob.bool())
    else:
        ref_pos=modulems.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=modulems.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        ref_pos=F.sigmoid(ref_pos)
        ref_neg=F.sigmoid(ref_neg)
        simplus=torch.masked_select(ref_pos, MaskGlob.bool())
        simmins=torch.masked_select(ref_neg, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()



def run_stats(net, test_loader, device,nans=0.0):
    torch.no_grad()
    net.eval()
    Simsplus=[]
    Simsmoins=[]
    for _, batch in tqdm(enumerate(test_loader, 0),
                         total= len(test_loader), 
                         desc="Running qualification"):
        simP,simN=testing_step_dense(batch,
                                     net,
                                     device,
                                     nans)
        print(" min max simP ", torch.min(simP), "  ", torch.max(simP))
        print(" min max simN ", torch.min(simN), "  ", torch.max(simN))
        gc.collect()
        print("Sizes of matching similarity tile ",simP.shape)
        # compute loss
        Simsplus.append(simP.numpy())
        Simsmoins.append(simN.numpy())
    return Simsplus,Simsmoins

def _intersection_curve(Simsplus,
                        Simsmoins,
                        model,
                        model_ckpt_path,
                        output_folder):
    
    fig1=plt.figure()

    Histplus,Histplusbins=np.histogram(Simsplus,
                                       bins=200,
                                       density=True)

    Histmoins,Histmoinsbins=np.histogram(Simsmoins,
                                         bins=200,
                                         density=True)

    plt.rcParams.update({'font.size': 10})
    plt.plot(Histplusbins[1:],
             Histplus, 
             label='Prob. mass of positive samples'
             )
    plt.plot(Histmoinsbins[1:],
             Histmoins, 
             label='Prob. mass of negative samples'
             )
    PourcentageIntersection=ComputeAreabetweencurves(list(tuple(zip(Histplusbins, Histplus))),
                                                     list(tuple(zip(Histmoinsbins, Histmoins))))
    
    #ROCCurveAuc(Simsplus,Simsmoins)

    plt.fill_between(Histplusbins[1:], Histplus, step="pre", alpha=0.2)
    plt.fill_between(Histmoinsbins[1:], Histmoins, step="pre", alpha=0.2)
    Histplus=np.cumsum(Histplus*np.diff(Histplusbins))
    Histmoins=np.cumsum(Histmoins*np.diff(Histmoinsbins))

    plt.vlines(x=np.mean(Simsplus),
                ymin=0, 
                ymax=3.0, 
                colors='blue', 
                ls=':', 
                lw=2, 
                label='mean+ ='+"{:.2f}".format(np.mean(Simsplus))
                )
    plt.vlines(x=np.mean(Simsmoins),
                ymin=0, 
                ymax=3.0, 
                colors='orange', 
                ls=':', 
                lw=2, 
                label='mean- ='+"{:.2f}".format(np.mean(Simsmoins))
                )
    plt.xlabel("Similarity values")
    plt.ylabel("Count Number (%)")

    _NAME_OUT=model.feature.__class__.__name__
    if model.mode!=DEFAULT_MODE:
        _NAME_OUT+="_MLP"
    NAME_EPOCH=os.path.basename(model_ckpt_path)[:-5]
    plt.title(_NAME_OUT+"_{:.2f}%".format(PourcentageIntersection*100))
    #legend_properties = {'weight':'bold'}
    plt.legend(fontsize=10)
    plt.savefig("{}/{}_{}_Surf_{:.2f}%".format(output_folder,
                                            _NAME_OUT,
                                            NAME_EPOCH,
                                            PourcentageIntersection*100)+".png")


def qualify(config: DictConfig):
    
    """
    Computes :
      - joint probability distributions of matching and non matching pixels after training 

    """
    # Set seed for random number generators in pytorch, numpy and python.random
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)


    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info("Extracting models architectures for inference!")
    # Instantiates the Model but overwrites everything with current config,
    # except module related params (nnet architecture)
    kwargs_to_override = copy.deepcopy(model.hparams)
    kwargs_to_override.pop(
        NEURAL_NET_ARCHITECTURE_CONFIG_GROUP, None
    )  # removes that key if it's there
    model = Model.load_from_checkpoint(config.model.ckpt_path, **kwargs_to_override)

    # Data loader for testing 
    test_dataloader=datamodule.test_dataloader()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Simsplus, Simsmoins=run_stats(model,test_dataloader,device,nans=0.0)
    Simsplus=np.concatenate(Simsplus, axis=0 )
    Simsmoins=np.concatenate(Simsmoins, axis=0 )

    # intersection curve 
    _intersection_curve(Simsplus,
                        Simsmoins,
                        model,
                        config.model.ckpt_path,
                        config.report.output_folder)
    
    # joint probabilities

    PlotJointDistribution(Simsplus,
                          Simsmoins,
                          model,
                          config.model.ckpt_path,
                          config.report.output_folder,
                          )







    
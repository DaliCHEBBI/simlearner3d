# Performing inference on new data

Refer to the tutorials ([Linux](./install_on_linux.md), [Windows](./install_on_wsl2.md)) for installation instructions.

To run inference, you will need:

- A checkpoint of a trained lightning module implementing model logic (class `simlearner3d.models.generic_model.Model`)
- A pair of epipolar images
- An installed and working version of [our micmac repository](https://github.com/DaliCHEBBI/micmac_SimLearning).

## Transforming .ckpt model to scripted .pt models

Our [our micmac version](https://github.com/DaliCHEBBI/micmac_SimLearning) leverages learned models together with [PyTorch c++ api](https://pytorch.org/get-started/locally/) to perform large scale stereo image matching. 
In order to use trained models, you need to script the model checkpoint from pytorchlightning to a .pt model. This is performed using the following command line:


```bash
python run.py \
task.task_name=extract_pt \
model.ckpt_path={/path/to/checkpoint.ckpt}
```

## Run inference on a pair of epipolar images

Inference routines are implemented in C++ into the open photogrammetry software [MicMac](https://github.com/micmacIGN/micmac). 

It tackles large scale stereo matching for large frame aerial imagery and EHR Satellite stereo reconstruction. Our models are embedded into a hierachical multi-resolution pipeline where lower resolution surface reconstructions serve as a prior for the next level reconstruction. This strategy is robust and allows integrating hand-crafted similarity metrics like (ncc, census,...) for lower resolution low context matching. 

To perform hierarchical matching using our similarity learning models, you need to: 

* git clone our repository and install `micmac` and `MicMac V2` after downloading plateform specific [torch cpp libraries](). You may need to modifiy `TORCH_INSTALL_PREFIX` in CMakeLists.txt to point to  `libtorch` folder. For our installation, we set it to `${MMVII_SOURCE_DIR}/libtorch`
* compile MMVII and micmac with libtorch 
* MicMac uses `xml specifications` files for hierachical dense image matching, we provide an example xml configuration file:
`
<ParamMICMAC>

<DicoLoc>
       <Symb> ZReg=0.005  </Symb>
       <Symb> DefCor=0.4  </Symb>
       <Symb> CostTrans=1.0 </Symb>

       <Symb> SzW=3 </Symb>
       <Symb> PenteMax=3.0 </Symb>
       <Symb> Interpol=eInterpolBiLin  </Symb>
       <Symb> DoEpi=true </Symb>
       <Symb> Ori=Epi    </Symb>

       <Symb> Im=TEST   </Symb>

       <Symb> Im1=Im1DUBL.tif </Symb>
       <Symb> Im2=Im2DUBL.tif </Symb>

       <Symb> Modele=XXX</Symb>
       <Symb> WITH_MODELE=true </Symb>
       <Symb> IncPix=100  </Symb>
       <Symb> NbDirProg=7   </Symb>
       <Symb> ExtImIn=tif   </Symb>
       <Symb> Purge=true   </Symb>
       <Symb> NbProc=4   </Symb>
       <Symb> OnCuda=false</Symb>
       <Symb> UseMLP=false</Symb>
  <!-- Parametres calcule -->
        
        <Symb>  DirMEC=${Modele} </Symb>
        <Symb>  DirPyram=Pyram/ </Symb>

</DicoLoc>

<Section_Terrain> 
      <IntervParalaxe>
            <Px1IncCalc> ${IncPix} </Px1IncCalc>
      </IntervParalaxe>
      
</Section_Terrain>

<Section_PriseDeVue>

   <GeomImages> eGeomImage_EpipolairePure </GeomImages> 

   <Images>
    	<Im1>   ${Im1} </Im1>
    	<Im2>   ${Im2}    </Im2>
   </Images>

   <MasqImageIn>
             <OneMasqueImage>
                <PatternSel>  (.*)\.${ExtImIn}  </PatternSel>
                <NomMasq>  $1_Masq.tif     </NomMasq>
             </OneMasqueImage>
             <AcceptNonExistingFile> true </AcceptNonExistingFile>
   </MasqImageIn>

</Section_PriseDeVue>


<!--  *************************************************************
       Parametres fixant le comportement
     de l'algo de mise en correspondance
-->
<Section_MEC>
	<ChantierFullImage1> true </ChantierFullImage1>

	<EtapeMEC>
    	    <DeZoom> -1 </DeZoom>
    	    <ModeInterpolation> ${Interpol} </ModeInterpolation>
		
	    <!-- param correl -->
	    <SzW> ${SzW} </SzW>

            <AlgoRegul> eAlgo2PrgDyn </AlgoRegul>

	    <Px1Pas>        1  </Px1Pas>
            <Px1DilatAlti>  7  </Px1DilatAlti>
	    <Px1DilatPlani> 3  </Px1DilatPlani>
	    <Px1Regul> ${ZReg} </Px1Regul>

            <GenImagesCorrel> true </GenImagesCorrel>
	    
            <SsResolOptim> 1 </SsResolOptim>
            <CoxRoyUChar> false </CoxRoyUChar>


	    <ModulationProgDyn>
               <EtapeProgDyn>
            	   <ModeAgreg> ePrgDAgrSomme </ModeAgreg>
                   <NbDir> ${NbDirProg} </NbDir>
               </EtapeProgDyn>
               <Px1PenteMax> ${PenteMax} </Px1PenteMax>
               <ArgMaskAuto>
            	   <ValDefCorrel> ${DefCor} </ValDefCorrel>
		   <CostTrans> ${CostTrans} </CostTrans>
                   <ReInjectMask> false </ReInjectMask> 
               </ArgMaskAuto>
            </ModulationProgDyn>


    </EtapeMEC>

	<!--  <EtapeMEC> <DeZoom> 16 </DeZoom> </EtapeMEC>	 -->
    <!-- <EtapeMEC> <DeZoom> 64 </DeZoom> </EtapeMEC> 
    <EtapeMEC> <DeZoom> 32 </DeZoom> </EtapeMEC>      -->
    <EtapeMEC> <DeZoom> 16 </DeZoom> </EtapeMEC>
    
    
	<EtapeMEC> <DeZoom> 8  </DeZoom> 
            <CorrelAdHoc>
                <SzBlocAH> 40000000 </SzBlocAH>
                <TypeCAH>
                    <ScoreLearnedMMVII >
                        <FileModeleCost> MVCNNCorrel</FileModeleCost>
                        <FileModeleParams>../MODEL_MSAFF_AERIAL_DECISION/.*.pt</FileModeleParams>
                        <FileModeleArch>UnetMLPMatcher</FileModeleArch>
                        <Cuda> ${OnCuda} </Cuda>
			<UsePredicNet> ${UseMLP} </UsePredicNet>
                    </ScoreLearnedMMVII>
                </TypeCAH>
            </CorrelAdHoc>
       </EtapeMEC>
  
	<EtapeMEC> <DeZoom> 4  </DeZoom> 
            <CorrelAdHoc>
                <SzBlocAH> 40000000 </SzBlocAH>
                <TypeCAH>
                    <ScoreLearnedMMVII >
                        <FileModeleCost> MVCNNCorrel</FileModeleCost>
                        <FileModeleParams>../MODEL_MSAFF_AERIAL_DECISION/.*.pt</FileModeleParams>
                        <FileModeleArch>UnetMLPMatcher</FileModeleArch>
                        <Cuda> ${OnCuda} </Cuda>
			<UsePredicNet> ${UseMLP} </UsePredicNet>
                    </ScoreLearnedMMVII>
                </TypeCAH>
            </CorrelAdHoc>
       </EtapeMEC>
	<EtapeMEC> 
        <DeZoom> 2  </DeZoom> 
            <CorrelAdHoc>
                <SzBlocAH> 40000000 </SzBlocAH>
                <TypeCAH>
                    <ScoreLearnedMMVII >
                        <FileModeleCost> MVCNNCorrel</FileModeleCost>
                        <FileModeleParams>../MODEL_MSAFF_AERIAL_DECISION/.*.pt</FileModeleParams>
                        <FileModeleArch>UnetMLPMatcher</FileModeleArch>
                        <Cuda> ${OnCuda} </Cuda>
			<UsePredicNet> ${UseMLP} </UsePredicNet>
                    </ScoreLearnedMMVII>
                </TypeCAH>
            </CorrelAdHoc>
    </EtapeMEC>	
    <EtapeMEC> 
        <DeZoom> 1  </DeZoom> 
            <CorrelAdHoc>
                <SzBlocAH> 40000000 </SzBlocAH>
                <TypeCAH>
                    <ScoreLearnedMMVII >
                        <FileModeleCost> MVCNNCorrel</FileModeleCost>
                        <FileModeleParams>../MODEL_MSAFF_AERIAL_DECISION/.*.pt</FileModeleParams>
                        <FileModeleArch>UnetMLPMatcher</FileModeleArch>
                        <Cuda> ${OnCuda} </Cuda>
			<UsePredicNet> ${UseMLP} </UsePredicNet>
                    </ScoreLearnedMMVII>
                </TypeCAH>
            </CorrelAdHoc>
        
    </EtapeMEC>
    
   <!-- <EtapeMEC>
        <DeZoom > 1 </DeZoom>
        <Px1Pas>   0.5  </Px1Pas>
    </EtapeMEC> -->

    <EtapeMEC>
            <DeZoom>  1  </DeZoom>
            <Px1Pas>   1.0     </Px1Pas>
            <AlgoRegul> eAlgoDequant </AlgoRegul>
	</EtapeMEC> 
        
	<HighPrecPyrIm> false </HighPrecPyrIm>
    
	<TypePyramImage>
               <Resol >    1          </Resol>
               <DivIm>    1 </DivIm>
               <TypeEl>  eFloat32Bits   </TypeEl>
        </TypePyramImage>

</Section_MEC>

<!--  *************************************************************
       Parametres fixant les resultats
     devant etre produits par l'algo
-->
<Section_Results>
    <GeomMNT> eGeomPxBiDim     </GeomMNT>
    <ZoomMakeTA> 16 </ZoomMakeTA>
    <GammaVisu> 2.0 </GammaVisu>
    <ZoomVisuLiaison> -1 </ZoomVisuLiaison>
    
</Section_Results>

<!--  *************************************************************
       Parametres lies a la gestions
     du "chantier" sur la machine
-->
<Section_WorkSpace>

    <WorkDir> ThisDir </WorkDir> 
    <TmpMEC> ${DirMEC}/ </TmpMEC>
    <TmpResult> ${DirMEC}/ </TmpResult>
    <TmpPyr> ${DirPyram} </TmpPyr>
    <PurgeMECResultBefore>  ${Purge} </PurgeMECResultBefore>

    <ByProcess>  ${NbProc} </ByProcess>

    <AvalaibleMemory> 2048 </AvalaibleMemory>
    <SzDalleMin> 540 </SzDalleMin>
    <SzDalleMax> 512 </SzDalleMax>
    <SzRecouvrtDalles> 20 </SzRecouvrtDalles>
    <SzMinDecomposCalc> 40 </SzMinDecomposCalc>
    <ComprMasque> eComprTiff_None </ComprMasque>


</Section_WorkSpace>

<Section_Vrac> 
     <DebugMM> true</DebugMM>
</Section_Vrac>

</ParamMICMAC>
`
* The aformentionned xml file activates our models from `Zoom 4`, i.e models are in charge of reconstruction for zooms: 4,2,1.
This is accomplished with and adhoc correlation that calls a set of .pt models (feature extraction + mlp).
`
            <CorrelAdHoc>
                <SzBlocAH> 40000000 </SzBlocAH>
                <TypeCAH>
                    <ScoreLearnedMMVII >
                        <FileModeleCost> MVCNNCorrel</FileModeleCost>
                        <FileModeleParams>../MODEL_MSAFF_AERIAL_DECISION/.*.pt</FileModeleParams>
                        <FileModeleArch>UnetMLPMatcher</FileModeleArch>
                        <Cuda> ${OnCuda} </Cuda>
			            <UsePredicNet> ${UseMLP} </UsePredicNet>
                    </ScoreLearnedMMVII>
                </TypeCAH>
            </CorrelAdHoc>
`
* Parametrization: To launch stereo matching between a pair of epipolar images,

`mm3d MICMAC File.xml +Im1=Im1.tif +Im2=Im2.tif  +DirMEC=MEC/ +OnCuda=0 +UsePredicNet=1`




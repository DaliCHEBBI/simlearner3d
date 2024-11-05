Search.setIndex({docnames:["apidoc/configs","apidoc/scripts","apidoc/simlearner3d.callbacks","apidoc/simlearner3d.model","apidoc/simlearner3d.models.modules","apidoc/simlearner3d.processing","apidoc/simlearner3d.utils","background/general_design","background/interpolation","guides/development","guides/train_new_model","index","introduction","tutorials/install_on_linux","tutorials/install_on_wsl2","tutorials/make_predictions","tutorials/prepare_dataset"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["apidoc/configs.rst","apidoc/scripts.rst","apidoc/simlearner3d.callbacks.rst","apidoc/simlearner3d.model.rst","apidoc/simlearner3d.models.modules.rst","apidoc/simlearner3d.processing.rst","apidoc/simlearner3d.utils.rst","background/general_design.md","background/interpolation.md","guides/development.md","guides/train_new_model.md","index.rst","introduction.md","tutorials/install_on_linux.md","tutorials/install_on_wsl2.md","tutorials/make_predictions.md","tutorials/prepare_dataset.md"],objects:{"":[[1,0,0,"-","run"]],"run.TASK_NAMES":[[1,2,1,"","EXTRACT"],[1,2,1,"","FINETUNE"],[1,2,1,"","FIT"],[1,2,1,"","HDF5"],[1,2,1,"","PREDICT"],[1,2,1,"","QUALIFY"],[1,2,1,"","TEST"]],"simlearner3d.callbacks":[[2,0,0,"-","comet_callbacks"],[2,0,0,"-","finetuning_callbacks"]],"simlearner3d.callbacks.comet_callbacks":[[2,1,1,"","LogCode"],[2,1,1,"","LogLogsPath"],[2,3,1,"","get_comet_logger"],[2,3,1,"","log_comet_cm"]],"simlearner3d.callbacks.comet_callbacks.LogCode":[[2,4,1,"","on_train_start"]],"simlearner3d.callbacks.comet_callbacks.LogLogsPath":[[2,4,1,"","setup"]],"simlearner3d.callbacks.finetuning_callbacks":[[2,1,1,"","FinetuningFreezeUnfreeze"]],"simlearner3d.callbacks.finetuning_callbacks.FinetuningFreezeUnfreeze":[[2,4,1,"","finetune_function"],[2,4,1,"","freeze_before_training"]],"simlearner3d.models":[[3,0,0,"-","generic_model"]],"simlearner3d.models.generic_model":[[3,1,1,"","Model"],[3,3,1,"","get_neural_net_class"]],"simlearner3d.models.generic_model.Model":[[3,4,1,"","configure_optimizers"],[3,4,1,"","forward"],[3,4,1,"","test_step"],[3,4,1,"","training_step"],[3,4,1,"","validation_step"]],"simlearner3d.models.generic_model.Model.forward.params":[[3,5,1,"","**kwargs"],[3,5,1,"","*args"]],"simlearner3d.models.generic_model.Model.test_step.params":[[3,5,1,"","batch"],[3,5,1,"","batch_idx"],[3,5,1,"","dataloader_idx"]],"simlearner3d.models.generic_model.Model.training_step.params":[[3,5,1,"","batch"],[3,5,1,"","batch_idx"],[3,5,1,"","dataloader_idx"]],"simlearner3d.models.generic_model.Model.validation_step.params":[[3,5,1,"","batch"],[3,5,1,"","batch_idx"],[3,5,1,"","dataloader_idx"]],"simlearner3d.models.generic_model.get_neural_net_class.params":[[3,5,1,"","class_name"]],"simlearner3d.models.modules.msaff":[[4,1,1,"","MSNet"]],"simlearner3d.models.modules.msaff.MSNet":[[4,4,1,"","forward"]],"simlearner3d.processing.datamodule":[[5,0,0,"-","hdf5"]],"simlearner3d.processing.datamodule.hdf5":[[5,1,1,"","HDF5StereoDataModule"]],"simlearner3d.processing.datamodule.hdf5.HDF5StereoDataModule":[[5,6,1,"","dataset"],[5,4,1,"","prepare_data"],[5,4,1,"","setup"],[5,4,1,"","test_dataloader"],[5,4,1,"","train_dataloader"],[5,4,1,"","val_dataloader"]],"simlearner3d.processing.dataset":[[5,0,0,"-","hdf5"],[5,0,0,"-","toy_dataset"],[5,0,0,"-","utils"]],"simlearner3d.processing.dataset.hdf5":[[5,1,1,"","HDF5Dataset"],[5,3,1,"","create_hdf5"]],"simlearner3d.processing.dataset.hdf5.HDF5Dataset":[[5,6,1,"","samples_hdf5_paths"]],"simlearner3d.processing.dataset.toy_dataset":[[5,1,1,"","TASK_NAMES"],[5,3,1,"","make_toy_dataset_from_test_file"]],"simlearner3d.processing.dataset.toy_dataset.make_toy_dataset_from_test_file.params":[[5,5,1,"","`split`"],[5,5,1,"","files"],[5,5,1,"","prepared_data_dir"],[5,5,1,"","split_csv"],[5,5,1,"","src_las_path"]],"simlearner3d.processing.dataset.utils":[[5,3,1,"","find_file_in_dir"]],"simlearner3d.processing.dataset.utils.find_file_in_dir.params":[[5,5,1,"","input_data_dir"]],"simlearner3d.processing.transforms":[[5,0,0,"-","augmentations"],[5,0,0,"-","compose"],[5,0,0,"-","transforms"]],"simlearner3d.processing.transforms.compose":[[5,1,1,"","CustomCompose"]],"simlearner3d.processing.transforms.compose.CustomCompose.params":[[5,5,1,"","transforms"]],"simlearner3d.processing.transforms.transforms":[[5,1,1,"","StandardizeIntensity"],[5,1,1,"","StandardizeIntensityCenterOnZero"],[5,1,1,"","ToTensor"]],"simlearner3d.processing.transforms.transforms.StandardizeIntensity":[[5,4,1,"","standardize_channel"]],"simlearner3d.train":[[1,3,1,"","train"]],"simlearner3d.train.train.params":[[1,5,1,"","config"]],"simlearner3d.utils":[[6,0,0,"-","utils"]],"simlearner3d.utils.utils":[[6,3,1,"","define_device_from_config_param"],[6,3,1,"","eval_time"],[6,3,1,"","extras"],[6,3,1,"","get_logger"],[6,3,1,"","log_hyperparameters"],[6,3,1,"","print_config"]],"simlearner3d.utils.utils.extras.params":[[6,5,1,"","config"]],"simlearner3d.utils.utils.print_config.params":[[6,5,1,"","config"],[6,5,1,"","fields"],[6,5,1,"","resolve"]],"simlearner3d.utils.utils.print_config.params.printed and in what order":[[6,5,1,"",""]],run:[[1,1,1,"","TASK_NAMES"],[1,3,1,"","launch_extract"],[1,3,1,"","launch_hdf5"],[1,3,1,"","launch_qualify"],[1,3,1,"","launch_train"]],simlearner3d:[[2,0,0,"-","callbacks"],[1,0,0,"-","train"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","method","Python method"],"5":["py","parameter","Python parameter"],"6":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:function","4":"py:method","5":"py:parameter","6":"py:property"},terms:{"0":[0,1,3,5,6,7,10,15],"001":0,"003933709606504788":0,"005":15,"03d":0,"04":14,"04_29_0005_im1":16,"04_29_0005_im2":16,"05_19_0017_im1":16,"05_19_0017_im2":16,"05_20_0003_im1":16,"05_20_0003_im2":16,"05_20_0006_im1":16,"05_20_0006_im2":16,"05_21_0004_im1":16,"05_21_0004_im2":16,"05_21_0014_im1":16,"05_21_0014_im2":16,"06_20_0001_im1":16,"06_20_0001_im2":16,"07_18_0011_im1":16,"07_18_0011_im2":16,"1":[0,3,5,6,7,13,15],"10":14,"100":15,"1024":[0,5],"11":[13,14],"12":[5,13],"12345":0,"123_456":5,"12500":7,"16":[0,15],"19041":14,"1_masq":15,"2":[0,3,5,7,14,15],"20":[0,15],"2004":14,"2016":7,"2019":7,"2020":7,"2022":14,"2023mmdd":9,"2048":15,"20_30_0002_im1":16,"20_30_0002_im2":16,"22":14,"25m":7,"3":[0,2,4,5,7,10,13,14,15],"30":[10,14],"30000":7,"32":[0,7,15],"3d":7,"4":[0,7,14,15],"40":15,"40000000":15,"5":[0,15],"50":[0,5],"50mx50m":7,"512":15,"540":15,"6":[0,2,3,14],"64":15,"7":15,"768":[0,5],"8":15,"9":[0,2],"abstract":5,"case":[1,3,9,14],"class":[1,2,3,4,5,7,15],"default":[3,5,11,14,16],"do":[3,5,7,10,16],"enum":1,"final":13,"function":[1,4,5],"import":14,"int":[2,3,5,6],"new":[1,9,11,16],"null":0,"public":13,"return":[1,2,3,5],"short":7,"super":3,"switch":3,"true":[0,6,10,15],"try":[10,14],"while":4,A:[3,6,9,15,16],As:[7,14],At:[3,14],For:[3,5,7,13,15],If:[2,3,5,7,10,13,14,16],In:[1,3,7,9,14,15],It:[5,7,11,12,14,15,16],Its:[11,12],On:7,Or:13,TO:14,That:14,The:[3,5,7,9,10,11,12,14,15,16],Then:[13,14],There:[5,7],To:[3,10,13,14,15,16],__init__:3,_args_:0,_sphinx_paramlinks_simlearner3d:[5,6],_target_:0,_version:9,about:5,abov:5,acc:3,acceler:[0,13],accept:[7,9,14],access:[6,9,14],accomplish:15,account:[7,10],accumulate_grad_batch:[0,3],accuraci:3,activ:[9,10,13,14,15],actual:[11,12],ad:9,adam:0,adapt:5,add:[5,14],add_imag:3,addit:[3,16],addition:16,additionali:6,additionnali:9,adhoc:15,admin:14,administr:14,adopt:7,aerial:[7,15],aforment:15,after:[10,14,15,16],afterward:4,again:[5,14],agre:14,agreement:14,ahead:14,al:7,algoregul:15,all:[1,2,4,5,7,14,16],allow:[7,11,12,15],alreadi:[5,14],also:[3,7],alter:1,altern:[7,10,13,14],although:4,an:[1,3,5,6,9,10,13,14,15,16],anaconda3:14,anaconda:13,ani:[3,14],anyth:3,anywher:13,api:15,app:9,append:16,apprend:16,apt:14,ar:[0,1,3,5,6,7,9,10,13,14,15],arbitrari:[5,9],architectur:[2,7],arg:[3,5],argmaskauto:15,argmax:3,argument:[3,10,13,16],ari0u:7,around:7,arrai:5,arrang:16,aspx:14,assign:5,associ:[9,10],attempt:7,attent:[11,12],au:10,augment:0,augmentations_list:0,author:7,auto_insert_metric_nam:0,auto_log_co2:0,auto_lr_find:[0,10],autom:10,automat:[3,10],automatic_optim:3,avail:14,avalaiblememori:15,averag:7,back:[3,7],backward:13,bar:3,base:[1,2,3,7,13],basefinetun:2,basenam:5,basename_l:16,basename_r:16,bash:14,basic:7,batch:[3,7,10],batch_idx:3,batch_siz:[0,5],becom:7,been:3,befor:[5,7,10,14],begin:2,belong:16,below:[13,14],benchmark:7,benefit:7,best:10,between:[3,11,12,15],block:14,bool:6,bootstrap:[11,12],both:7,branch:[9,13,14],branch_nam:13,browser:14,build:[1,7,9,13,14],built:[9,11,12],bypass:5,byprocess:15,c:[13,15],calcul:3,call:[2,3,4,5,10,15],callabl:5,callback:[0,6,11],can:[3,6,7,9,10,13,14,16],capabl:[7,10,13],card:[9,14],care:4,censu:15,challeng:7,chang:14,channel_data:5,charg:15,check:[13,14],checkpoint:[0,1,10,15],checkpointed:10,choos:[3,10,14],cicd:9,ckpt:10,ckpt_path:[0,1,10,15],clamp:5,clamp_sigma:5,class_nam:[2,3],classif:1,click:14,clipandcomputeusingpatchs:0,clone:[14,15],close:7,cmakelist:15,code:[2,11,12,13],code_dir:[0,2],collect:5,com:[13,14],comet:[0,2,10],comet_callback:0,comet_project_nam:0,comet_workspac:0,cometlogg:[0,2],command:[14,15,16],commit:9,common:3,commun:7,compar:7,compat:13,compil:15,complet:[9,14],complex:7,compliant:9,compos:[1,6],compris:4,comprmasqu:15,comput:[3,4,7],conceiv:7,conceptu:7,conda:[9,13,14],condit:1,conduct:[11,12],config:[0,1,3,6,9,10],configur:[1,6,7,10,11,15],configure_optim:3,confmat:2,confus:7,consid:5,constant:10,contain:[1,5,10,16],content:[6,16],context:15,continu:14,contrast:1,contribut:7,control:[3,6],convent:9,cooldown:0,copi:[5,14],core:6,correct:[5,14],correl:15,correladhoc:15,correspond:[11,12],cost:7,costtran:15,could:16,coupl:[6,14],cours:7,coxroyuchar:15,cpp:15,cpu:[0,7],craft:15,creat:[5,9,10,13,14],create_hdf5:[1,5,16],creation:5,credenti:10,criterion:0,csv:[5,10],cu113:13,cuda:[6,11,13,15],cudatoolkit:13,cue:[11,12],current_epoch:2,custom:10,customcompos:5,cwd:0,d:[3,14],d_in:2,dalichebbi:[13,14],data:[3,5,7,10,11,12],data_dir:[0,5,16],dataload:[3,5],dataloader_idx:3,datamodul:[0,6,16],dataset:[0,1,3,7,10],dataset_descript:6,dataset_dir:16,date:[10,14],ddp:10,deactiv:2,debug:[0,6],decent:[11,12],decid:3,decod:[3,7],decor:6,deep:[11,12],deepspe:3,def:3,defaut:[7,10,13],defcor:15,defin:[4,9,10],define_device_from_config_param:6,dens:[11,12,15],denser:7,densifypx_dmtrain_sdr:16,densiti:7,depend:13,design:[11,12],detail:[7,9],detect:7,determin:6,develop:11,devic:[0,13],dezoom:15,dict:[0,3,5],dictconfig:[1,6],dictionari:3,differ:[1,7],dim:3,dimens:2,dir:5,direcli:7,directli:[5,7,13,14],directori:[1,2,5,14,16],dirmec:15,dirpath:0,dirpyram:15,disabl:[0,3,6,10],disp1:5,disp:16,dispar:[1,5,11,12,16],displai:14,distinct:1,distribut:[5,14,16],distro:14,distronam:14,divim:15,dmtrain_sdr:16,doc:3,docker:9,document:9,doe:13,doepi:15,don:[3,5],done:[5,14],doom:7,down:14,download:[5,14,15],downstream:5,driver:[13,14],duplic:5,durat:6,dure:7,e:[1,3,5,7,9,10,14,15],each:[5,7,10],ealgo2prgdyn:15,ealgodequ:15,earlier:14,early_stop:0,earlystop:0,eas:5,easier:[6,13,14],easiest:13,ecomprtiff_non:15,edit:[5,13,14],effect:[5,7,10],efloat32bit:15,egeomimage_epipolairepur:15,egeompxbidim:15,ehr:15,einterpolbilin:15,elev:14,embed:15,en:3,enabl:[3,7,11,12,13,14],encapsul:16,encod:[3,7],end:[2,3],enough:13,ensur:14,enter:14,enumer:[1,5],env:[0,10,13,14],env_exampl:10,environ:[9,10,14],epi:15,epipolar:[11,12],epoch:[0,1,3,10],epoch_:0,eprgdagrsomm:15,error:14,estim:[11,12],et:7,etapemec:15,etapeprogdyn:15,etc:6,eurosdr:16,eurosdr_vahingen_disp_train:16,eurosdr_vahingen_left_train:16,eurosdr_vahingen_masq_train:16,eurosdr_vahingen_right_train:16,eval:3,eval_tim:6,eval_transform:5,evalu:[1,10],everi:4,everyth:[2,14],ex:14,exampl:[3,15,16],example_imag:3,exist:16,expect:7,experi:[3,7,10,11,12],experiment_nam:0,explicit:7,extimin:15,extra:6,extract:[1,15],extract_pt:[1,15],extractor:[11,12],f:[13,14],face:7,factor:0,factori:3,fail:14,fals:[0,3,15],false1:0,false2:0,famou:16,fancier:3,fashion:7,faster:[7,13],featur:[0,9,11,12,14,15],fed:7,feed:5,feedback:7,few:[7,9,14],field:[5,6],file:[0,1,2,5,6,10,14,15,16],filemodelearch:15,filemodelecost:15,filemodeleparam:15,filenam:0,fill:10,filter:5,find_file_in_dir:5,finder:10,finetun:[1,2],finetune_funct:2,finetuningfreezeunfreez:2,first:[5,7,10],fit:[0,1,2,5],fix:7,fixedpoint:7,flag:10,flexibl:[7,11,12],focu:[11,12],focus:7,folder:[0,10,15,16],follow:[1,5,7,9,10,13,14,15,16],forc:6,forg:13,format:[11,12],former:4,forward:[3,4,7,14],found:14,frame:15,freez:2,freeze_before_train:2,frequent:7,friendli:6,from:[1,2,3,5,6,7,10,11,12,13,14,15],full:[7,16],fuller:7,functionn:9,functool:0,further:9,g:[1,3,5,7,9,10],gain:7,game:14,gan:3,geforc:14,gener:[3,5,11,16],generic_model:[0,3,15],genimagescorrel:15,geometr:[7,8,11,12],get:[2,3,5,7,14],get_comet_logg:2,get_logg:6,get_method:0,get_neural_net_class:3,git:[14,15],github:[9,13,14],given:[11,12],glanc:0,glibcxx_3:14,gnu:14,goe:3,gpu:[3,5,6,14],gpus_param:6,gradient:3,grai:5,grid:[3,7],gridsampl:7,ground:7,gt:[1,5],ha:[3,7],hand:[7,15],handl:13,happen:5,hardwar:5,have:[3,13,14,16],hdf5:[0,1,10],hdf5_file_path:[0,5,16],hdf5dataset:5,hdf5stereodatamodul:[0,5],height:5,help:14,here:[0,3,7,14,16],hierach:15,hierarch:[7,15],high:[7,9],higher:14,highprecpyrim:15,hinder:7,hoc:9,hold:14,home:14,hook:4,host:9,hous:9,how:[9,11],howev:5,html:3,http:[3,13,14],hu:7,hydra:[0,1,6,10,11,12],hyperparamet:10,hypothesi:7,i:[1,5,7,9,10,15],ignor:[4,5],ignore_warn:0,ii:7,im1:15,im1dubl:15,im2:15,im2dubl:15,im:15,imag:[1,3,5,9,10,11,12,16],image_path:16,image_paths_by_split_dict:5,image_paths_by_split_dict_typ:5,imageri:15,images_pre_transform:[0,5],implement:[3,4,5,7,9,11,12,15],importerror:14,includ:[3,5],incpix:15,index:[3,5,11,14],infer:[7,11,13,14],inform:[0,5],init:14,initi:[5,6,14],inject:10,inplan:[0,4],input:5,input_data_dir:5,instal:[10,11,15],instanc:[4,13],instanti:[1,5],instead:4,instruct:[10,13,15],integ:5,integr:15,interest:[3,13],interestingli:7,intern:3,interpol:[7,11,15],intersect:7,intric:7,io:3,item:[3,16],iter:[3,5,7,11,12],its:[5,6,9,13],joint:[11,12],keep:[10,11,12,16],kei:3,kernel:14,keyword:3,kind:7,knn:11,kpconv:7,kwarg:[3,5],la:[1,5],labels_hat:3,larg:[5,7,11,12,15],later:7,latest:[3,13,14],latter:[4,11,12],launch:15,launch_extract:1,launch_hdf5:1,launch_qualifi:1,launch_train:1,layer:2,lead:1,learn:[1,3,11,12,15],learning_r:0,learningratemonitor:0,least:13,left:[5,16],len:3,less:7,level:[7,9,15],leverag:[7,9,15],lib:14,libarari:14,librari:[6,9,10,11,12,13,14,15],libstdc:14,libtorch:15,licens:14,lidar:7,lightn:[1,3,5,6,10,11,12,15],lightning_modul:3,lightningdatamodul:6,lightningmodul:6,like:[3,5,7,15],limit:7,line:[15,16],lint:9,linux:[11,14,15],list:[5,6,14],liter:5,loact:5,load:[5,16],local:[7,10],localhost:14,locat:14,log:[2,3,6,10],log_cod:0,log_comet_cm:2,log_dict:3,log_every_n_step:0,log_hyperparamet:6,log_logs_dir:0,log_momentum:0,logcod:[0,2],logger:[0,2,3,6,10],logging_interv:0,logic:15,logit:7,loglogspath:[0,2],logs_dir:10,look:5,loss:[1,3],low:15,lower:15,lr:[0,1],lr_monitor:0,lr_schedul:[0,1],lread_images_and_create_full_data_obj:16,m:9,machin:14,mai:[7,9,10,14,15],main:[6,7,9],mak:16,make:[9,10,13,14],make_grid:3,make_hdf5:16,make_toy_dataset_from_test_fil:5,mamba:13,manag:[0,13,14],mani:7,manual:[3,10,14],map:[5,11,12],margin:0,mask:[1,16],masked_triplet_loss:0,maskedtripletloss:0,masq:[5,16],masq_divid:[0,5],master:[13,14],match:[5,11,12,13,14,15],matrix:[7,13],max_epoch:0,md:9,mean:[3,7,9,10,14],mec:15,meet:7,memori:7,merg:[9,11],method:[2,3,5,6,7,14],metric:[2,3,7,15],micmac:15,might:[3,7,10,14],min:0,min_delta:0,min_epoch:0,mind:7,minut:14,mismatch:14,ml:[2,10],mlp:[11,12,15],mm3d:15,mmodel:7,mmvii:15,mmvii_source_dir:15,mode:[0,3,6,7,13,14],modeagreg:15,modeinterpol:15,model:[0,1,5,6,9,11,12,14],model_checkpoint:0,model_msaff_aerial_decis:15,modelcheckpoint:0,modifi:6,modifii:15,modul:[3,6,11,15],modulationprogdyn:15,momentum:0,monitor:0,more:[0,5,7,10],most:7,move:14,msaff:[11,12],msnet:[0,4],msq1:5,multi:[3,5,6,15],multigpu:10,multipl:[3,5,11],multipli:7,must:[1,3,7,14],mvcnncorrel:15,myria3d:7,n:[7,13],name:[3,6,9,10,14,16],nation:7,navig:14,nbdir:15,nbdirprog:15,nbproc:15,ncc:15,ndarrai:5,necessari:[5,14],need:[3,4,5,7,10,14,15],net:[3,7],network:[1,3,4],neural:[1,3,4],neural_net_class_nam:0,neural_net_hparam:0,newer:13,next:[3,7,10,15],nexu:9,nn:3,nocc_refine_densifypx_dmtrain_sdr:16,nomin:10,non:[11,12,14],none:[2,3,5,6,10],normal:[0,3,5],normalizations_list:0,now:[13,14],np:5,num_class:2,num_nod:0,num_sanity_val_step:0,num_work:[0,5],number:[5,6,10],numpi:5,nvidia:[13,14],nvml:14,o:5,object:[1,5,7,16],oc:0,occlus:[11,12,16],occlusion_mask:16,occur:[7,14],offici:7,older:13,omegaconf:[1,6],on_train_start:2,onc:10,oncuda:15,one:[1,3,4,5,10,16],onli:[3,5,7,14],onlin:14,open:[14,15],oper:[3,7,14],opposit:7,opt1:3,opt2:3,optim:[0,2,3],optimizer_idx:2,option:[2,5,6],optionnali:1,order:[6,7,15],ori:15,other:[7,10,14],otherwis:5,our:[4,7,15],out:[0,2,3,5,10],out_dir:16,outdoor:7,output:3,over:7,overal:16,overfit:10,overlap:[5,8],overridden:4,own:[3,7],packag:[9,14,16],page:[9,10,14],pair:[1,11,12],paper:[7,10],param:[5,6,15],paramet:[1,3,5,6],parametr:15,parcimoni:7,part:[13,14,16],partial:0,particular:7,pass:[3,4,7,9],patch_siz:[0,5],path:[5,10,14,15,16],patienc:[0,1],pattern:5,pentemax:15,pep8:9,per:10,perfom:10,perform:[1,4,5,10,11],persist:14,phase:2,photogrammetri:15,pip:[13,14],pipelin:[1,15],pixel:[11,12],pl_modul:2,place:[6,7],plateform:15,pleas:14,point:[14,15],pointnet:7,posit:5,possibl:3,powershel:14,ppa:14,pre_filt:5,pre_filter_below_n_point:5,precomput:5,predict:[1,2,6,7,10,11],prefer:7,prefetch_factor:[0,5],prefix:14,prepar:[1,5,10,11],prepare_data:5,prepare_dataset:16,prepared_data_dir:5,press:14,previou:16,print:6,print_config:[0,6],prior:15,probabl:[11,12,13],proce:10,process:[0,7,9,11,16],prod:9,produc:3,product:9,progress:[3,8],project_nam:0,prompt:14,propag:7,properli:14,properti:5,provid:[10,15,16],pull:[7,9],pure:[13,14],purg:15,purgemecresultbefor:15,purpos:5,push:9,put:3,px1dilatalti:15,px1dilatplani:15,px1pa:15,px1pentemax:15,px1regul:15,py:[9,10,15,16],pypi:13,pyram:15,pytest:9,python:[6,9,10,15,16],pytorch:[1,3,7,10,11,12,13,14,15],pytorch_lightn:[0,1,2,6],pytorchlightn:15,qi:7,quadro:14,quadruplet:16,qualif:[11,12],qualifi:1,queri:5,quickli:3,r:14,ra:9,randla:7,random:7,rang:10,rapid:[11,12],rate:[1,3],raw:5,re:16,read:3,read_images_and_create_full_data_obj:[0,5],readi:[5,10,14,16],readthedoc:3,recent:13,recept:5,recip:[4,13],recommend:[3,5,14],reconstruct:15,reduc:[7,10,16],reducelronplateau:0,reduct:7,refer:[0,6,10,15],regist:4,regular:8,reinjectmask:15,rel:7,relat:[13,14],releas:9,reli:[11,12],reliabl:7,reload:5,reload_dataloaders_every_n_epoch:5,remov:14,renam:10,replac:14,repositori:[9,14,15],repres:7,represent:16,request:[3,7,9],requir:[3,7,13,14],reset:1,resol:15,resolut:[7,15],resolv:6,restart:14,result:[5,7,10,16],resum:1,review:14,rich:[6,7],right:[5,14,16],robust:[7,15],role:9,routin:[5,15],rtx:14,run:[2,4,9,13,14,16],runtim:[0,10],runtimeerror:14,s3di:7,s:[3,5,11,14],safe:2,same:[3,7,16],sampl:[5,7,16],sample_img:3,sampler:5,samples_hdf5_path:5,satellit:15,save:[6,10],save_dir:0,save_last:0,save_top_k:0,scale:[5,7,11,12,15],schedul:[1,3],scorelearnedmmvii:15,script:11,scroll:14,scv:16,section:[5,10],section_mec:15,section_prisedevu:15,section_result:15,section_terrain:15,section_vrac:15,section_workspac:15,see:[5,7,9,10,13,14,16],seed:[0,6],segment:7,select:14,self:3,semant:[7,9],semantic3d:7,semantickitti:7,semin:7,separ:[11,12],sequenc:6,sequenti:2,serv:[9,15],set:[1,3,5,7,10,15,16],setup:[2,5],setup_env:13,sever:5,sh:14,should:[4,5],show:0,shut:14,shutdown:14,sign_disp_multipli:[0,5],silent:4,sim_base_run_fr:10,simdebug:10,similar:[11,12,15],simlearner3d:[0,9,12,15,16],simpl:7,simpli:[2,16],simplic:7,simplifi:7,sinc:[4,7],singl:[3,5,10],situat:7,size:[5,7,10],skip:3,small:[5,16],smaller:7,so:[3,10,14,16],softwar:15,solut:7,some:[3,10,14],someth:3,sota:7,sourc:[0,1,2,3,4,5,6,14],special:3,specif:[3,5,7,13,14,15],specifi:[1,5,6,7,10],spent:7,split:5,split_csv:5,split_csv_path:[0,5,16],src_las_path:5,ssresoloptim:15,stage:[2,5,9],standard:[5,11,12],standardize_channel:5,standardizeintens:5,standardizeintensitycenteronzero:[0,5],start:[1,2,7,13],state:[1,5,14],stem:10,step:[0,3,7,9,10],stereo:[0,5,15,16],still:13,str:[2,3,5,6],strategi:[7,15],stronger:1,structur:[6,11,12],sub:16,subclass:4,subdir:1,subdirectori:5,subfold:[5,10],suboptim:7,subsequ:16,subset:16,subtil:5,subtile_overlap_predict:[0,5],subtile_overlap_train:[0,5],subtile_width:[0,5],success:9,sudo:14,suit:16,sum:3,superior:7,supervis:7,support:[3,10,11],sure:14,surfac:15,symb:15,system:[13,14],szblocah:15,szdallemax:15,szdallemin:15,szmindecomposcalc:15,szrecouvrtdal:15,szw:15,t:[3,5],tackl:15,tag:9,take:[4,7,14,16],tarbal:[13,14],task:[0,1,6,10,15,16],task_nam:[0,1,5,10,15,16],td_prepar:5,tell:[3,11,12,16],templat:[11,12],tensor:[3,5,16],term:7,termin:14,test:[0,1,2,3,5,7,8,14,15],test_acc:3,test_dataload:5,test_loss:3,test_step:[3,5],text:[3,14],than:[7,10,13,14],thank:13,thei:16,them:[4,7],thi:[1,3,4,5,6,7,9,10,11,12,13,14,15,16],thing:3,third:7,thisdir:15,thoma:7,thu:[7,16],tif:[5,15,16],tile:5,tile_height:[0,5],tile_width:[0,5],time:[5,7,8,10,16],tini:7,tmpmec:15,tmppyr:15,tmpresult:15,todo:11,togeth:[5,15],toi:[5,10],toolchain:14,toolkit:[13,14],torch:[0,3,5,15,16],torch_geometr:7,torch_install_prefix:15,torchvis:3,totensor:5,toy_dataset:16,tpu:3,track:9,train:[2,3,5,7,11,12,15],train_dataload:5,train_transform:5,trainabl:6,trainer:[0,1,2,6,10],training_step:3,transform:[0,7],tree:6,tri:5,trigger:7,tune:2,tupl:5,turn:[5,16],tutori:[9,10,15],txt:[15,16],type:[1,3,5,14],typecah:15,typeel:15,typepyramimag:15,typic:[1,10],ubuntu:14,unet32:[11,12],unet:[11,12],unetmlpmatch:15,unfreez:2,unfreeze_decoder_train_epoch:2,unfreeze_fc_end_epoch:2,union:[5,7],unless:5,until:7,up:10,updat:[2,14],upgrad:[13,14],upload:2,upon:[11,12],urban:7,us:[1,2,3,5,6,7,9,10,11,12,13,14,15,16],useful:5,usemlp:15,usepredicnet:15,usernam:14,usual:7,util:[0,3,11,16],v2:15,v3:7,v:9,val:[3,5],val_acc:3,val_dataload:5,val_loss:[0,3],valdefcorrel:15,valid:[1,2,3,5,10],validation_step:[3,5],valu:[0,1,3,5],variabl:[7,10],veget:7,verbos:0,version:[7,13,14,15],via:9,virtual:[10,14],wa:[7,11,12],wai:[7,13,14],want:7,warn:[6,8,13],we:[0,3,7,13,14,15,16],well:[5,14],were:7,what:[3,6],whatev:3,wheel:13,when:[2,3,9,10],where:[5,10,15,16],whether:[6,14],which:[1,3,6,7,9,10,16],why:14,width:5,window:[14,15],wise:5,with_model:15,within:[4,7,14],without:[5,11,12],work:[2,7,8,13,14,15],work_dir:0,workdir:15,workflow:9,workspac:0,wsl2:11,wsl:14,www:14,x10:7,x5:7,x86_64:14,x:[3,4],xml:15,xxx:15,y:[3,5,13],y_mean:5,y_std:5,yaml:[9,10],ye:14,yml:[13,14],you:[3,5,10,13,14,15,16],your:[3,10,13,14],yourself:5,z:3,zoom:15,zreg:15},titles:["Default configuration","Scripts","simlearner3d.callbacks","simlearner3d.models","simlearner3d.models.modules","simlearner3d.processing","simlearner3d.utils","General design of the package","KNN-Interpolation to merge multiple predictions [TODO]","Developer\u2019s guide","How to train new models","Simlearner3D &gt; Documentation","&lt;no title&gt;","Install Simlearner3D on Linux","Install Simlearner3d on WSL2 with CUDA support","Performing inference on new data","Preparing data for training"],titleterms:{"default":0,"function":16,"import":7,"new":[10,15],anaconda:14,approach:7,attent:4,augment:5,background:11,callback:2,cd:9,ci:9,ckpt:15,cloud:7,code:9,comet_callback:2,compos:5,configur:0,content:2,continu:9,creat:16,csv:16,cuda:14,data:[15,16],datamodul:5,dataset:[5,16],deliveri:9,design:7,develop:9,document:11,environ:13,epipolar:15,essenc:7,evalu:7,fast:7,finetuning_callback:2,gener:7,get:[11,16],gpu:10,guid:[9,11],hdf5:[5,16],how:10,imag:15,improv:7,indic:11,infer:[10,15],instal:[13,14],integr:9,interpol:8,kei:7,knn:8,learn:10,linux:13,logging_callback:2,merg:8,model:[3,4,7,10,15],modul:[2,4],msaff:4,multi:10,multipl:8,optim:10,packag:[7,11,13],pair:15,peprocess:16,perform:[7,15],point:7,practic:7,predict:8,prepar:16,prerequisit:13,process:5,pt:15,quick:10,quickli:16,rate:10,refer:11,right:7,run:[1,10,15],s:9,script:[1,15],select:7,set:[13,14],setup:10,should:7,simlearner3d:[1,2,3,4,5,6,11,13,14],sourc:13,speed:7,split:16,start:[11,16],structur:7,submodul:2,subsampl:7,support:14,tabl:11,test:[9,10,16],todo:8,toi:16,toy_dataset:5,train:[1,10,16],transform:[5,15],troubleshoot:[13,14],unet32:4,unet:4,up:[13,14],util:[5,6],val:16,version:9,virtual:13,wsl2:14}})
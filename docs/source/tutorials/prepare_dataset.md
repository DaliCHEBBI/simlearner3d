# Preparing data for training

## Peprocessing functions

The loading function is `lread_images_and_create_full_data_obj` by default. It takes `image_paths` which a quadruplet of  `left`, `right`, `disparity` and `occlusion_mask` image names and generates a `Data` object `simlearner3d.utils.utils.Data` which is simply a set of `torch.Tensor` representations of the image set. 

## Creating Train/Val/Test split dataset and split.csv 

To create the famous folder of sub sets arranged as Train, Val and Test, we use `task.task_name="prepare_dataset"` which takes files:
* Item left `full path` image names 
* Item right `full path` images names 
* Item disparity `full path` images names 
* Item occlusion masks `full path` images names 

Here is an example of a command that create `Stereo` folder with samples distributed as Train, Val or Test

```bash
DATASET_DIR="/path/to/images/disparities/maks/sets/of/files"
python simlearner3d/processing/dataset/toy_dataset.py +task.task_name="prepare_dataset" \
prepare_dataset.left="${DATASET_DIR}eurosdr_vahingen_left_train.txt" \
prepare_dataset.right="${DATASET_DIR}eurosdr_vahingen_right_train.txt" \
prepare_dataset.disp="${DATASET_DIR}eurosdr_vahingen_disp_train.txt" \
prepare_dataset.masq="${DATASET_DIR}eurosdr_vahingen_masq_train.txt" \
prepare_dataset.out_dir="${DATASET_DIR}Stereo"
```

The resulting `Stereo` folder contains the following sub folders:

* Item train/
* Item val/
* Item test/

It additionally contains a `split.csv` file that tells which quadruplet of images (left, right, disparity, mask) belongs to which subset (train,val,test)

Here is an example of split.csv content 

```
basename_l,basename_r,disparity,masq,split
DMTrain_SDR-05_20_0003_Im1.tif,DMTrain_SDR-05_20_0003_Im2.tif,DensifyPx_DMTrain_SDR-05_20_0003_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-05_20_0003_Im1.tif,train
DMTrain_SDR-20_30_0002_Im1.tif,DMTrain_SDR-20_30_0002_Im2.tif,DensifyPx_DMTrain_SDR-20_30_0002_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-20_30_0002_Im1.tif,train
DMTrain_SDR-06_20_0001_Im1.tif,DMTrain_SDR-06_20_0001_Im2.tif,DensifyPx_DMTrain_SDR-06_20_0001_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-06_20_0001_Im1.tif,train
DMTrain_SDR-05_20_0006_Im1.tif,DMTrain_SDR-05_20_0006_Im2.tif,DensifyPx_DMTrain_SDR-05_20_0006_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-05_20_0006_Im1.tif,train
DMTrain_SDR-05_21_0014_Im1.tif,DMTrain_SDR-05_21_0014_Im2.tif,DensifyPx_DMTrain_SDR-05_21_0014_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-05_21_0014_Im1.tif,train
DMTrain_SDR-05_19_0017_Im1.tif,DMTrain_SDR-05_19_0017_Im2.tif,DensifyPx_DMTrain_SDR-05_19_0017_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-05_19_0017_Im1.tif,train
DMTrain_SDR-04_29_0005_Im1.tif,DMTrain_SDR-04_29_0005_Im2.tif,DensifyPx_DMTrain_SDR-04_29_0005_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-04_29_0005_Im1.tif,train
DMTrain_SDR-05_21_0004_Im1.tif,DMTrain_SDR-05_21_0004_Im2.tif,DensifyPx_DMTrain_SDR-05_21_0004_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-05_21_0004_Im1.tif,train
DMTrain_SDR-07_18_0011_Im1.tif,DMTrain_SDR-07_18_0011_Im2.tif,DensifyPx_DMTrain_SDR-07_18_0011_Im1.tif,Nocc_refine_DensifyPx_DMTrain_SDR-07_18_0011_Im1.tif,train
```


If you have an additional dataset with left, right, disparity and masks files, you can apprend this set to the previous one by re-running the same command with the new files and keeping `prepare_dataset.out_dir` the same so that they could be split and appended to the previous subset.



## Create a HDF5 Dataset

After generating the sub splits of a dataset (train/val/test) and their subsequent split.scv file, we can generate an overall hdf5 file that encapsulates all the dataset into one file thus reducing training times. To do that, you can run the following command line which takes the following arguments:

* Item `datamodule.data_dir`: The directory where train,val,test folders exist
* Item `datamodule.split_csv_path`: the split.csv split file full path
* Item `datamodule.hdf5_file_path`: the to-generate hdf5 file full path

Here is an example:


```bash
DATASET_DIR="/path/to/images/disparities/maks/sets/of/files"
python run.py task.task_name=create_hdf5 \
datamodule.data_dir="${{DATASET_DIR}}Stereo" \
datamodule.split_csv_path="${{DATASET_DIR}}Stereo/split.csv" \
datamodule.hdf5_file_path="${DATASET_DIR}Stereo/eurosdr.hdf5"
```

## Getting started quickly with a toy dataset

A LAS file is provided as part of the test suite. It can be turned into a small, training-ready dataset to get started with the package. 

To create a toy dataset run :
```
python myria3d/pctl/dataset/toy_dataset.py
```

You will see a new file: `/test/data/toy_dataset.hdf5`.
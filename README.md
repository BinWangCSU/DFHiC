# DFHiC

# Installation
DFHiC can be downloaded by
```shell
git clone https://github.com/BinWangCSU/DFHiC
```

# Enviroment setup
## Dependency 
- Python==3.6.10
- Tensorflow-gpu==1.10.0
- Tensorlayer==1.9.1
- numpy==1.14.5
- scikit-image==0.14.5
- scikit-learn==0.19.2

The code is compatible with both TensorFlow v1 and TensorLayer. Our models are trained with GPUs. 
See `environment.yml` for all prerequisites, and you can also install them using the following command.

```shell
conda env create -f environment.yml
```

# Instructions
We provide detailed step-by-step instructions for running DFHiC model for reproducing the results in the original paper and processed train data and test data be provided [here](https://drive.google.com/drive/folders/12EQWb1OEsA16wRmXv_cxPLv1FIkEyAGh).
##  Download raw aligned sequencing reads

We download alighed sequencing reads([GSE62525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525)) from Rao *et al*. 2014 (e.g. ```GSM1551550_HIC001_merged_nodups.txt.gz``` ), and you can donwlaod data using the `raw_data_download_script.sh` script. You will download data to `CELL` folder, such as `GM12878`.
```shell
bash raw_data_download_script.sh GM12878 
```

## Data preprocessing

We preprocess Hi-C data from alighed sequencing reads using `preprocess.sh` and `generate_train_data.py`. One can directly downsample raw data and generate raw Hi-C contacts matrix by using `preprocess.sh`, and finally save the data in this folder.
```shell
bash preprocess.sh GM12878 10000 juicer_tools.jar 
```
Data for training and evaluating the model can be obtained by directly runing `generate_train_data.py`, and the resulting training and test sets are saved in `preprocess/data/CELL` folder. We provide training files in [here](https://drive.google.com/drive/folders/12EQWb1OEsA16wRmXv_cxPLv1FIkEyAGh).
```shell
python generate_train_data.py GM12878 16
```

## Train DFHiC model
To train:
```shell
python run_train.py [GPU_ID] [CHECKPOINT_PATH] [GRAPH_PATH] [BLOCK_SIZE]
python run_train.py 0 checkpoint/ graph/ 40
```

To evaluate DFHiC model on test data:
```shell
python run_test.py [GPU_ID]
python run_test.py -1
```
We provide pretained weights for DFHiC model.

## Predicting
We can directly enhance the entire chromosome Hi-C matrix by `run_prediction.py`, and you can also enhance your own data through DFHiC:
```shell
python run_predict.py [GPU_ID] [CHROME_ID]
python run_predict.py -1 22
```

# License
This project is licensed under the MIT License - see the LICENSEfile for details

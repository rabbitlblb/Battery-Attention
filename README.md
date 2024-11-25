# Environment requirement
We recommend using the conda environment.
```
conda env create -f environment.yaml
```
# Dataset preparation
## Download
Download from the link in our paper. 
Please make sure the data structure is like the following. 

```
    |--data
        |--battery_brand1
            |--data
            |column.pkl
        |--battery_brand2
            |--...
        |--battery_brand3
            |--...
        |--SOH_snippets
            |--...
    
```

## File content

Each `pkl` file is a tuple including two parts. The first part is charging time series
data. `column.pkl` contains column names for the time sequence data. 
The second part is meta data which contains fault label, car number, charge segment number, anomaly label and mileage or capacity. 

## Generate path information for five-fold validation

To facilitate the organization of training and test data, we use 1) a python dict to save 
`car number-snippet paths` information, which is named as `all_car_dict.npz.npy`, and 2) a dict to save the
randomly shuffled normal and abnormal car number to perform five-fold training and testing, which is 
named as `ind_odd_dict*.npz.npy`. By default, the code is running on the first brand. So our code
is now running on `ind_odd_dict_all.npz.npy`. 

To build the `all_car_dict.npz.npy` and `ind_odd_dict*.npz.npy`, run

`cd data`

Run `five_fold_train_test_split.ipynb` and then you get all the files saved in 
`five_fold_utils\`.
(Running each cell of the `five_fold_train_test_split.ipynb` may take 
a few minutes. If not, please check the data path carefully.)

The cell output of each cell contains randomly shuffled `ind_car_num_list` 
and `ood_car_num_list`. You may print it out to see the car numbers you are using. 

## Generate path information for SOH estimation

Similarly, you need to change the path in `data\SOH_snippets\change.ipynb` and run it to change the path information to current path.

# Run battery attention

## train
Please check the `configs/model_params_BA.json` files carefully for hyperparameter settings.
And use `fold_num` to do the five-fold training and testing. To start training, run
```
cd Battery_Attention
python main_five_fold.py --config_path configs/model_params_BA.json --fold_num 0
```
If you want to fully run the five-fold experiments, you should run five times with different 
`--fold_num`.
After training, the reconstruction errors of data are recorded in `save_model_path` configured by the
`model_params_BA.json` file.

## transformer, DyAD and independent models

You can see the config json in `configs` folder.
We set the battery attention rank to zero to get a transformer without battery attention.

## evaluation methods

We implemented a lot of evaluation methods in `evaluate.py'.
We only use the reconstruction error.
If you want to use other evaluation methods, you can change the code to use them.

# AutoEncoder

## train
To start training, run
```
cd AE_and_SVDD
python traditional_methods.py --method auto_encoder --normalize --fold_num 0
```
If you want to fully run the five-fold experiments, you should run five times with different 
`--fold_num`.

# LSTM-AD

## train
To start training, run
```
cd Recurrent-Autoencoder-modify
python main.py configs/config_lstm_ae_battery_0.json
```
# GDN

**Setting another brand:** By default, we are using brand 1. To run experiments on other brands, 
one should manually change the variable
`ind_ood_car_dict` in `GDN_battery/datasets/TimeDataset.py` and `GDN_battery/main.py` 
similarly as above. 

## train
```
cd GDN_battery
bash run_battery.sh 3 battery 1 0 20
```
where `3` is the gpu number, `battery` is the dataset name, 
`1` is the fold number (also can be 0/2/3/4) `0` means use all data to train and `20` is epoch number.
For details, please see the `.sh` file. 

# XGBoost and LightGBM

## train
To start training, run
```
cd SOH_XGB_LGBM
python main.py --method xgboost --fold_num 0
python main.py --method lightgbm --fold_num 0
```

# Unseen

We write the unseen model independently.
You can run it similarly in the `*_unseen` folders. 
The only thing you need to change is the dataloaders.
In battery_attention, you can set the list of unseen types in `model_params_BA.json`.
In other methods, you need to add `--unseen` to set the unseen type.

# Capacity Estimation

We write the unseen model independently.
You can run it similarly in the `*_unseen` folders. 
The only thing you need to change is the dataloaders.
In battery_attention, you can set the list of unseen types in `model_params_BA.json`.
In other methods, you need to add `--unseen` to set the unseen type.

# Calculate the results
You can calculate the AUROC values and the accuracy of capacity estimation with 
jupyter-notebooks in `notebooks`.

# Code Reference
We use partial code from 
```
https://github.com/962086838/Battery_fault_detection_NC_github
https://github.com/yzhao062/pyod
https://github.com/d-ailin/GDN
https://github.com/PyLink88/Recurrent-Autoencoder
``` 

import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm
import random

def _get_car_num_list(ind_ood_car_dict_path, debug=False):
    ind_ood_car_dict = np.load(ind_ood_car_dict_path, allow_pickle=True).item()
    return ind_ood_car_dict['ind_sorted'], ind_ood_car_dict['ood_sorted']

def _get_car_num(ind_car_num_list, ood_car_num_list, fold_num, sample_rate, unknown):
    inn = len(ind_car_num_list)
    odn = len(ood_car_num_list)
    train = ind_car_num_list[:int(fold_num * inn / 5)] + ind_car_num_list[int((fold_num + 1) * inn / 5):]
    #val = ind_car_num_list[int(fold_num * inn / 5):int((fold_num + 1) * inn / 5)] + ood_car_num_list
    test = ind_car_num_list[int(fold_num * inn / 5):int((fold_num + 1) * inn / 5)] + ood_car_num_list
    if len(unknown) > 0:
        if int(ind_car_num_list[0]) // 1000 in unknown:
            train = []
            test = ind_car_num_list + ood_car_num_list
        else:
            test = []
            train = ind_car_num_list + ood_car_num_list
    if sample_rate != 1:
        train = train[:max(int(len(train) * sample_rate), 1)]
        #val = val[:max(int(len(val) * sample_rate / 2), 1)] + val[-max(int(len(val) * sample_rate / 2), 1):]
        test = test[:max(int(len(test) * sample_rate / 2), 1)] + test[-max(int(len(test) * sample_rate / 2), 1):]
    return train, test, test

def load_dataset(car_number, all_car_dict, p_bar):
    dataset = []
    for each_num in car_number:
        p_bar.set_description(f'loading files - car number {each_num}')
        for each_pkl in all_car_dict[each_num]:
            train1 = torch.load(each_pkl)
            if np.isnan(train1[0]).any():
                for x in train1[0]:
                    print(x)
                print(train1[1], each_pkl)
                exit()
            dataset.append(train1)
            p_bar.update(1)
    return dataset

def load(data_path, all_car_dict_path='../five_fold_utils/all_car_dict.npz.npy', 
     ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict_all.npz.npy',
     train=True, fold_num=0, sample_rate=1, unknown = []):
    ind_car_num_list, ood_car_num_list = _get_car_num_list(ind_ood_car_dict_path)
    all_car_dict = np.load(all_car_dict_path, allow_pickle=True).item()

    train_number, val_number, test_number = _get_car_num(ind_car_num_list, ood_car_num_list, fold_num, sample_rate=sample_rate, unknown = unknown)

    print('train car number:', train_number)
    print('val car number:', val_number)
    print('test car number:', test_number)
    
    
    total_pkl = 0
    for each_num in train_number + val_number + test_number:
        total_pkl += len(all_car_dict[each_num])
        
    p_bar = tqdm(total=total_pkl)
    
    return load_dataset(train_number, all_car_dict, p_bar), load_dataset(val_number, all_car_dict, p_bar), load_dataset(test_number, all_car_dict, p_bar)

if __name__ == '__main__':
    for i in [1,2,4,5,6]:
        dataset = Dataset(data_path='.', sample_rate=1, print_bar=False, all_car_dict_path='../../five_fold_utils/all_car_dict.npz.npy', ind_ood_car_dict_path=f'../../five_fold_utils/ind_odd_dict{i}.npz.npy')
        data = np.array([d[0] for d in dataset.battery_dataset])
        np.set_printoptions(suppress=True, precision=2, linewidth=1000)
        print(i, np.mean(data, axis=(0,1)), np.std(data, axis=(0,1)))
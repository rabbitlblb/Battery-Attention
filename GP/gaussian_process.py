import odditylib as od
import torch
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fold_num', help='fold_num', type = int, default=0)
parser.add_argument('--dataset', help='dataset', type = str, default='brand1')


args = parser.parse_args()

# Reading data
fold_num = args.fold_num
all_car_dict = np.load('../five_fold_utils/all_car_dict.npz.npy', allow_pickle=True).item()
# print(all_car_dict)
# assert 1==0

ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict_all.npz.npy', allow_pickle=True).item()
ind_car_num_list = ind_ood_car_dict['ind_sorted']
ood_car_num_list = ind_ood_car_dict['ood_sorted']


train_car_number = []
test_car_number = []
for i in range(1, 7):
    ind_ood_car_dict = np.load(f'../five_fold_utils/ind_odd_dict{i}.npz.npy', allow_pickle=True).item()
    ind_car_num_list = ind_ood_car_dict['ind_sorted']
    ood_car_num_list = ind_ood_car_dict['ood_sorted']
    train_car_number += ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] + ind_car_num_list[
                                                                        int((fold_num + 1) * len(ind_car_num_list) / 5):]
    test_car_number += ind_car_num_list[
                            int(fold_num * len(ind_car_num_list) / 5):int((fold_num + 1) * len(ind_car_num_list) / 5)] + ood_car_num_list

train_X = []
train_y = []
ind_snippet_number = 0
ood_snippet_number = 0
for each_num in tqdm(train_car_number):
    for each_pkl in all_car_dict[each_num]:
        if each_num in ind_car_num_list:
            ind_snippet_number += 1
        else:
            ood_snippet_number += 1
        train1 = torch.load(each_pkl)
        train_X.append(np.array(train1[0])[:, 0:6].reshape(1, 128, 6))  # (128, 6)
        train_y.append(int(train1[1]['label'][0]))

train_X = np.concatenate(train_X, axis=0)
train_y = np.vstack(train_y)

print(train_X.shape)
print(train_y.shape)

y_train_scores = []
for each_X, each_y in tqdm(zip(train_X, train_y), total=len(train_X)):
    fit_error = 0
    for each_channel_num in range(each_X.shape[1]):
        each_data = each_X[:, each_channel_num]
        detector = od.Oddity()  # Creating a default Oddity detector
        detector.fit(each_data.reshape(-1, 1))  # Fitting the detector on our data
        mu, cov = detector.mu, detector.cov
        fit_error += sum((mu - each_data) ** 2)
    y_train_scores.append(fit_error)
os.makedirs("gaussian_process", exist_ok=True)
os.makedirs(f"gaussian_process/{args.dataset}", exist_ok=True)
np.save(f"gaussian_process/{args.dataset}/" + f'y_train_scores_fold{args.fold_num}.npy', y_train_scores)



test_X = []
test_y = []
ind_snippet_number = 0
ood_snippet_number = 0
for each_num in tqdm(test_car_number):
    for each_pkl in all_car_dict[each_num]:
        if each_num in ind_car_num_list:
            ind_snippet_number += 1
        else:
            ood_snippet_number += 1
        train1 = torch.load(each_pkl)
        test_X.append(np.array(train1[0])[:, 0:6].reshape(1, 128, 6))
        test_y.append(int(train1[1]['label'][0]))
test_X = np.concatenate(test_X, axis=0)
test_y = np.vstack(test_y)

print(test_X.shape)
print(test_y.shape)

y_test_pred = []
y_test_scores = []
y_test = []
for each_X, each_y in tqdm(zip(test_X, test_y), total=len(test_X)):
    fit_error = 0
    for each_channel_num in range(each_X.shape[1]):
        each_data = each_X[:, each_channel_num]
        detector = od.Oddity()  # Creating a default Oddity detector
        detector.fit(each_data.reshape(-1, 1))  # Fitting the detector on our data
        mu, cov = detector.mu, detector.cov
        fit_error += sum((mu - each_data) ** 2)
    y_test_scores.append(fit_error)
    y_test.append(each_y)

os.makedirs("gaussian_process", exist_ok=True)
os.makedirs(f"gaussian_process/{args.dataset}", exist_ok=True)


np.save(f"gaussian_process/{args.dataset}/" + f'y_test_pred_fold{args.fold_num}.npy', y_test_pred)
np.save(f"gaussian_process/{args.dataset}/" + f'y_test_scores_fold{args.fold_num}.npy', y_test_scores)
np.save(f"gaussian_process/{args.dataset}/" + f'y_test_fold{args.fold_num}.npy', y_test)


import argparse
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

def load_dataset(fold_num, train=True):
    all_car_dict = np.load('../data/SOH_snippets/changed_files/all_car_dict.npz.npy', allow_pickle=True).item()

    car_number = []
    for i in range(1, 5):
        ind_ood_car_dict = np.load(f'../data/SOH_snippets/changed_files/ind_odd_dict{i}.npz.npy', allow_pickle=True).item()
        ind_car_num_list = ind_ood_car_dict['ind_sorted']
        ood_car_num_list = ind_ood_car_dict['ood_sorted']
        if train:
            car_number += ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] + ind_car_num_list[
                                                                    int((fold_num + 1) * len(ind_car_num_list) / 5):]
        else:
            car_number += ind_car_num_list[
                        int(fold_num * len(ind_car_num_list) / 5):int((fold_num + 1) * len(ind_car_num_list) / 5)] + ood_car_num_list

    #car_number = car_number[:10]
    
    X = []
    y = []
    cars = []
    print('car_number is ', car_number)

    p_bar = tqdm(total = np.sum([len(all_car_dict[each_num]) for each_num in car_number]))

    for each_num in car_number:
        p_bar.set_description(f'Processing car {each_num}')
        for each_pkl in all_car_dict[each_num]:
            train1 = torch.load(each_pkl[0])
            X.append(train1[0][:, 0:6].reshape(1, -1))
            y.append(train1[1]['label'][0] / 100.0)
            cars += [train1[1]['car']]
            p_bar.update(1)
    
    p_bar.close()

    X = np.vstack(X)
    y = np.vstack(y)
    cars = np.array(cars)

    return X, y, cars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch battery Example')
    parser.add_argument('--method', type=str, default='xgboost')
    parser.add_argument('--fold_num', type=int, default=0)

    args = parser.parse_args()

    X_train, y_train, cars_train = load_dataset(fold_num=args.fold_num, train=True)
    X_test, y_test, cars_test = load_dataset(fold_num=args.fold_num, train=False)
    
    print('loaded data')
    print('X_train shape', X_train.shape, 'X_test shape', X_test.shape)

    parameters = {
        'learning_rate': [0.1, 0.01],
        'feature_fraction': [0.5, 0.8, 1.0],
        'num_leaves': [8],
        'max_depth': [3, 6]
    }

    if args.method == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', eval_metric='rmse')
        grsearch = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', verbose=0, n_jobs=1)
        grsearch = grsearch.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    elif args.method == 'lightgbm':
        model = lgb.LGBMRegressor(n_estimators=100)
        grsearch = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', verbose=0, n_jobs=1)
        grsearch = grsearch.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse')
    else:
        raise NotImplementedError
    
    model = grsearch.best_estimator_
    y_pred = model.predict(X_test)
    
    mse = defaultdict(list)
    for i in range(len(cars_test)):
        mse[cars_test[i]].append((y_pred[i] - y_test[i]) ** 2)
    
    rmse = {k: np.mean(v) for k, v in mse.items()}
    
    os.makedirs('./results', exist_ok=True)
    with open(f'./results/{args.method}_results_fold_{args.fold_num}.json', 'w') as f:
        json.dump(rmse, f)

    # print("Loaded configs at %s" % args.config_path)
    # print("args", args)

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from decimal import Decimal
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.suod import SUOD
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, ensemble
from tqdm import tqdm
from model.dataset import _get_car_num_list, _get_car_num
import traceback
import shutil

class Evaluate:
    def __init__(self, args):
        """
        :param project: class model.projects.Project object
        """
        self.args = args
        auc_picture_path = os.path.join(self.args.current_path, "auc")
        shutil.rmtree(auc_picture_path, ignore_errors=True)
        self.mkdir(auc_picture_path)
        self.args.auc_picture_path = auc_picture_path
        
        
    @staticmethod
    def mkdir(path):
        """
        create folders
        :param path: path
        """
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)

    @staticmethod
    def get_feature_label(data_path, mixcar=False, max_group=None):
        if max_group is None:
            max_group = len(data_path) // 2
        data, label = [], []
        for f in tqdm(sorted(os.listdir(data_path), key=lambda x: x.split('_')[0])):
            if int(f.split('_')[0]) > max_group:
                break
            else:
                if f.endswith(".file"):
                    temp_label = torch.load(open(os.path.join(data_path, f), 'rb'))
                    if mixcar:
                        temp_label['car'] = ['0'+_ for _ in temp_label['car']]
                    label += np.array(
                        [[i[0] for i in temp_label['label']], temp_label['car'], temp_label['mileage'].tolist(), [float(x) for x in temp_label['rec_error']]]).T.tolist()
                elif f.endswith(".npy"):
                    data += np.load(os.path.join(data_path, f)).tolist()
        
        return np.array(data), np.array(label)

    @staticmethod
    def calculate_rec_error(train_x, train_label, test_x, test_label):
        train_rec_sorted_index = np.argsort(-train_label[:, -1].astype(float))
        train_res = [[train_label[i][1], train_label[i][0], train_label[i][2], float(train_label[i][-1])] for i in train_rec_sorted_index]
        test_rec_sorted_index = np.argsort(-test_label[:, -1].astype(float))
        test_res = [[test_label[i][1], test_label[i][0], test_label[i][2], float(test_label[i][-1])] for i in test_rec_sorted_index]
        
        return pd.DataFrame(train_res, columns=['car', 'label', 'mileage', 'rec_error']), pd.DataFrame(test_res, columns=['car', 'label', 'mileage', 'rec_error'])
    
    @staticmethod
    def calculate_svm_score(train_x, train_label, test_x, test_label):
        linearsvc = svm.LinearSVC(random_state=0, dual=False)

        params_search = {'C': [1e-3, 1e-2, 1, 10, 100, 1000], 'tol': [0.0001, 0.00001]}
        grid_search_params = {'estimator': linearsvc,
                              'param_grid': params_search,
                              'cv': 3,
                              'n_jobs': 32,
                              'verbose': 12}
        grsearch = GridSearchCV(**grid_search_params)
        grsearch = grsearch.fit(train_x, np.array(train_label[:,0]))
        
        best_model = grsearch.best_estimator_

        svm_train_scores = best_model.decision_function(train_x)
        svm_test_scores = best_model.decision_function(test_x)

        svm_sorted_index_train = np.argsort(-svm_train_scores)
        svm_sorted_index_test = np.argsort(-svm_test_scores)
        train_res = [train_label[i][[1, 0, 2]].tolist() + [svm_train_scores[i]] for i in svm_sorted_index_train]
        test_res = [test_label[i][[1, 0, 2]].tolist() + [svm_test_scores[i]] for i in svm_sorted_index_test]
        
        return pd.DataFrame(train_res, columns=['car', 'label', 'mileage', 'svm_score']), pd.DataFrame(test_res, columns=['car', 'label', 'mileage', 'svm_score'])
    
    @staticmethod
    def calculate_lgbm_score(train_x, train_label, test_x, test_label):
        learning_rate = [0.01, 0.1]
        num_leaves = [8]
        max_depth = [3,6]
        feature_fraction = [0.5, 0.8, 1]

        parameters = {'learning_rate': learning_rate,
                      'feature_fraction': feature_fraction,
                      'num_leaves': num_leaves,
                      'max_depth': max_depth}
        model = LGBMClassifier(n_estimators=80, n_jobs=-1, verbose=0)

        ## 进行网格搜索
        grsearch = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=0, n_jobs=-1)
        grsearch = grsearch.fit(train_x, np.array(train_label[:,0]))
        
        best_model = grsearch.best_estimator_

        lgbm_train_scores = best_model.predict_proba(train_x)[:, 1]
        lgbm_test_scores = best_model.predict_proba(test_x)[:, 1]

        lgbm_sorted_index_train = np.argsort(-lgbm_train_scores)
        lgbm_sorted_index_test = np.argsort(-lgbm_test_scores)
        train_res = [train_label[i][[1, 0, 2]].tolist() + [lgbm_train_scores[i]] for i in lgbm_sorted_index_train]
        test_res = [test_label[i][[1, 0, 2]].tolist() + [lgbm_test_scores[i]] for i in lgbm_sorted_index_test]
        
        return pd.DataFrame(train_res, columns=['car', 'label', 'mileage', 'lgbm_score']), pd.DataFrame(test_res, columns=['car', 'label', 'mileage', 'lgbm_score'])
    
    @staticmethod
    def calculate_pyod_score(train_x, train_label, test_x, test_label, method, **kwargs):
        eval(f'exec("from pyod.models.{method.lower()} import {method}")')
        print(method, kwargs)
        clf = eval(method)(**kwargs)
        if method == 'XGBOD':
            clf.fit(train_x, np.array(train_label[:,0]))
        else:
            clf.fit(train_x)
        
        clf_train_scores = clf.decision_function(train_x)
        clf_test_scores = clf.decision_function(test_x)

        clf_sorted_index_train = np.argsort(-clf_train_scores)
        clf_sorted_index_test = np.argsort(-clf_test_scores)
        train_res = [train_label[i][[1, 0, 2]].tolist() + [clf_train_scores[i]] for i in clf_sorted_index_train]
        test_res = [test_label[i][[1, 0, 2]].tolist() + [clf_test_scores[i]] for i in clf_sorted_index_test]
        
        return pd.DataFrame(train_res, columns=['car', 'label', 'mileage', f'{method}_score']), pd.DataFrame(test_res, columns=['car', 'label', 'mileage', f'{method}_score'])
    
    def calculate_AUC(self, method, name, train_x, train_label, test_x, test_label, **kwargs):
        try:
            print(f'Evaluating by {method}')
            if method[:4] == 'pyod':
                train_result, test_result = self.calculate_pyod_score(train_x, train_label, test_x, test_label, method[5:], **kwargs)
                train_result.to_csv(os.path.join(self.args.result_path, f"train_segment_{name}_{method}_scores.csv"))
                test_result.to_csv(os.path.join(self.args.result_path, f"test_segment_{name}_{method}_scores.csv"))
                #self.pltAUC(train_result, test_result, f'{name}_{method}_score')
            else:
                train_result, test_result = eval(f'self.calculate_{method}')(train_x, train_label, test_x, test_label)
                train_result.to_csv(os.path.join(self.args.result_path, f"train_segment_{name}_{method}.csv"))
                test_result.to_csv(os.path.join(self.args.result_path, f"test_segment_{name}_{method}.csv"))
                #self.pltAUC(train_result, test_result, f'{name}_{method}')
            print(f'Evaluate Ended by {method}')
        except:
            traceback.print_exc()
    
    def find_drop(self, k, method, label, scores, gran = 100):
        gran = 100
        best_fpr, best_tpr, best_thresholds, best_label, best_score, best_auc, best_drop = [],[],[],[],[],0,0
        aucs = []
        for i in range(gran):
            score = [np.nanmean(score[min(int(i / gran * score.shape[0]), score.shape[0] - 1):]) for score in scores]
            fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
            if np.isnan(tpr[0]):
                return -1,-1,-1,-1
            AUC = auc(fpr, tpr)
            aucs.append(float(AUC))
            if float(AUC) >= float(best_auc):
                best_fpr, best_tpr, best_thresholds, best_label, best_score, best_auc, best_drop = fpr, tpr, thresholds, label, score, AUC, i
        return best_fpr, best_tpr, best_score, best_auc
    
    def pltAUC(self, train_result, test_result, method):
        try:
            data = np.array(test_result)
            
            test_car_number = np.unique(test_result['car'])
            #print('test cat number:', test_car_number)
            
            test = defaultdict(list)
            
            for each_car_num in test_car_number:
                this_car = data[np.where(data[:, 0]==each_car_num)]
                if len(this_car) == 0:
                    print(f'{each_car_num} has none data!')
                    continue
                this_car = this_car[np.argsort(this_car[:, -2].astype(float))]
                this_car = this_car[np.where(this_car[:, -1] != np.nan)]
                this_car_score = np.cumsum(this_car[:,-1]) / range(1, this_car.shape[0] + 1)
                
                test[this_car[0][0][0]].append([int(this_car[0][0]), int(this_car[0][1]), this_car_score])
            
            cars, labels, best_scores = [], [], []
            best_aucs = []
            plts = []
            for k, v in test.items():
                car, label, scores = [vt[0] for vt in v], [vt[1] for vt in v], [vt[2] for vt in v]
                best_fpr, best_tpr, best_score, best_auc = self.find_drop(k, method, label, scores)
                if best_auc<0:
                    continue
                cars += car
                labels += label
                best_scores += best_score
                best_aucs.append(best_auc)
                plts.append((best_fpr, best_tpr, f'battery{k}:{best_auc}'))
            l = 5
            print(len(cars), len(best_scores))
            pd.DataFrame({'car': cars, 'label': labels, 'score': best_scores}).to_csv(f'{self.args.result_path}/{method}_scores.csv')
            for i in range(0, len(cars), l):
                for j in range(min(l, len(cars)-i)):
                    print(cars[i+j], labels[i+j], best_scores[i+j], end=' ')
                print()
            best_auc = np.mean(best_aucs)
            print(f'{method} AUC : {best_auc}')
            for fpr, tpr, label in plts:
                plt.plot(fpr, tpr, label=label)
            plt.title(f'{method} AUC : {best_auc}')
            plt.legend()
            plt.savefig(f'{self.args.auc_picture_path}/{method}-{best_auc}.png')
            plt.close('all')
        except:
            traceback.print_exc()

    def eval(self, train_feature_path, test_feature_path, name, mixcar=False):
        train_x, train_label = self.get_feature_label(train_feature_path, mixcar=mixcar, max_group=20000)
        #train_x, train_label = np.zeros((1, 16)), np.zeros((1, 4))
        print("Loading feature is :", train_x.shape)
        print("Loading label is :", train_label.shape)

        test_x, test_label = self.get_feature_label(test_feature_path, mixcar=mixcar, max_group=20000)
        print("Loading test feature is :", test_x.shape)
        print("Loading test label is :", test_label.shape)
        '''
        ban = np.array([x[1][0] for x in test_label])
        print(set(ban))
        for i in set(ban):
            tr = test_label[ban == i][:,-1].astype(float)
            tr = (tr - tr.mean())/tr.std()
            tr = np.array([Decimal(t).to_eng_string() for t in tr])
            test_label[ban == i,-1] = tr
        '''
        # print(test_label)
        
        #Supervised Data
        te_tr_x, te_te_x, te_tr_l, te_te_l = train_test_split(test_x, test_label, test_size=0.5)
        train_x_sv = np.vstack((train_x, te_tr_x))
        train_label_sv = np.vstack((train_label, te_tr_l))
        test_x_sv = te_te_x
        test_label_sv = te_te_l
        
        # print(test_label)

        #Unsupervised Fast
        self.calculate_AUC('rec_error', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_IForest', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), n_jobs=-1)
        #self.calculate_AUC('pyod_COPOD', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), n_jobs=-1)
        #self.calculate_AUC('pyod_LODA', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_INNE', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_PCA', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), tol=0.01)
        
        #Supervised Fast
        #self.calculate_AUC('svm_score', name, train_x_sv.copy(), train_label_sv.copy(), test_x_sv.copy(), test_label_sv.copy())
        #self.calculate_AUC('lgbm_score', name, train_x_sv.copy(), train_label_sv.copy(), test_x_sv.copy(), test_label_sv.copy())
        
        #Unsupervised Slow
        #self.calculate_AUC('pyod_LUNAR', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), verbose=3)
        #self.calculate_AUC('pyod_DIF', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_AnoGAN', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_ALAD', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_ROD', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), parallel_execution=True)
        #self.calculate_AUC('pyod_SO_GAAL', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_MO_GAAL', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy())
        #self.calculate_AUC('pyod_SUOD', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), verbose=True)
        #self.calculate_AUC('pyod_KPCA', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), tol=0.001, n_jobs = -1, sampling = True)
        #self.calculate_AUC('pyod_OCSVM', name, train_x.copy(), train_label.copy(), test_x.copy(), test_label.copy(), kernel='linear')
        
        #Supervised Slow
        #self.calculate_AUC('pyod_XGBOD', name, train_x_sv.copy(), train_label_sv.copy(), test_x_sv.copy(), test_label_sv.copy(), n_jobs=-1)
        
        
    def main(self):
        self.eval(self.args.feature_path, self.args.save_feature_path, 'normal')
        self.eval(self.args.tf_feature_path, self.args.tf_save_feature_path, 'tf', mixcar=True)
        
if __name__ == '__main__':
    import argparse
    import json
    #from anomaly_detection.model import projects

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--modelparams_path', type=str,
                        default=os.path.join( '/home/user/cleantest/2021-12-04-15-19-38/model','model_params.json'))

    args = parser.parse_args()

    with open(args.modelparams_path, 'r') as file:
        p_args = argparse.Namespace()
        model_params=json.load(file)
        p_args.__dict__.update(model_params["args"])
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.modelparams_path)
    print("args", args)
    Evaluate(args).main()

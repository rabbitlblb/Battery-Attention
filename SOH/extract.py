import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate
from model import dataset
from utils import to_var, collate, Normalizer, PreprocessNormalizer, idx2cn, Normalizer
from model import tasks
from utils import get_dataset
import pickle
from collections import defaultdict
import numpy as np
import json

def norm(x, p):
    return (x - p[0]) / np.maximum(np.maximum(1e-4, p[1]), 0.1 * (p[2] - p[3]))

def extract(args, step, dataset, data_loader_battery, model, data_task, feature_path, tf_feature_path, p_bar, noise_scale, variable_length):
    """
    extract features
    """
    iteration = 0
    label_data = tasks.Label(column_name="mileage", training_set=dataset)
    model.eval()
    
    label_loss = defaultdict(list)
    
    #normal extract
    with torch.no_grad():
        for dataloader in data_loader_battery:
            rec_error = 0
            battery_id = 0
            for batch in dataloader:
                batch_ = to_var(batch[0]).float()
                seq_lengths = batch[1]['seq_lengths'] if variable_length else batch[0].shape[1]
                car_numbers = np.zeros(args['battery_num'])
                battery_id = idx2cn(batch[1]['car'][0])
                car_numbers[idx2cn(batch[1]['car'][0])] = 1
                log_p, mean, log_v, z, mean_pred = model(batch_, encoder_filter=data_task.encoder_filter,
                                                            decoder_filter=data_task.decoder_filter,
                                                            seq_lengths=seq_lengths,
                                                            car=car_numbers,
                                                            cn = batch[1]['car'],
                                                            noise_scale=noise_scale)
                
                for i in range(len(mean_pred)):
                    label_loss[batch[1]['car'][0]] += [(float(mean_pred[i]), float(batch[1]['capacity'][i]) / 100.0)]
                
                p_bar.update(1)
                iteration += 1
    
    with open(os.path.join(feature_path, '../result/label_loss.json'), 'w') as f:
        json.dump(label_loss, f)
            
class Extraction:
    """
    feature extraction
    """

    def __init__(self, args, fold_num=0):
        """
        :param project: class model.projects.Project object
        """
        self.args = args
        self.fold_num = fold_num

    def main(self):
        """
        test: normalized test data
        task: task, e.g. EvTask、JeveTask
        model: model
        """
        model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
        with open(model_params_path, 'r') as load_f:
            prams_dict = json.load(load_f)
        step = prams_dict['step']
        model_params = prams_dict['args']
        start_time = time.time()
        self.normalizer = pickle.load(open(os.path.join(self.args.current_model_path, "norm.pkl"), 'rb'))
        sr = self.args.sample_rate
        data_pre = [dataset.load(self.args.train_path, ind_ood_car_dict_path=f'../data/SOH_snippets/changed_files/ind_odd_dict{i}.npz.npy', fold_num=self.fold_num, sample_rate=self.args.sample_rate) for i in  np.arange(1, self.args.battery_num+1)]
        data_pre_train = [dp[0] for dp in data_pre]
        data_pre_val = [dp[1] for dp in data_pre]
        data_pre_test = [sorted(dp[2], key = lambda x: x[1]['car']) for dp in data_pre]
        test_battery, test, data_loader_battery = get_dataset(self, data_pre_test)
        total_iter = np.array([len(dl) for dl in data_loader_battery]).sum()
        
        task = tasks.Task(task_name=model_params["task"], columns=model_params["columns"])

        # load checkpoint
        model_torch = os.path.join(model_params["current_model_path"], "model.torch")
        model = to_var(torch.load(model_torch)).float()
        model.encoder_filter = task.encoder_filter
        model.decoder_filter = task.decoder_filter
        model.noise_scale = model_params["noise_scale"]
        
        print("sliding windows dataset length is: ", len(test))
        #print("model", model)

        # extact feature
        model.eval()
        p_bar = tqdm(total=total_iter, desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        extract(model_params, step, test, data_loader_battery, model, task, model_params["save_feature_path"], model_params["tf_save_feature_path"], p_bar, model_params["noise_scale"],
                model_params["variable_length"])
        p_bar.close()
        print("Feature extraction of all test saved at", model_params["save_feature_path"])
        print("The total time consuming：", time.time() - start_time)


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    Extraction(args).main()

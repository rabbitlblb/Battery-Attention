import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc
from collections import defaultdict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
from model import tasks
from model import dynamic_vae, distinct_model
from model import transformer
from model import transformer_all
from model import transformer_with_battery, transformer_multi_head, transformer_with_BA
from utils import to_var, collate, Normalizer, PreprocessNormalizer, idx2cn, get_dataset
from model import dataset
from extract import extract

class Train_adaption:
    """
    for training
    """

    def __init__(self, args, fold_num=0):
        """
        initialization, load project arguments and create folders
        """
        self.args = args
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        current_path = os.path.join(self.args.save_model_path, time_now+'_fold%d'%fold_num)
        self.mkdir(current_path)
        self.current_path = current_path
        self.current_epoch = 1
        self.step = 1
        self.loss_dict = OrderedDict()
        self.fold_num = fold_num

        loss_picture_path = os.path.join(current_path, "loss")
        feature_path = os.path.join(current_path, "feature")
        tf_feature_path = os.path.join(current_path, "tf_feature")
        current_model_path = os.path.join(current_path, "model")
        save_feature_path = os.path.join(current_path, "mean")
        tf_save_feature_path = os.path.join(current_path, "tf_mean")
        result_path = os.path.join(current_path, "result")
        # create folders
        self.mkdir(loss_picture_path)
        self.mkdir(feature_path)
        self.mkdir(tf_feature_path)
        self.mkdir(current_model_path)
        self.mkdir(result_path)
        self.mkdir(save_feature_path)
        self.mkdir(tf_save_feature_path)

        self.args.loss_picture_path = loss_picture_path
        self.args.feature_path = feature_path
        self.args.tf_feature_path = tf_feature_path
        self.args.result_path = result_path
        self.args.save_feature_path = save_feature_path
        self.args.tf_save_feature_path = tf_save_feature_path
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path

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

    def main(self):
        """
        training
        load training data, preprocessing, create & train & save model, save parameters
        train: normalized data
        model: model
        loss: nll kl label
        rec_error: reconstruct error
        """
        print("Loading data to memory. This may take a few minutes...")
        ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict_all.npz.npy', allow_pickle=True).item()
        self.ind_car_num_list, self.ood_car_num_list = ind_ood_car_dict['ind_sorted'], ind_ood_car_dict['ood_sorted']
        data_pre = [dataset.load(self.args.train_path, ind_ood_car_dict_path=f'../data/SOH_snippets/changed_files/ind_odd_dict{i}.npz.npy', fold_num=self.fold_num, sample_rate=self.args.sample_rate, unknown = self.args.unknown) for i in np.arange(1, self.args.battery_num+1)]
        data_pre_train, data_pre_val, data_pre_test = [dp[0] for dp in data_pre], [dp[1] for dp in data_pre], [dp[2] for dp in data_pre]
        data_pre_all = [dp[0] + dp[1] + dp[2] for dp in data_pre]
        self.normalizer = [Normalizer(dfs=[dpt[i][0] for i in range(len(dpt))],
                                     variable_length=self.args.variable_length, debug=str(dpt[0][1]['car'])[0]) for dpt in data_pre_all]
        
        self.train_battery, self.train, self.data_loader_battery_train = get_dataset(self, data_pre_train, axiv=self.args.adaption)
        if not self.args.adaption:
            self.data_loader_battery_train = [self.data_loader_battery_train]
        self.val_battery, self.val, self.data_loader_battery_val = get_dataset(self, data_pre_val)
        self.test_battery, self.test, self.data_loader_battery_test = get_dataset(self, data_pre_test)
        self.battery_num = min(len(self.data_loader_battery_train), self.args.battery_num)
        self.total_epoch = self.args.epochs * self.battery_num if self.args.adaption else self.args.epochs
        
        print("Data loaded successfully.")

        self.args.columns = torch.load(os.path.join(os.path.dirname(self.args.train_path), "column.pkl"))
        self.data_task = tasks.Task(task_name=self.args.task, columns=self.args.columns)
        params = dict(
            rnn_type=self.args.rnn_type,
            hidden_size=self.args.hidden_size,
            latent_size=self.args.latent_size,
            seq_length=self.args.seq_length,
            battery_num=self.args.battery_num,
            battery_info_rank=self.args.battery_info_size,
            num_layers=self.args.num_layers,
            bidirectional=self.args.bidirectional,
            kernel_size=self.args.kernel_size,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
            variable_length=self.args.variable_length,
            decoder_low_rank=self.args.decoder_low_rank,
            encoder_embedding_size=self.data_task.encoder_dimension,
            decoder_embedding_size=self.data_task.decoder_dimension,
            output_embedding_size=self.data_task.output_dimension)
        
        if self.args.model_type == "transformer_with_battery":
            model = to_var(transformer_with_battery.DynamicVAE(**params)).float()
        elif self.args.model_type == "transformer_with_BA":
            model = to_var(transformer_with_BA.DynamicVAE(**params)).float()
        elif self.args.model_type == "dynamic_vae":
            model = to_var(dynamic_vae.DynamicVAE(**params)).float()
        elif self.args.model_type == "transformer":
            model = to_var(transformer.DynamicVAE(**params)).float()
        elif self.args.model_type == "NC5in5":
            model = to_var(distinct_model.DynamicVAE(dynamic_vae, **params)).float()
        elif self.args.model_type == "transformer5in5":
            model = to_var(distinct_model.DynamicVAE(transformer, **params)).float()
        else:
            model = None

        print("model", model)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        
        time_start = time.time()
        try:
            total_iter = sum([sum([len(dl) for dl in dlb]) for dlb in self.data_loader_battery_train])
            total_iter += sum([len(dl) for dl in self.data_loader_battery_val]) * self.battery_num if self.args.adaption else sum([len(dl) for dl in self.data_loader_battery_val])
            self.label_data = tasks.Label(column_name="capacity", training_set=self.train)
            p_bar = tqdm(total=total_iter * self.args.epochs, desc='training', ncols=160, mininterval=1,
                         maxinterval=10, miniters=1)
            while self.current_epoch <= self.args.epochs:
                log_p, target = self.model_train(model, self.data_loader_battery_train[0], p_bar)
                #self.model_val(model, p_bar, self.data_loader_battery_test, 'test')
                AUCS = self.model_val(model, p_bar, self.data_loader_battery_val, 'val')
                
                p_bar.set_description('Saving Loss - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
                loss_info = {'train_mean_loss': self.train_total_loss / (1 + self.train_iteration), 'train_nll_loss': self.train_total_nll / (1 + self.train_iteration),
                            "train_label_loss": self.train_total_label / (1 + self.train_iteration), "train_kl_loss": self.train_total_kl / (1 + self.train_iteration),
                            'val_mean_loss': self.val_total_loss / (1 + self.val_iteration), 'val_nll_loss': self.val_total_nll / (1 + self.val_iteration),
                            "val_label_loss": self.val_total_label / (1 + self.val_iteration), "val_kl_loss": self.val_total_kl / (1 + self.val_iteration),
                            "label_losses": self.label_losses,
                            "AUCS": AUCS}
                self.save_loss(loss_info, log_p, target)
                p_bar.set_description('Ended - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
                
                self.scheduler.step()
                self.current_epoch += 1
            
            if self.args.adaption:
                battery_params = []
                for name, param in model.named_parameters():
                    if name.split('.')[-1] in ['qlc', 'qrc', 'klc', 'krc', 'vlc', 'vrc', 'olc', 'orc']:
                        battery_params.append(param)
                    else:
                        param.requires_grad = False
                    
                for i in range(1, min(len(self.data_loader_battery_train), self.args.battery_num)):
                    self.optimizer = torch.optim.AdamW(battery_params, lr=self.args.learning_rate, weight_decay=1e-6)
                    self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs,
                                                eta_min=self.args.cosine_factor * self.args.learning_rate)
                    for epoch in range(self.args.epochs):
                        if(len(self.data_loader_battery_train[i]) == 0):
                            self.current_epoch += self.args.epochs
                            break
                        log_p, target = self.model_train(model, self.data_loader_battery_train[i], p_bar)
                        #self.model_val(model, p_bar, self.data_loader_battery_test, 'test')
                        AUCS = self.model_val(model, p_bar, self.data_loader_battery_val, 'val')
                        
                        p_bar.set_description('Saving Loss - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
                        loss_info = {'train_mean_loss': self.train_total_loss / (1 + self.train_iteration), 'train_nll_loss': self.train_total_nll / (1 + self.train_iteration),
                                    "train_label_loss": self.train_total_label / (1 + self.train_iteration), "train_kl_loss": self.train_total_kl / (1 + self.train_iteration),
                                    'val_mean_loss': self.val_total_loss / (1 + self.val_iteration), 'val_nll_loss': self.val_total_nll / (1 + self.val_iteration),
                                    "val_label_loss": self.val_total_label / (1 + self.val_iteration), "val_kl_loss": self.val_total_kl / (1 + self.val_iteration),
                                    "AUCS": AUCS}
                        self.save_loss(loss_info, log_p, target)
                        p_bar.set_description('Ended - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
                        
                        self.scheduler.step()
                        self.current_epoch += 1

            p_bar.close()

        except KeyboardInterrupt:
            print("Caught keyboard interrupt; quit training.")
            pass

        print("Train completed, save information")
        # save model and parameters
        model.eval()
        self.model_result_save(model)
        self.loss_visual()
        print("The total time consuming: ", time.time() - time_start)
        print("All parameters have been saved at", self.args.feature_path)

    def model_train(self, model, dataloaders, p_bar):
        p_bar.set_description('convert model to train - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
        model.train()
        self.train_total_loss, self.train_total_nll, self.train_total_label, self.train_total_kl, self.train_iteration = 0, 0, 0, 0, 0
        for dataloader in dataloaders:
            for batch in dataloader:
                batch_ = to_var(batch[0]).float()
                seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else batch[0].shape[1]
                car_numbers = np.zeros(self.args.battery_num)
                car_numbers[idx2cn(batch[1]['car'][0])] = 1
                log_p, mean, log_v, z, mean_pred = model(batch_,
                                                        encoder_filter=self.data_task.encoder_filter,
                                                        decoder_filter=self.data_task.decoder_filter,
                                                        seq_lengths=seq_lengths,
                                                        car=car_numbers,
                                                        cn = batch[1]['car'],
                                                        noise_scale=self.args.noise_scale)
                target = self.data_task.target_filter(batch_)
                #p_bar.set_description('calculating loss(forward) - Epoch %d/%i' % (self.current_epoch, self.args.epochs))

                nll_loss, kl_loss, kl_weight = self.loss_fn(log_p, target, mean, log_v)
                label_loss = self.label_data.loss(batch, mean_pred, is_mse=True)
                loss = (self.args.nll_weight * nll_loss + self.args.latent_label_weight * label_loss + kl_weight *
                        kl_loss / batch_.shape[0])

                #p_bar.set_description('update params(backward) - Epoch %d/%i' % (self.current_epoch, self.args.epochs))
                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #p_bar.set_description('calculating loss(backward) - Epoch %d/%i' % (self.current_epoch, self.args.epochs))
                # calculate loss
                self.train_total_loss += loss.item()
                self.train_total_nll += nll_loss.item()
                self.train_total_label += label_loss.item()
                self.train_total_kl += kl_loss.item() / batch_.shape[0]
                loss_info = {'mean_loss': self.train_total_loss / (1 + self.train_iteration), 'nll_loss': self.train_total_nll / (1 + self.train_iteration),
                            "label_loss": self.train_total_label / (1 + self.train_iteration), "kl_loss": self.train_total_kl / (1 + self.train_iteration)}
                p_bar.set_postfix(loss_info)
                p_bar.set_description('training %s - Epoch %d/%i' % (batch[1]['car'][0], self.current_epoch, self.total_epoch))

                self.step += 1
                p_bar.update(1)
                self.train_iteration += 1
        return log_p, target
    
    @torch.no_grad()
    def model_val(self, model, p_bar, dataloaders, name):
        p_bar.set_description('convert model to eval - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
        model.eval()
        self.val_total_loss, self.val_total_nll, self.val_total_label, self.val_total_kl, self.val_iteration = 0, 0, 0, 0, 0
        rec_error = defaultdict(list)
        self.label_losses = {}
        for dataloader in dataloaders:
            label_losses = []
            for batch in dataloader:
                batch_ = to_var(batch[0]).float()
                seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else batch[0].shape[1]
                car_numbers = np.zeros(self.args.battery_num)
                car_numbers[idx2cn(batch[1]['car'][0])] = 1
                cn = batch[1]['car'][0]
                log_p, mean, log_v, z, mean_pred = model(batch_,
                                                        encoder_filter=self.data_task.encoder_filter,
                                                        decoder_filter=self.data_task.decoder_filter,
                                                        seq_lengths=seq_lengths,
                                                        car=car_numbers,
                                                        cn = batch[1]['car'],
                                                        noise_scale=self.args.noise_scale)
                target = self.data_task.target_filter(batch_)
                
                nll_loss, kl_loss, kl_weight = self.loss_fn(log_p, target, mean, log_v)
                label_loss = self.label_data.loss(batch, mean_pred, is_mse=True)
                loss = (self.args.nll_weight * nll_loss + self.args.latent_label_weight * label_loss + kl_weight *
                        kl_loss / batch_.shape[0])
                
                #calculate rec error
                mse = torch.nn.MSELoss(reduction='mean')
                for i in range(batch_.shape[0]):
                    rec_error[batch[1]['car'][i]].append(float(mse(log_p[i], target[i])))
                 
                # calculate loss
                self.val_total_loss += loss.item()
                self.val_total_nll += nll_loss.item()
                self.val_total_label += label_loss.item()
                label_losses += [label_loss.item()]
                self.val_total_kl += kl_loss.item() / batch_.shape[0]
                loss_info = {'mean_loss': self.val_total_loss / (1 + self.val_iteration), 'nll_loss': self.val_total_nll / (1 + self.val_iteration),
                            "label_loss": self.val_total_label / (1 + self.val_iteration), "kl_loss": self.val_total_kl / (1 + self.val_iteration)}
                p_bar.set_postfix(loss_info)
                p_bar.set_description('evaling - Epoch %d/%i' % (self.current_epoch, self.total_epoch))

                p_bar.update(1)
                self.val_iteration += 1
            self.label_losses[cn] = np.mean(label_losses)
        
        labels, rec_errors = defaultdict(list), defaultdict(list)
        for k,v in rec_error.items():
            cn = k[0]
            labels[cn].append(0 if k in self.ind_car_num_list else 1)
            rec_errors[cn].append(np.mean(v))
        
        p_bar.set_description('Calculating ROC - Epoch %d/%i' % (self.current_epoch, self.total_epoch))
        AUCS = []
        for cn in labels.keys():
            # print(cn, labels[cn], rec_errors[cn])
            fpr, tpr, thresholds = metrics.roc_curve(labels[cn], rec_errors[cn], pos_label=1)
            AUC = auc(fpr, tpr)
            if np.isnan(tpr).any():
                continue
            tpr = np.nan_to_num(tpr, nan=1)
            plt.plot(fpr, tpr, label=f'battery{cn}-{AUC}')
            AUCS.append(float(AUC))
        plt.title(f'rec error AUC: {np.mean(AUCS)}')
        plt.legend()
        plt.savefig(f'{self.args.loss_picture_path}/rec-error-{self.current_epoch}-{name}-{np.mean(AUCS)}).png')
        plt.close('all')
        return AUCS
        
    def model_result_save(self, model):
        """
        save model
        :param model: vae or transformer
        :return:
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict,
                        'step': self.step}
        with open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)
        model_path = os.path.join(self.args.current_model_path, "model.torch")
        torch.save(model, model_path)
        norm_path = os.path.join(self.args.current_model_path, "norm.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump(self.normalizer, f)

    def loss_fn(self, log_p, target, mean, log_v):
        """
        loss function
        :param log_p: transformed prediction
        :param target: target
        :param mean:
        :param log_v:
        :return: nll_loss, kl_loss, kl_weight
        """
        nll = torch.nn.SmoothL1Loss(reduction='mean')
        nll_loss = nll(log_p, target)
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
        kl_weight = self.kl_anneal_function()
        return nll_loss, kl_loss, kl_weight

    def kl_anneal_function(self):
        """
        anneal update function
        """
        if self.args.anneal_function == 'logistic':
            return self.args.anneal0 * float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif self.args.anneal_function == 'linear':
            return self.args.anneal0 * min(1, self.step / self.args.x0)
        else:
            return self.args.anneal0

    def loss_visual(self):
        """
        draw loss curves
        """
        if self.args.epochs == 0:
            return
        x = list(self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        train_mean_loss = df_loss['train_mean_loss'].values.astype(float)
        train_nll_loss = df_loss['train_nll_loss'].values.astype(float)
        train_label_loss = df_loss['train_label_loss'].values.astype(float)
        train_kl_loss = df_loss['train_kl_loss'].values.astype(float)
        val_mean_loss = df_loss['val_mean_loss'].values.astype(float)
        val_nll_loss = df_loss['val_nll_loss'].values.astype(float)
        val_label_loss = df_loss['val_label_loss'].values.astype(float)
        val_kl_loss = df_loss['val_kl_loss'].values.astype(float)
        AUCS = df_loss['AUCS'].values

        plt.figure(dpi=300, figsize=(30,30))
        plt.subplot(3, 1, 1)
        plt.plot(x, train_mean_loss, 'r.-', label='train_mean_loss')
        plt.plot(x, val_mean_loss, 'b.-', label='val_mean_loss')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        label = 1
        for i in range(len(AUCS[0])):
            AUC = [x[i] for x in AUCS]
            plt.plot(x, np.array(AUC), label=f'battery{label}')
            label += 1
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(x, train_nll_loss, 'ro-', label='train_nll_loss')
        plt.plot(x, val_nll_loss, 'bo-', label='val_nll_loss')
        plt.legend()

        plt.subplot(3, 3, 8)
        plt.plot(x, train_label_loss, 'ro-', label='train_label_loss')
        plt.plot(x, val_label_loss, 'bo-', label='val_label_loss')
        plt.legend()

        plt.subplot(3, 3, 9)
        plt.plot(x, train_kl_loss, 'ro-', label='train_kl_loss')
        plt.plot(x, val_kl_loss, 'bo-', label='val_kl_loss')
        plt.legend()
        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')

    def save_loss(self, loss_info, log_p, target):
        """
        save loss
        """
        self.loss_dict[str(self.current_epoch)] = loss_info
        with open(os.path.join(self.args.current_model_path, 'loss.json'), "w") as f:
            json.dump(self.loss_dict, f, indent=4)
        n_image = log_p.shape[-1]
        for i in range(n_image):
            plt.subplot(n_image, 1, i + 1)
            plt.plot(log_p[0, :, i].cpu().detach().numpy(), 'y',
                     label='lp-' + str(self.current_epoch))
            plt.plot(target[0, :, i].cpu().detach().numpy(), 'c',
                     label='tg-' + str(self.current_epoch))
            plt.legend()
        loss_path = os.path.join(self.args.loss_picture_path, "%i_epoch.jpg" % self.current_epoch)
        plt.savefig(loss_path)
        plt.close('all')

    def getmodelparams(self):
        return os.path.join(self.args.current_model_path, 'model_params.json')



if __name__ == '__main__':
    import argparse

    #from anomaly_detection.model import projects

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), './params.json'))

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    print("args", args)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, plot_chart, test_params_flop, plot_scatter, plot_cm, plot_heatmap
from utils.metrics import metrics, R_LAST

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import math
import random
import pandas
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            print('max iterations:', train_steps)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\r\titers: {0}, epoch: {1} | loss: {2:.7f}\tspeed: {3:.4f}s/iter; left time: {4:.4f}s".format(i + 1, epoch + 1, loss.item(), speed, left_time ), end='')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs_last = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        test_num = 0

        means, _ , scales = test_data.indices_scaler()
        mean = means[-1]
        scale = scales[-1]

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #outputs = outputs.detach().cpu().numpy()
                #batch_y = batch_y.detach().cpu().numpy()

                # inverse standardization
                pred = outputs * scale + mean 
                true = batch_y * scale + mean 
                batch_x = batch_x * scale + mean

                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()

                batch_x_last = batch_x[:, -1, -1].detach().cpu().numpy() #(1024)

                preds.append(pred)
                trues.append(true)
                inputs_last.append(batch_x_last)

                if i % 20 == 0 and not self.args.visual_samples < test_num:
                    test_num += 1

                    input_data = batch_x.detach().cpu().numpy()
                    input_and_true = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                    input_and_pred = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)

                    plot_chart(input_and_true, input_and_pred, self.args.seq_len, self.args.pred_len, os.path.join(folder_path, str(i) + '.pdf'))
                    test_num += 1

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        trues = np.array(trues) #(i, 1024, 60, 1)
        preds = np.array(preds) #(i, 1024, 60, 1)
        inputs_last = np.array(inputs_last) #(i, 1024)
        #print(inputs_last.shape)
        #print(trues.shape)

        max_i = np.unravel_index(np.argmax(np.squeeze(trues) - inputs_last[...,np.newaxis]), preds.shape)
        #print(trues[max_i])
        #print(inputs_last[max_i[0],max_i[1]])

        #混同行列
        trues_first =  np.ravel(trues[:, :,  0, 0])
        trues_last  =  np.ravel(trues[:, :, -1, 0])
        preds_first =  np.ravel(preds[:, :,  0, 0])
        preds_last  =  np.ravel(preds[:, :, -1, 0])

        updown_gt_first  = np.where(np.ravel(inputs_last) < trues_first, 0, 1) # Up: 0, Down: 1
        updown_gt_last   = np.where(np.ravel(inputs_last) < trues_last, 0, 1)
        updown_pd_first  = np.where(np.ravel(inputs_last) < preds_first, 0, 1)
        updown_pd_last   = np.where(np.ravel(inputs_last) < preds_last, 0, 1)

        plot_cm(updown_gt_first, updown_pd_first, title='1m later', index=["Up", "Down"], columns=["Up", 'Down'], filename=os.path.join(folder_path, 'cm_first.png'))
        plot_cm(updown_gt_last,  updown_pd_last,  title='1h later', index=["Up", "Down"], columns=["Up", 'Down'], filename=os.path.join(folder_path, 'cm_last.png'))

        updown_acc_first = accuracy_score(updown_gt_first, updown_pd_first)
        updown_acc_last  = accuracy_score(updown_gt_last, updown_pd_last)
        print('updown_acc_first:{}, updown_acc_last:{}'.format(updown_acc_first, updown_acc_last))

        pd_eval = preds.reshape(-1, preds.shape[-2]) #(i*1024, 60, 1)
        gt_eval = trues.reshape(-1, trues.shape[-2]) #(i*1024, 60, 1)

        # 散布図
        # 生のpriceの相関係数を出しても意味ない
        #plot_scatter(gt_eval, pd_eval, sampling=True, title='raw value', r=r_last, filename=os.path.join(folder_path, 'scatter_raw_value.png'))

        # 最後の値との差分
        inputs_last = inputs_last[..., np.newaxis]
        gt_diff_from_last = pd_eval - np.ravel(inputs_last)[...,np.newaxis]
        pd_diff_from_last = gt_eval - np.ravel(inputs_last)[...,np.newaxis]
        r_diff_last = R_LAST(gt_diff_from_last, pd_diff_from_last)

        plot_scatter(gt_diff_from_last, pd_diff_from_last, sampling=True, title='diff', r=r_diff_last, filename=os.path.join(folder_path,'scatter_diff_last.png'))

        # 最後の値からの変化率 = 差分/最後の値
        gt_change_from_last = gt_diff_from_last / np.ravel(inputs_last)[...,np.newaxis]
        pd_change_from_last = pd_diff_from_last / np.ravel(inputs_last)[...,np.newaxis]
        r_change_from_last = R_LAST(gt_change_from_last, pd_change_from_last)

        # metrics
        mae, mse, rmse, mape, mspe, rse, corr, r_first, r_last = metrics(gt_change_from_last, pd_change_from_last)
        print('mse:{}, mae:{}, mape:{}, r_last:{}'.format(mse, mae, mape, r_last))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        print('r(change_ratio) 1h later :',r_change_from_last)
        plot_scatter(gt_change_from_last, pd_change_from_last, sampling=True, title='change_ratio', r=r_change_from_last, filename=os.path.join(folder_path,'scatter_change_last.png'))

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()

        means, _ , scales = test_data.indices_scaler()
        mean = means[-1]
        scale = scales[-1]

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(pred_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        """
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        """

        return

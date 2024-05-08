from kan import KAN, create_dataset
import pandas as pd
import numpy as np
import torch

def create_dataset(ts_name, tr_names, yname, n_train):
    xnames = ['C (GPa)', 'dP/dh (N/m)', 'Wp/Wt']
    name = '../data/' + ts_name + '.csv'
    datatr = pd.read_csv(name)
    x = datatr.loc[:, xnames].values
    y = datatr.loc[:, yname].values
    np.random.seed()
    tr_ind = np.random.choice(x.shape[0], size=n_train, replace=False)
    x_train = x[tr_ind]
    y_train = y[tr_ind]
    if tr_names == ts_name:
        x_test = np.delete(x, tr_ind, axis=0)
        y_test = np.delete(y, tr_ind, axis=0)
    else:
        name = '../data/' + tr_names + '.csv'
        datats = pd.read_csv(name)
        x = datats.loc[:, xnames].values
        y = datats.loc[:, yname].values
        x_test = x
        y_test = y
    dataset = {}
    dataset['train_input'] = torch.from_numpy(x_train).float()
    dataset['test_input'] = torch.from_numpy(x_test).float()
    dataset['train_label'] = torch.from_numpy(y_train).float()
    dataset['test_label'] = torch.from_numpy(y_test).float()
    return dataset

def KAN_single(ts_name, tr_names, yname, n_train, size=10, width=[3,20,1], grid=20, k=5):
    loss = np.zeros(size)
    for i in range(size):
        while loss[i] == 0 or np.isnan(loss[i]):
            model = KAN(width=width, grid=grid, k=k)
            dataset = create_dataset(ts_name, tr_names, yname, n_train)
            lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
            loss[i] = lost_list['test_loss'][0]
    print('loss ', np.mean(loss), ' ', np.std(loss))
    return loss

def KAN_muilti(ts_name, tr_hi, tr_lo, yname, n_train, size=10, width=[3,20,1], grid=20, k=5):
    loss = np.zeros(size)
    for i in range(size):
        while loss[i] == 0 or np.isnan(loss[i]):
            lower = 0
            while lower == 0 or np.isnan(lower):
                model = KAN(width=width, grid=grid, k=k)
                dataset = create_dataset(tr_lo, tr_lo, yname, n_train)
                lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
                lower = lost_list['test_loss'][0]
            dataset = create_dataset(ts_name, tr_hi, yname, n_train)
            lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = True)
            loss[i] = lost_list['test_loss'][0]
    print('loss ', np.mean(loss), ' ', np.std(loss))
    return loss






loss = KAN_single('TI33_25', 'TI33_25', 'sy (GPa)', 20)
loss = KAN_single('TI33_25', 'TI33_25', 'sy (GPa)', 20)
loss = KAN_muilti('TI33_25', 'TI33_25', '2D_70_quad', 'sy (GPa)', 20)
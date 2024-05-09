from kan import KAN, create_dataset
import pandas as pd
import numpy as np
import torch

def create_dataset(ts_name, tr_names, yname, n_train):
    xnames = ['C (GPa)', 'dP/dh (N/m)', 'Wp/Wt']
    name = '../data/' + ts_name + '.csv'
    datatr = pd.read_csv(name)
    x = datatr.loc[:, xnames].values
    if yname == 'Er (GPa)' and ts_name[1] == 'D':
        y = EtoEr(datatr.loc[:, 'E (GPa)'].values, datatr.loc[:, 'nu'].values)
    else:
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

def KAN_one(ts_name, tr_names, n_train, yname, size=10, width=[3,7,7,1], grid=20, k=5):
    loss = np.zeros(size)
    for i in range(size):
        while loss[i] == 0 or np.isnan(loss[i]):
            model = KAN(width=width, grid=grid, k=k, bias_trainable=True, sp_trainable=True, sb_trainable=True)
            dataset = create_dataset(ts_name, tr_names, yname, n_train)
            lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
            loss[i] = lost_list['test_loss'][0]
    print('loss ', np.mean(loss), ' ', np.std(loss))
    with open('output.txt', 'a') as f:
        f.write('KAN_two ' + yname + ' ' + str(np.mean(loss)) + ' ' + str(np.std(loss)) + ' ' + t2s(ts_name) + ' ' + t2s(tr_names) + ' ' + str(n_train))
    return

def KAN_two(ts_name, tr_hi, n_hi, tr_lo, n_lo, yname, size=10, width=[3,7,7,1], grid=20, k=5):
    loss = np.zeros(size)
    for i in range(size):
        while loss[i] == 0 or np.isnan(loss[i]):
            lower = 0
            while lower == 0 or np.isnan(lower):
                model = KAN(width=width, grid=grid, k=k, bias_trainable=True, sp_trainable=True, sb_trainable=True)
                dataset = create_dataset(tr_lo, tr_lo, yname, n_lo)
                lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
                lower = lost_list['test_loss'][0]
            dataset = create_dataset(ts_name, tr_hi, yname, n_hi)
            lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
            loss[i] = lost_list['test_loss'][0]
    print('loss ', np.mean(loss), ' ', np.std(loss))
    with open('output.txt', 'a') as f:
        f.write('KAN_two ' + yname + ' ' + str(np.mean(loss)) + ' ' + str(np.std(loss)) + ' ' + t2s(ts_name) + ' ' + t2s(tr_hi) + ' ' + str(n_hi) + ' ' + t2s(tr_lo) + ' ' + str(n_lo))
    return

def KAN_three(ts_name, tr_highest, n_highest, tr_hi, n_hi, tr_lo, n_lo, yname, size=10, width=[3,7,7,1], grid=20, k=5):
    loss = np.zeros(size)
    for i in range(size):
        while loss[i] == 0 or np.isnan(loss[i]):
            lower = 0
            while lower == 0 or np.isnan(lower):
                model = KAN(width=width, grid=grid, k=k, bias_trainable=True, sp_trainable=True, sb_trainable=True)
                dataset = create_dataset(tr_lo, tr_lo, yname, n_lo)
                lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
                dataset = create_dataset(tr_hi, tr_hi, yname, n_hi)
                lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
                lower = lost_list['test_loss'][0]
                lower = lost_list['test_loss'][0]
            dataset = create_dataset(ts_name, tr_highest, yname, n_highest)
            lost_list = model.train(dataset, opt='LBFGS', steps=3, update_grid = False)
            loss[i] = lost_list['test_loss'][0]
    print('loss ', np.mean(loss), ' ', np.std(loss))
    with open('output.txt', 'a') as f:
        f.write('KAN_two ' + yname + ' ' + str(np.mean(loss)) + ' ' + str(np.std(loss)) + ' ' + t2s(ts_name) + ' ' + t2s(tr_highest) + ' ' + str(n_highest) + ' ' + t2s(tr_hi) + ' ' + str(n_hi) + ' ' + t2s(tr_lo) + ' ' + str(n_lo))
    return

def EtoEr(E, nu):
    nu_i, E_i = 0.0691, 1143
    return 1 / ((1 - nu ** 2) / E + (1 - nu_i ** 2) / E_i)

def t2s(names):
    if type(names) is str:
        return names
    else:
        return '[' + ','.join(names) + ']'






KAN_one('TI33_500', 'TI33_25', 10, 'Er (GPa)')
KAN_two('TI33_500', 'TI33_25', 10, '3D_quad', 10, 'Er (GPa)')
KAN_three('TI33_500', 'TI33_25', 10, '3D_quad', 10, '2D_70_quad', 10, 'Er (GPa)')
KAN_three('TI33_500', '2D_70_quad', 10, '3D_quad', 10, 'TI33_25', 10, 'Er (GPa)')
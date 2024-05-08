from kan import KAN, create_dataset
import pandas as pd
import numpy as np
import torch
torch.device='mps'



def create_dataset(ts_name, tr_name, yname, n_train):
    x_names = ['C (GPa)', 'dP/dh (N/m)', 'Wp/Wt']
    name = '../data/' + ts_name + '.csv'
    data = pd.read_csv(name)
    x = data.loc[:, x_names].values
    y = data.loc[:, yname].values
    np.random.seed()
    tr_ind = np.random.choice(x.shape[0], size=n_train, replace=False)
    print('indices', tr_ind)
    name = tr_name
    data = pd.read_csv(name)
    x = data.loc[:, x_names].values
    y = data.loc[:, yname].values
    x_train = x[tr_ind]
    y_train = y[tr_ind]
    x_test = np.delete(x, tr_ind, axis=0)
    y_test = np.delete(y, tr_ind, axis=0)
    dataset = {}
    dataset['train_input'] = torch.from_numpy(x_train).float()
    dataset['test_input'] = torch.from_numpy(x_test).float()
    dataset['train_label'] = torch.from_numpy(y_train).float()
    dataset['test_label'] = torch.from_numpy(y_test).float()
    return dataset

model = KAN(width=[3,20,1], grid=20, k=5)
dataset = create_dataset('TI33_25', 'TI33_25', 'Er (GPa)', 20)
train_loss = model.train(dataset, opt="LBFGS", steps=3, update_grid = False)
dataset = create_dataset('TI33_25', 'TI33_250', 'Er (GPa)', 20)
train_loss = model.train(dataset, opt="LBFGS", steps=3, update_grid = False)
dataset = create_dataset('TI33_25', 'TI33_500', 'Er (GPa)', 20)
train_loss = model.train(dataset, opt="LBFGS", steps=3, update_grid = False)

model = KAN(width=[3,20,20,1], grid=20, k=5)
dataset = create_dataset('TI33_25', 'TI33_500', 'Er (GPa)', 20)
train_loss = model.train(dataset, opt="LBFGS", steps=3, update_grid = False)

model = KAN(width=[3,20,1], grid=20, k=5)
dataset = create_dataset('TI33_25', 'TI33_25', 'Er (GPa)', 20)
model.train(dataset, opt="LBFGS", steps=3, update_grid = False)

model2 = KAN(width=[3,20,1], grid=20, k=5)
dataset = create_dataset('TI33_25', 'TI33_25', 'sy (GPa)', 20)
model2.train(dataset, opt="LBFGS", steps=3, update_grid = True)

model = KAN(width=[3,20,1], grid=20, k=5)
dataset = create_dataset('TI33_25', 'TI33_25', 'sy (GPa)', 20)
model.train(dataset, opt="LBFGS", steps=3, update_grid = False)


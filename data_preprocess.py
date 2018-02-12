import re
import numpy as np
from datetime import timedelta
from dateutil import parser
import pandas as pd
import os

def make_data(data_dir):
    df = pd.read_csv('{}/EVO_Training_Data_Original.csv'.format(data_dir), index_col=False)
    time = df.apply(lambda x: str(int(x['date '])) +' ' +str(int(x['H'])) + ':' +str(int(x['M'])) + ':' + str(int(x['S'])), axis=1)
    datetime = time.apply(parser.parse)
    df['datetime'] = datetime

    row_list = list()
    step_size = 5
    skip_num = 0
    for i in range(len(df) - step_size):
        skip_time = False
        tmp = df['datetime'].iloc[i:i+step_size]
        cap_nom = df['cap_nom'].iloc[i:i+step_size]
        first = tmp.iloc[0]
        if len(cap_nom.unique()) == 1:
            for idx, j in tmp.iteritems():
                if j - first > timedelta( minutes=12):
                    skip_num +=1
                    skip_time = True
                    break
                else:
                    first = j
            if not skip_time:
                row_list.append(df.iloc[i:i+step_size])
    print('skipped :', skip_num)

    columns = ['mdot', 'Tod', 'RHod', 'mode', 'Tid', 'Vidu', 'cap_nom', 'Qsens', 'Qlat']
    data = np.zeros((len(row_list * step_size), len(columns)))
    for idx, value in enumerate(row_list):
        data[idx * (step_size) :(idx + 1)* (step_size)] = value[columns].values

    data = data.reshape(-1, step_size, len(columns))

    target_columns = ['Load_s', 'Load_l']
    target = np.zeros((len(row_list) * step_size, len(target_columns)))
    for idx, value in enumerate(row_list):
        target[idx * (step_size) :(idx + 1)* (step_size)] = value[target_columns].values

    target = target.reshape(-1, step_size, len(target_columns))

    target = target[:, step_size - 1, :]

    np.savez('{}/data'.format(data_dir), X=data, y=target)

def make_data_single(data_dir):
    seed = int(re.search('seed(\d)', data_dir).group(1))
    data = pd.read_csv('data/training_data_all_interval_120_mode.csv')

    float_vars = ['mdot', 'Tod', 'RHod', 'Tid', 'Vidu', 'cap_nom', 'Qsens', 'Qlat']
    int_vars = ['mode']
    target_vars = ['Lsens', 'Llat']

    # data['mode'] = data.apply(lambda x: np.where(x[mode_var] == 1)[0][0], axis=1)

    np.random.seed(seed)

    idx_permute = np.random.permutation(data.index)
    trn_ratio = int(len(idx_permute) * 0.9)
    trn_idx = idx_permute[:trn_ratio]
    test_idx = idx_permute[trn_ratio:]

    trn_data = data.loc[trn_idx]
    test_data = data.loc[test_idx]

    os.makedirs('data/seed{}_trn'.format(seed))
    os.makedirs('data/seed{}_test'.format(seed))

    np.savez('data/seed{}_trn/data.npz'.format(seed), float_vars=trn_data[float_vars],
             int_vars=trn_data[int_vars], target_vars=trn_data[target_vars])
    np.savez('data/seed{}_test/data.npz'.format(seed), float_vars=test_data[float_vars],
             int_vars=test_data[int_vars], target_vars=test_data[target_vars])
    np.savez('data/seed{}_trn/idx_permute.npz'.format(seed), trn_idx = trn_idx,
             test_idx=test_idx)
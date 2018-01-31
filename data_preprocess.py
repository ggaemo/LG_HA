import numpy as np
from datetime import timedelta
from dateutil import parser
import pandas as pd


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
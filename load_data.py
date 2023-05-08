# -*- coding: utf-8 -*-

import numpy as np
def load_data(args):

    data = np.load('data_test.npy', allow_pickle=True).item()

    train_data = data['1'] + data['3'] + data['4'] + data['5'] + data['7'] + data['8'] + data['12'] + data['13'] + data['20'] + \
                 data['22'] + data['23'] + data['24'] + data['30'] + data['32'] + data['33'] + data['35']
                 
    dev_data = data['37'] + data['41']

    test_data = data['14'] + data['16'] + data['18'] + data['19']

    return train_data, dev_data, test_data
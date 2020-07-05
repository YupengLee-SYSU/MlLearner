import numpy as np
from numpy import ndarray


def combine_data_target(data:ndarray, target:ndarray)-> ndarray:
    o_dim = 1 if len(target.shape) == 1 else target.shape[1]
    o_data = np.zeros([data.shape[0], data.shape[1]+o_dim])
    o_data[:, 0: data.shape[1]] = data
    o_data[:, data.shape[1]: data.shape[1]+o_dim] = target.reshape(
        [data.shape[0], int(target.size/data.shape[0])])
    return o_data

def normalize(i_data:ndarray)-> ndarray:
    i_mean = np.mean(i_data)
    i_var = np.var(i_data)
    return (i_data-i_mean)/i_var

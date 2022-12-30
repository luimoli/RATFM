import numpy as np

def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))

def get_MAPE(pred, real):
    mapes = []
    for i in range(len(pred)):
        gt_sum = np.sum(np.abs(real[i]))
        er_sum = np.sum(np.abs(real[i] - pred[i]))
        mapes.append(er_sum / gt_sum)
    return np.mean(mapes)

def get_p2p_MAPE(pred, real):
    ori_real = real.copy()
    epsilon = 1 # if use small number like 1e-5 resulting in very large value
    real[real == 0] = epsilon 
    return np.mean(np.abs(ori_real - pred) / np.abs(real))

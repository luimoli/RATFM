import numpy as np
import seaborn as sns
import torch
import os
import cv2


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


def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)


def get_topk_vector(arr, mask):
    '''
    this function gets the reshaped vector through a 2d mask.
    arr.shape : [B,C,H,W]
    mask.shape: [H,W]
    return [B,C,num_of_true_grids]
    '''
    b,c,h,w = arr.shape
    b_list = []
    for i in range(b):
        c_list = []
        for j in range(c):
            c_list.append(np.expand_dims(arr[i][j][mask], 0))
        b_list.append(np.expand_dims(np.concatenate(c_list,0), 0))
    res = np.concatenate(b_list, 0)
    return res






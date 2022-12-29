import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
#if float(torchvision.__version__[:3]) < 0.7:
#    from torchvision.ops import _new_empty_tensor
#    from torchvision.ops.misc import _output_size


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))


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


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # # TODO make this more general
    # if tensor_list[0].ndim == 3:
    #     if torchvision._is_tracing():
    #         # nested_tensor_from_tensor_list() does not export well to ONNX
    #         # call _onnx_nested_tensor_from_tensor_list() instead
    #         return _onnx_nested_tensor_from_tensor_list(tensor_list)

    # # TODO make it support different-sized images
    # max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    # batch_shape = [len(tensor_list)] + max_size
    batch_shape = tensor_list.shape
    b, c, h, w = tensor_list.shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    # else:
    #     raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)



def get_patch_all(fine_arr, n): 
    '''
    arr is some feature map [b,c,h,w] like [batch,2,16,16].
    n is crop size ---arr will be crop into n*n
    '''
    arr = fine_arr
    assert arr.shape[2] == arr.shape[3]
    sep = arr.shape[2] // n  # sep = patch size
    patch_list = []
    for i in range(n):
        for j in range(n):
            patch_list.append(arr[:,:, i*sep:(i+1)*sep, j*sep:(j+1)*sep])

    # patches = torch.cat(patch_list, dim=0)
    # return patches       #return the stacked patches
    return patch_list  #return a list which contains patches

def recover_patch_from_list(fine_patch_list, n):
    '''
    fine_patch_list is a list containing patches that could be recovered to orginal size.
    n is crop size ---original arr is stacked by n*n mini areas.
    '''
    assert len(fine_patch_list) == n*n
    row_list=[]
    for i in range(0,n*n,n):
        row_list.append(torch.cat(fine_patch_list[i:i+n], 3)) 
    assert len(row_list) == n
    finemap = torch.cat(row_list, 2) 
    return finemap


def get_patch_all_arr(fine_arr, n): 
    '''
    arr is some feature map [b,c,h,w] like [batch,2,16,16].
    n is crop size ---arr will be crop into n*n
    '''
    arr = fine_arr
    assert arr.shape[2] == arr.shape[3]
    sep = arr.shape[2] // n  # sep = patch size
    patch_list = []
    for i in range(n):
        for j in range(n):
            patch_list.append(arr[:,:, i*sep:(i+1)*sep, j*sep:(j+1)*sep])
    patches = torch.cat(patch_list, dim=0)
    return patches       #return the stacked patches

def recover_patch_from_arr(arr_patches, n):
    '''
    arr_patches is the stacked patches [batch_size*n*n, c, h/n, w/n].
    n is crop size ---arr is stacked by n*n mini areas.
    '''
    arr = arr_patches
    split_size = arr.shape[0] // (n*n) #crop the pic into n*n pieces, dim0= batch_size*n*n, split_size=batch_size
    fine_patch_list = torch.split(arr, split_size, dim=0) 
    row_list=[]
    for i in range(0,n*n,n):
        row_list.append(torch.cat(fine_patch_list[i:i+n], 3)) 
    finemap = torch.cat(row_list, 2) 
    return finemap


def get_visual(arrs,save_path, suffix, error_range_list):
    '''
    visualization a batch 
    '''
    for i in range(len(arrs)):
        f, ax = plt.subplots(figsize=(15, 15))
        f_n, ax_n = plt.subplots(figsize=(15, 15))
        f_m, ax_m = plt.subplots(figsize=(15, 15))
        arr = arrs[i][0]
        arr_n = arr.copy()
        arr_m = arr.copy()
        if suffix[:2] == 'er':
            res = sns.heatmap(arr,vmin=0,vmax=error_range_list[0],cmap='YlGnBu_r',annot=False, ax=ax,cbar=True, xticklabels=False,yticklabels=False,square=True)
            # res.figure.axes[-1].yaxis.label.set_size(25)
            cbar = res.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            res_n = sns.heatmap(arr_n,vmin=0,vmax=error_range_list[1],cmap='YlGnBu_r',annot=False, ax=ax_n,cbar=True, xticklabels=False,yticklabels=False,square=True)
            cbar_n = res_n.collections[0].colorbar
            cbar_n.ax.tick_params(labelsize=20)
            res_m = sns.heatmap(arr_m,vmin=0,vmax=error_range_list[2],cmap='YlGnBu_r',annot=False, ax=ax_m,cbar=True, xticklabels=False,yticklabels=False,square=True)
            cbar_m = res_n.collections[0].colorbar
            cbar_m.ax.tick_params(labelsize=20)

            fig = res.get_figure()
            fig.savefig(save_path + str(i)+ '_etop_' + str(error_range_list[0]) + '.jpg', bbox_inches="tight", pad_inches=0.0)
            fig_n = res_n.get_figure()
            fig_n.savefig(save_path + str(i)+ '_etop_' + str(error_range_list[1]) + '.jpg', bbox_inches="tight", pad_inches=0.0)
            fig_m = res_m.get_figure()
            fig_m.savefig(save_path + str(i)+ '_etop_' + str(error_range_list[2]) + '.jpg', bbox_inches="tight", pad_inches=0.0)
        else:
            res = sns.heatmap(arr,vmin=0,vmax=np.max(arr),cmap='RdYlGn_r',annot=False, ax=ax,cbar=False, xticklabels=False,yticklabels=False,square=True)

            res_n = sns.heatmap(arr_n,vmin=0,vmax=np.max(arr),cmap='RdYlGn_r',annot=False, ax=ax_n,cbar=True, xticklabels=False,yticklabels=False,square=True)
            cbar_n = res_n.collections[0].colorbar
            cbar_n.ax.tick_params(labelsize=20)

            fig = res.get_figure()
            fig.savefig(save_path + str(i)+'.jpg', bbox_inches="tight", pad_inches=0.0)
            fig_cbar = res_n.get_figure()
            fig_cbar.savefig(save_path + str(i)+'_cbar.jpg', bbox_inches="tight")


def gene_visual(tensor_list, i, visual_path, error_range_list):
    '''
    tensor list: [(tensor, suffix),(tensor, suffix)...] 
    suffix: gt/pr/...
    i: iteration number  (0)
    visual_path: where to save visualization pics
    '''
    for tensor, suffix in tensor_list:
        get_visual(np.abs(tensor.cpu().detach().numpy()), os.path.join(visual_path,str(i)+'_'+ suffix+ '_'), suffix, error_range_list)



def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def log_losses_tensorboard(writer, current_losses, i_iter):
    dic={}
    for loss_name, loss_value in current_losses.items():
        dic.update({f'{loss_name}':to_numpy(loss_value)})
    
    writer.add_scalars(f'train_loss',dic,i_iter)
    # writer.add_scalar(f'loss/{loss_name}', to_numpy(loss_value), i_iter)

def log_vals_tensorboard(writer, loss_val, val_iter):
    writer.add_scalar(f'val/loss_val', loss_val, val_iter)

def print_losses(current_losses, epoch, i_iter, file_):
    list_strings = []
    list_draw = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.5f} ')
        list_draw.append(f'{to_numpy(loss_value):.5f}')
    full_string = ' '.join(list_strings)
    full_draw = ' '.join(list_draw)
    # tqdm.write(f'iter = {i_iter} {full_string}')
    print(f'epoch= {epoch} iter= {i_iter} {full_string}', file=file_)



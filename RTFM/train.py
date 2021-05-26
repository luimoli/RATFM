import os
import sys
import warnings
import numpy as np
import argparse
import warnings
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .utils.metrics import *
from .utils.misc import *

from .utils.data_sr_road import get_dataloader_sr
from .models.RTFM import Mixmap

from .modules.transformer import build_transformer
from .modules.position_encoding import build_position_encoding

from tqdm import tqdm

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=60,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_residuals', type=int, default=16,
                    help='number of residual units')
parser.add_argument('--base_channels', type=int, default=128, 
                    help='number of feature maps')
parser.add_argument('--road_channels', type=int,
                    default=1, help='number of feature maps')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--img_width', type=int, default=64,
                    help='image width')
parser.add_argument('--img_height', type=int, default=64,
                    help='image height') 
parser.add_argument('--channels', type=int, default=2,   # (inflow + outflow) | XiAn & Chengdu: 2 channel | beijing:1 channel
                    help='number of flow image channels')
parser.add_argument('--folder_name', type=str, default='xian',
                    help='folder_name to save models ')                   
parser.add_argument('--dataset_name', type=str, default='xian',  # xian | cdu | P1
                    help='which dataset to use')
parser.add_argument('--city', type=str, default='xian', # cdu | xian | P1 | no  
                    help='which city_road_map to use')
parser.add_argument('--run_num', type=int, default=0,
                    help='save model folder serial number')
# * Transformer
parser.add_argument('--enc_layers', default=2, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=2, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int, 
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=128, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")


opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
warnings.filterwarnings('ignore')
# path for saving model---------------------------------------------
while os.path.exists('saved_model/{}/{}-{}-{}-{}'.format(opt.folder_name,
                                             opt.n_residuals,
                                             opt.base_channels,
                                             opt.ext_flag,
                                             opt.run_num)): opt.run_num+=1
save_path = 'saved_model/{}/{}-{}-{}-{}'.format(opt.folder_name,
                                             opt.n_residuals,
                                             opt.base_channels,
                                             opt.ext_flag,
                                             opt.run_num)
print(save_path)
os.makedirs(save_path, exist_ok=True)

# visual_path = os.path.join(save_path,'visual_train')
# os.makedirs(visual_path, exist_ok=True)
# visual_valid_path = os.path.join(save_path,'visual_valid')
# os.makedirs(visual_valid_path, exist_ok=True)

# vis = visdom.Visdom(env="visual loss", port=10041)
# TENSORBOARD_LOGDIR = f'{save_path}/tensorboards'
# log_output = open("%s/log.txt" % save_path, 'w')
# # log_draw = open("%s/draw.txt" % save_path, 'w')
# writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# # # initial model
transformer = build_transformer(opt, opt.hidden_dim)
position_embedding = build_position_encoding(opt, opt.hidden_dim)
model = Mixmap(position_embedding, transformer, 
                in_channels=opt.channels,
                out_channels=opt.channels,
                base_channels=opt.base_channels,
                road_channels=opt.road_channels,
                img_width=opt.img_width,
                img_height=opt.img_height,
                ext_flag=opt.ext_flag)

model.apply(weights_init_normal)
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
print_model_parm_nums(model, 'Mixmap')

criterion = nn.MSELoss()
if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set----------------------
source_datapath = os.path.join('data', opt.dataset_name)
train_dataloader = get_dataloader_sr(
    source_datapath, opt.batch_size, 'train', opt.city, opt.channels)
valid_dataloader = get_dataloader_sr(
    source_datapath, opt.batch_size, 'valid', opt.city, opt.channels)

# Optimizers----------------------------------------------
# lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# training phase ========================================
iter = 0
rmses = [np.inf]
maes = [np.inf]
mapes = [np.inf]
mapes_in, mapes_out = [np.inf], [np.inf]

# # adjust lr
dic = {30:1e-4}
print('leaining rate changes when [epoch/lr]:',dic)

# start trainning =======================================
pbar = tqdm(range(opt.n_epochs))
# for epoch in tqdm(range(opt.n_epochs)):
for epoch in pbar:
    pbar.set_description(save_path[12:])

    for i, (real_coarse_A, ext, real_fine_A, road_A) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        #---forward------------
        gen_hr = model(real_coarse_A, ext, road_A)
        loss = criterion(gen_hr, real_fine_A)

        loss.backward()
        optimizer.step()

        iter += 1

        # validation----------------------------------------
        if iter % len(train_dataloader) == 0:
            model.eval()
            total_mape = 0
            total_mape_in, total_mape_out = 0.0, 0.0
            with torch.no_grad():
                for j, (flows_c, ext, flows_f, road) in enumerate(valid_dataloader): 
                    preds = model(flows_c, ext, road) 
                    preds_ = preds.cpu().detach().numpy()
                    flows_f_ = flows_f.cpu().detach().numpy()
                    total_mape += get_MAPE(preds_, flows_f_) * len(flows_c)
            mape = total_mape / len(valid_dataloader.dataset)
            # select best MAPES to seve model
            if mape < np.min(mapes):
                tqdm.write("epoch\t{}\titer\t{}\tMAPE\t{:.6f}".format(epoch, iter, mape))
                #--save model at each iter--
                # torch.save(model.state_dict(),'{}/model-{}.pt'.format(save_path, iter))
                torch.save(model.state_dict(),'{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tMAPE\t{:.6f}\n".format(epoch, iter, mape))
                f.close()
            mapes.append(mape)
            
    if epoch in dic:
        lr = dic[epoch]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        tqdm.write(f'epoch{epoch}- G_LR - now is {lr}')


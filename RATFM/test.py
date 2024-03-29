import os
import warnings
import argparse
import numpy as np

import torch
from .utils.metrics import *
from .utils.misc import *

from .utils.data_sr_road import get_dataloader_sr
from .models.RATFM import RATFM
from .modules.transformer import build_transformer
from .modules.position_encoding import build_position_encoding

warnings.filterwarnings("ignore")

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_residuals', type=int, default=16,
                    help='number of residual units')
parser.add_argument('--base_channels', type=int,
                    default=128, help='number of feature maps')
parser.add_argument('--road_channels', type=int,
                    default=1, help='number of feature maps')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--batch_size', type=int, default=16,
                    help='training batch size')
parser.add_argument('--map_width', type=int, default=64,
                    help='image width')
parser.add_argument('--map_height', type=int, default=64,
                    help='image height')
parser.add_argument('--channels', type=int, default=2,  
                    help='number of flow image channels')                                         
parser.add_argument('--dataset_name', type=str, default='XiAn',  #  XiAn | ChengDu | TaxiBJ-P1 
                    help='which dataset to use')
parser.add_argument('--city_road_map', type=str, default='xian',  # cdu | xian | bj | no  
                    help='which city_road_map to use')
parser.add_argument('--run_num', type=int, default=0,
                    help='save model folder')
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

model_path = 'model/{}-{}-{}'.format(opt.dataset_name, 
                                    opt.ext_flag,
                                    opt.run_num)

# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# # initial model
transformer = build_transformer(opt, opt.hidden_dim)
position_embedding = build_position_encoding(opt, opt.hidden_dim)
model = RATFM(position_embedding, transformer,
                in_channels=opt.channels,
                out_channels=opt.channels,
                base_channels=opt.base_channels,
                road_channels=opt.road_channels,
                map_width=opt.map_width,
                map_height=opt.map_height,
                n_residuals=opt.n_residuals,
                ext_flag=opt.ext_flag)

# load model
model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))

if cuda: model.cuda()
print_model_parm_nums(model, 'RATFM')

# load testset
datapath = os.path.join('data', opt.dataset_name)
dataloader = get_dataloader_sr(
    datapath, opt.batch_size, 'test', opt.city_road_map, opt.channels)

# testing phase--------------------------------------------------------------
total_mse, total_mae, total_mape, total_p2p_mape = 0, 0, 0, 0
total_mape_in, total_mape_out = 0.0, 0.0

with torch.no_grad():
    model.eval()
    for j, (test_data, ext, test_labels, road) in enumerate(dataloader):
        preds = model(test_data, ext, road)
        preds = preds.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()

        total_mse += get_MSE(preds, test_labels) * len(test_data)
        total_mae += get_MAE(preds, test_labels) * len(test_data)
        total_mape += get_MAPE(preds, test_labels) * len(test_data)
        total_p2p_mape += get_p2p_MAPE(preds, test_labels.copy()) * len(test_data)

rmse = np.sqrt(total_mse / len(dataloader.dataset))
mae = total_mae / len(dataloader.dataset)
mape = total_mape / len(dataloader.dataset)
p2p_mape = total_p2p_mape / len(dataloader.dataset)

print('Test RMSE = {:.6f}, MAE = {:.6f}, MAPE = {:.6f}, p2p_MAPE = {:.6f}'.format(rmse, mae, mape, p2p_mape))

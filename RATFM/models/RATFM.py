import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from ..utils.misc import get_nested_tensor, NestedTensor
from ..modules.road_1dconv import OneD_Block

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Residual_OneD_Block(nn.Module):
    def __init__(self, in_features):
        super(Residual_OneD_Block, self).__init__()

        conv_block = [OneD_Block(in_features, in_features),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      OneD_Block(in_features, in_features),
                      nn.BatchNorm2d(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Mixmap(nn.Module):
    def __init__(self, position_embedding, transformer, in_channels=2, out_channels=2, 
                n_residual_blocks_prefix=5,n_residuals=16,
                 base_channels=64, road_channels=1,ext_dim=7, map_width=64, map_height=64, ext_flag=False, hidden_dim=128):
        super(Mixmap, self).__init__()
        self.ext_flag = ext_flag
        self.map_width = map_width
        self.map_height = map_height
        self.in_channels = in_channels
        self.out_channels = out_channels

        if ext_flag and in_channels==2: #xian and chengdu
            self.embed_day = nn.Embedding(8, 2) # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3) # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(15, 3) #  14types: ignore 0, thus use 15
            self.ext2lr = nn.Sequential(
                nn.Linear(10, 128), 
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, map_width * map_height),
                nn.ReLU(inplace=True)
            )
        if ext_flag and in_channels==1:# beijing
            self.embed_day = nn.Embedding(8, 2) # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3) # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(18, 3) # ignore 0, thus use 18  
            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128), 
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, map_width * map_height),
                nn.ReLU(inplace=True)
            )

        if ext_flag:
            conv1_in = in_channels + 1
        else:
            conv1_in = in_channels
        conv3_in = base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks_prefix):
            res_blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Residual blocks after introducing road network map
        res_blocks_post = []
        for _ in range(n_residuals - n_residual_blocks_prefix):
            res_blocks_post.append(ResidualBlock(base_channels))
        self.res_blocks_post = nn.Sequential(*res_blocks_post)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.BatchNorm2d(base_channels))
        
        # output conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_in, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # final conv
        self.conv4 = nn.Conv2d(base_channels, out_channels, 1)

        self.conv_merge = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_trans = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), 
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True))

        if in_channels == 2:
            # ROAD NETWORK BRANCH
            self.res_road = nn.Sequential(
                nn.Conv2d(road_channels, base_channels, 3, 1, 1),
                nn.ReLU(inplace=True),         
                OneD_Block(base_channels, base_channels),
                Residual_OneD_Block(base_channels), 
                Residual_OneD_Block(base_channels),
                nn.MaxPool2d(2)
                )
            self.down_pool = nn.AvgPool2d(2)
            self.road_down_pool = nn.MaxPool2d(2)

        elif in_channels == 1:
            # ROAD NETWORK BRANCH
            self.res_road = nn.Sequential(
                nn.Conv2d(road_channels, base_channels, 3, 1, 1),
                nn.ReLU(inplace=True),         
                OneD_Block(base_channels, base_channels),
                Residual_OneD_Block(base_channels), 
                Residual_OneD_Block(base_channels)
                )
            self.down_pool = nn.AvgPool2d(4)
            self.road_down_pool = nn.MaxPool2d(4)

        else:
            print('input channle error!')

        # tranformer architecture 
        self.transformer = transformer
        self.position_embedding = position_embedding


    def forward(self, cmap, ext, roadmap):
        inp = cmap
        # external factor modeling 
        if self.ext_flag and self.in_channels==2: # XiAn and ChengDu
            ext_out1 = self.embed_day(ext[:, 0].long().view(-1, 1)).view(-1, 2) 
            ext_out2 = self.embed_hour(ext[:, 1].long().view(-1, 1)).view(-1, 3) 
            ext_out3 = self.embed_weather(ext[:, 4].long().view(-1, 1)).view(-1, 3) 
            ext_out4 = ext[:, 2:4] 
            ext_out = self.ext2lr(torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.map_width, self.map_height) 
            inp = torch.cat([cmap, ext_out], dim=1)
        if self.ext_flag and self.in_channels==1: # TaxiBJ-P1
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.map_width, self.map_height)
            inp = torch.cat([cmap, ext_out], dim=1)

        # short-range inference
        out1 = self.conv1(inp)
        out1 = self.res_blocks(out1)

        rout = self.res_road(roadmap)
        out_mix = torch.cat((out1, rout), dim=1)
        out_mix = self.conv_merge(out_mix)

        out_mix = self.res_blocks_post(out_mix)
        out2 = self.conv2(out_mix)
        out3 = self.conv3(out2)

        # pooling 
        out_pool = self.down_pool(out3)   # short-range feature - avgpooling
        rout_pool = self.road_down_pool(rout)  # road feature   - maxpooling
        
        # long-range inference
        if isinstance(out_pool, (list, torch.Tensor)):
            out_nested = get_nested_tensor(out_pool)
        src, mask = out_nested.decompose()
        pos = self.position_embedding(out_nested).to(out_nested.tensors.dtype)
        assert mask is not None
        long_out = self.transformer(src, mask, rout_pool, pos)
        long_out = F.interpolate(long_out,size=list(out3.shape[-2:]))

        # merge short-range and long-range
        out = torch.add(long_out, out3)
        out = self.conv_trans(out)
        out = self.conv4(out)
        
        return out





        

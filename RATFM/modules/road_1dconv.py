import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)


# class OneD_Block_hv(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(OneD_Block_hv, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity
#         self.deconv1 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
#         )
#         self.deconv2 = nn.Conv2d(
#             in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
#         )
#         self.conv3 = nn.Conv2d(
#             in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x1 = self.deconv1(x)
#         x2 = self.deconv2(x)
#         x = torch.cat((x1, x2), 1)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x


class OneD_Block_hv(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(OneD_Block_hv, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 2)
        # self.relu1 = nn.ReLU(inplace=True)
        self.deconv1 = nn.Conv2d(
            in_channels, in_channels // 2, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels, in_channels // 2, (9, 1), padding=(4, 0)
        )
        # self.conv3 = nn.Conv2d(
        #     in_channels, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x = torch.cat((x1, x2), 1)
        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)
        return x


class OneD_Block(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(OneD_Block, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 4)
        # self.relu1 = nonlinearity
        # self.deconv1 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        # )
        # self.deconv2 = nn.Conv2d(
        #     in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        # )
        # self.deconv3 = nn.Conv2d(
        #     in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0)
        # )
        # self.deconv4 = nn.Conv2d(
        #     in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4)
        # )
        # # self.norm2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 8)
        # # self.relu2 = nonlinearity
        # self.conv3 = nn.Conv2d(
        #     in_channels // 4 + in_channels // 8, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nonlinearity

        # #----------new----------
        # self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 4)
        # self.relu1 = nonlinearity
        self.deconv1 = nn.Conv2d(
            in_channels, in_channels // 2, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels, in_channels // 2, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels, in_channels // 4, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels, in_channels // 4, (1, 9), padding=(0, 4)
        )
        # self.norm2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 8)
        # self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(
            in_channels + in_channels // 2, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nonlinearity

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        # x1 = self.deconv1(x)
        # x2 = self.deconv2(x)
        # x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        # x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        # x = torch.cat((x1, x2, x3, x4), 1)
        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)
        # return x

        # # # --------new--------------
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        # x = F.interpolate(x, scale_factor=2)
        # x = self.norm2(x)
        # x = self.relu2(x)
        x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride=1, activation='ReLu'):
        super(Conv_layer, self).__init__()

        self.conv_layer = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channel)

        self.activation_layer = None
        if activation == 'ReLu':
           self.activation_layer = nn.ReLU(True)
        elif activation == 'LeakyReLU':
           self.activation_layer = nn.LeakyReLU(0.2, True)
        elif activation == 'Sigmoid':
           self.activation_layer = nn.Sigmoid()
        elif activation == 'Tanh':
           self.activation_layer = nn.Tanh()
        # else:
        #    assert (1 == 0)

    def forward(self, x):
        conv_fea = self.conv_layer(x)
        out   = self.bn(conv_fea)

        if self.activation_layer is not None:
           out = self.activation_layer(out)

        return out
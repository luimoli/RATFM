import torch
import torch.nn as nn
import torch.nn.functional as F

class OneD_Block(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(OneD_Block, self).__init__()
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
        self.conv = nn.Conv2d(
            in_channels + in_channels // 2, n_filters, 1)

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv(x)
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


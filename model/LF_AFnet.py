import torch
import torch.nn as nn


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = args.channels
        self.factor = args.scale_factor # S
        n_groups, n_blocks = 4, 6

        # Initial Feature Extraction
        self.spaFE_init = SpaFE(1, channels, kernel_size=3)
        # Spatial-Angular Decouple and Fusion
        self.decouple = Cascaded_DF_Groups(n_groups, n_blocks, channels)
        # Reconstruction
        self.reconstruction = ReconBlock(channels, self.factor)

    def forward(self, x, angRes):
        angRes = angRes[0].item() # A

        # Bilinear
        [B, _, H, W] = x.size()
        h = H // angRes
        w = W // angRes
        x_upscale = x.view(B, 1, angRes, h, angRes, w)
        x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B*angRes**2, 1, h, w)
        x_upscale = torch.nn.functional.interpolate(x_upscale, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x_upscale = x_upscale.view(B, angRes, angRes, 1, h*self.factor, w*self.factor)
        x_upscale = x_upscale.permute(0,3,1,4,2,5).contiguous().view(B, 1, H*self.factor, W*self.factor) # [B, 1, A*h*S, A*w*S]

        # Initial Feature Extraction
        x = SaiAddPadding(x, angRes)                            # [B, 1, A*(h+1), A*(w+1)]
        buffer_x = self.spaFE_init(x, angRes)                   # [B, C, A*(h+1), A*(w+1)]

        # Spatial-Angular Interaction
        buffer_x = self.decouple(buffer_x, angRes)
        buffer_x = SaiDelPadding(buffer_x, angRes)              # [B, C, A*h, A*w]

        # Reconstruction
        out = self.reconstruction(buffer_x, angRes) + x_upscale # [B, 1, A*h*S, A*w*S]
        return out


class Flexible_Conv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernel_size):
        super(Flexible_Conv2d, self).__init__()
        self.conv2d_kernel = nn.Conv2d(inChannels, outChannels, kernel_size=kernel_size, bias=False)
        self.kernel_size = kernel_size

    def arbitrary_func(self, x, stride, padding, dilation):
        if self.kernel_size == 1:
            padding = 0
        out = torch.nn.functional.conv2d(x, weight=self.conv2d_kernel.weight, stride=stride, padding=padding,
                                         dilation=dilation, bias=None)
        return out


class SpaFE(Flexible_Conv2d):
    def __init__(self, inChannels, outChannels, kernel_size):
        super(SpaFE, self).__init__(inChannels, outChannels, kernel_size)

    def forward(self, x, angRes):
        buffer_xs = self.arbitrary_func(x, stride=1, padding=1, dilation=1)
        out = SaiClearPadding(buffer_xs, angRes)
        return out


class AngFE(Flexible_Conv2d):
    def __init__(self, inChannels, outChannels, kernel_size):
        super(AngFE, self).__init__(inChannels, outChannels, kernel_size)
        self.num_pad = kernel_size // 2

    def forward(self, x, angRes):
        height = x.size(2)//angRes
        width = x.size(3)//angRes
        buffer_xa = self.arbitrary_func(x, stride=1, padding=[height*self.num_pad, width*self.num_pad], dilation=[height, width])
        out = SaiClearPadding(buffer_xa, angRes)
        return out


class DF_Block(nn.Module):
    def __init__(self, channels):
        super(DF_Block, self).__init__()
        self.AngFE = AngFE(channels, channels, kernel_size=3)
        self.SpaFE = SpaFE(channels, channels, kernel_size=3)
        self.Fuse = nn.Conv2d(3*channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x, angRes):
        buffer_xa_1 = self.ReLU(self.AngFE(x, angRes))
        buffer_xa_2 = self.ReLU(self.AngFE(buffer_xa_1, angRes))
        buffer_xs = self.ReLU(self.SpaFE(x, angRes))

        buffer = torch.cat((buffer_xs, buffer_xa_1, buffer_xa_2), 1)
        out = self.ReLU(self.Fuse(buffer)) + x

        return out


class DF_Group(nn.Module):
    def __init__(self, n_block, channels):
        super(DF_Group, self).__init__()
        modules = []
        self.n_block = n_block
        for i in range(n_block):
            modules.append(DF_Block(channels))
        self.chained_blocks = nn.Sequential(*modules)
        self.SpaFE = SpaFE(channels, channels, kernel_size=3)

    def forward(self, x, angRes):
        buffer_x = x
        for i in range(self.n_block):
            buffer_x = self.chained_blocks[i](buffer_x, angRes)
        out = self.SpaFE(buffer_x, angRes) + x
        return out


class Cascaded_DF_Groups(nn.Module):
    def __init__(self, n_group, n_block, channels):
        super(Cascaded_DF_Groups, self).__init__()
        self.n_group = n_group
        body = []
        for i in range(n_group):
            body.append(DF_Group(n_block, channels))
        self.body = nn.Sequential(*body)
        self.SpaFE = SpaFE(channels, channels, kernel_size=3)

    def forward(self, x, angRes):
        buffer_x = x
        for i in range(self.n_group):
            buffer_x = self.body[i](buffer_x, angRes)
        out = self.SpaFE(buffer_x, angRes) + x
        return out


class ReconBlock(nn.Module):
    def __init__(self, channels, upscale_factor):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=1, stride=1, bias=False)
        self.PixelShuffle = nn.PixelShuffle(upscale_factor)
        self.FinalConv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, angRes):
        buffer_x = self.PreConv(x)
        buffer_x = self.PixelShuffle(buffer_x)
        out = self.FinalConv(buffer_x)

        return out

def SaiAddPadding(x, angRes):
    B, C, H, W = x.shape
    h, w = H // angRes, W // angRes
    x = x.view(B, C, angRes, h, angRes, w).permute(0,1,2,4,3,5).contiguous().view(B, -1, h, w)
    x = torch.nn.functional.pad(x, pad=[0, 1, 0, 1], mode='constant', value=0)
    out = x.view(B, C, angRes, angRes, h+1, w+1).permute(0,1,2,4,3,5).contiguous().view(B, C, angRes*(h+1), angRes*(w+1))
    return out

def SaiDelPadding(x, angRes):
    B, C, H, W = x.shape
    h, w = H // angRes, W // angRes
    x = x.view(B, C, angRes, h, angRes, w).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, h, w)
    x = x[:, :, 0:h-1, 0:w-1]
    out = x.view(B, C, angRes, angRes, h - 1, w - 1).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, angRes * (h - 1),
                                                                                                 angRes * (w - 1))
    return out

def SaiClearPadding(x, angRes):
    B, C, H, W = x.shape
    h, w = H // angRes, W // angRes
    x[:,:, h:angRes:, w:angRes:] = 0

    return x

def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=None):
        loss = self.criterion_Loss(SR, HR)

        return loss



if __name__ == "__main__":
    import time
    from option import args
    net = get_model(args).cuda()
    from thop import profile
    angRes = 5
    total = sum([param.nelement() for param in net.parameters()])
    angRes = torch.tensor([angRes]).int()
    start = time.clock()
    input = torch.randn(1, 1, 32 * angRes, 32 * angRes).cuda()
    flops, params = profile(net, inputs=(input, angRes))
    elapsed = (time.clock() - start)
    print("   Time used:", elapsed)
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of parameters: %.2fM' % (params/1e6))
    print('   Number of FLOPs: %.2fG' % (flops*2/1e9))

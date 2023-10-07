import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
 
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        #map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
 
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        # out = self.spatial_attention(out) * out
        return out

class Conv2DBlock(nn.Module):
    """ Conv + ReLU + BN"""
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Double2DConv2(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv2, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (1, 1))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_3 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_4 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_5 = Conv2DBlock(in_dim, out_dim, (5, 5))
        self.conv_6 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_7 = Conv2DBlock(out_dim*3, out_dim, (3, 3))

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)

        x2 = self.conv_3(x)
        x2 = self.conv_4(x2)

        x3 = self.conv_5(x)
        x3 = self.conv_6(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_7(x)
        x = x + x2

        return x
    
class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNetV2(nn.Module):
    """ Original structure but less two layers 
        Total params: 10,161,411
        Trainable params: 10,153,859
        Non-trainable params: 7,552
    """
    def __init__(self, in_dim=9, out_dim=3):
        super(TrackNetV2, self).__init__()
        self.down_block_1 = Double2DConv2(in_dim=in_dim, out_dim=64)
        self.down_block_2 = Double2DConv2(in_dim=64, out_dim=128)
        self.down_block_3 = Double2DConv2(in_dim=128, out_dim=256)
        self.bottleneck = Triple2DConv(in_dim=256, out_dim=512)
        self.up_block_1 = Double2DConv(in_dim=768, out_dim=256)
        self.up_block_2 = Double2DConv(in_dim=384, out_dim=128)
        self.up_block_3 = Double2DConv(in_dim=192, out_dim=64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.cbam1 = CBAM(channel=256) #only channel attention
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=64)

        self.cbam0_2 = CBAM(channel=256)
        self.cbam1_2 = CBAM(channel=128)
        self.cbam2_2 = CBAM(channel=64)

    def forward(self, x):
        """ model input shape: (F*3, 288, 512), output shape: (F, 288, 512) """
        x1 = self.down_block_1(x)                                   # (64, 288, 512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                 # (64, 144, 256)
        x2 = self.down_block_2(x)                                   # (128, 144, 256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                 # (128, 72, 128)
        x3 = self.down_block_3(x)                                   # (256, 72, 128), one less conv layer
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                 # (256, 36, 64)
        x = self.bottleneck(x)                                      # (512, 36, 64)
        x3 = self.cbam0_2(x3)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)  # (768, 72, 128) 256+512
        
        x = self.up_block_1(x)                                      # (256, 72, 128), one less conv layer
        x = self.cbam1(x)
        x2 = self.cbam1_2(x2)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)  # (384, 144, 256) 256+128
        
        x = self.up_block_2(x)                                      # (128, 144, 256)
        x = self.cbam2(x)
        x1 = self.cbam2_2(x1)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)  # (192, 288, 512) 128+64
        
        x = self.up_block_3(x)                                      # (64, 288, 512)
        x = self.cbam3(x)
        x = self.predictor(x)                                       # (3, 288, 512)
        x = self.sigmoid(x)
        return  x


# from torchsummary import summary
# Tr = TrackNetV2().cuda()
# summary(Tr, (9, 288, 512))
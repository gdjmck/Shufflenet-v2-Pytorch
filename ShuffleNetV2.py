import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            #print('\tx1:', x1.shape)
            x2 = x[:, (x.shape[1]//2):, :, :]
            #print('\tx2:', x2.shape)
            out = self._concat(x1, self.banch2(x2))
            #print('\tout:', out.shape)
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class Decoder(nn.Module):
    def __init__(self, block, inp, oup, scale=8):
        super(Decoder, self).__init__()
        self.scale = scale
        self.block = block
        assert (math.log2(scale)).is_integer()
        times = int(math.log2(scale))
        assert inp % scale == 0
        '''
        self.net_1 = nn.Sequential(block(int(inp), int(inp/2), 1, 2), 
                                    nn.ConvTranspose2d(int(inp/2), int(inp/2), kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(int(inp/2)))
        self.net_2 = nn.Sequential(block(int(inp/2), int(inp/4), 1, 2), 
                                    nn.ConvTranspose2d(int(inp/4), int(inp/4), kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(int(inp/4)))
        self.net_3 = nn.Sequential(block(int(inp/4), int(inp/8), 1, 2), 
                                    nn.ConvTranspose2d(int(inp/8), int(inp/8), kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(int(inp/8)))
        '''
        self.net = nn.Sequential(*[nn.Sequential(block(int(inp/2**i), int(inp/2**(i+1)), 1, 2), 
                                    nn.ConvTranspose2d(int(inp/2**(i+1)), int(inp/2**(i+1)), kernel_size=2, stride=2), 
                                    nn.BatchNorm2d(int(inp/2**(i+1)))) for i in range(times)])
        self.squeeze = nn.Conv2d(int(inp/2**times), oup, kernel_size=1)


    def forward(self, x):
        x_rec = self.net(x)
        x_rec = self.squeeze(x_rec)
        return x_rec


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        
        assert input_size % 32 == 0
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(5, input_channel, 2)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
	            #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
        
        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

        # building generator of reconstructing face region
        self.generator = Decoder(InvertedResidual, self.stage_out_channels[-1], 3, scale=int(input_size/4))

    def forward(self, x):
        #print('\tmodel input:', x.shape)
        x = self.conv1(x)
        #print('\tconv1:', x.shape)
        x = self.maxpool(x)
        #print('\tmaxpool:', x.shape)
        x = self.features(x)
        #print('\tfeatures:', x.shape)
        x = self.conv_last(x)
        encode = x
        x_recon = self.generator(encode)
        #print('reconstruct x:', x_recon.shape)

        #print('\tconv last:', x.shape)
        x = self.globalpool(x)
        #print('\tglobal pool:', x.shape)
        x = x.view(-1, self.stage_out_channels[-1])
        #print('\tflatten:', x.shape)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        #print('\tmodel output:', x.shape)
        return x, x_recon

def shufflenetv2(width_mult=1.):
    model = ShuffleNetV2(width_mult=width_mult)
    return model
    
if __name__ == "__main__":
    """Testing
    """
    import numpy as np
    model = ShuffleNetV2(n_class=9, input_size=128)
    #print(model)
    x = torch.Tensor(np.zeros((1, 3, 128, 128)))
    y = model(x)
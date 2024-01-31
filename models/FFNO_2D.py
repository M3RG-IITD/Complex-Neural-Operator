import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math

# Complex Pytorch
import torchcomplex.nn as cn

# Fractional Fourier Transform 
from torch_frft.layer import DFrFTLayer, FrFTLayer
from torch_frft.frft_module import frft
from torch_frft.dfrft_module import dfrft, dfrftmtx

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
################################################################
# fractional fourier layer
################################################################

class SpectralFractionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, alpha: float = 0.5):
        super(SpectralFractionalConv2d, self).__init__()

        """
        2D Fractional Fourier layer. It does FFFT, linear transform, and Inverse FFFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        self.alpha1 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
    
        self.layer = cn.Conv2d(self.in_channels, self.out_channels, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fractional Fourier coeffcients up to factor of e^(- something constant)
        x_ft = dfrft(dfrft(x, self.alpha1, dim=-1), self.alpha2, dim=-2)
        
        # 1x1 convolutionl in complex domain
        out_ft = self.layer(x_ft)

        # Return to physical space
        x = dfrft(dfrft(out_ft, -self.alpha2, dim=-2), -self.alpha1, dim=-1).real
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        self.modes1 = args.num_basis
        self.modes2 = args.num_basis
        self.width = args.d_model
        self.padding = [int(x) for x in args.padding.split(',')]

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.fconv0 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fconv1 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fconv2 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fconv3 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc0 = nn.Linear(in_channels + 2, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.fconv0(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.fconv1(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.fconv2(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.fconv3(x)
        x = x1 + x2 + x3

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
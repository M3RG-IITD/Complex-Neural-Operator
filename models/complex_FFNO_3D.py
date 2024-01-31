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
# complex GeLU
################################################################
class ComplexGELU(nn.Module):
    def __init__(self):
        super(ComplexGELU, self).__init__()

    def forward(self, input):
        real_gelu = F.gelu(input.real)
        imag_gelu = F.gelu(input.imag)

        return torch.complex(real_gelu, imag_gelu)
    
################################################################
# 3d complex fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.ifftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

################################################################
# complex fractional fourier layer
################################################################

class SpectralFractionalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, alpha: float = 0.5):
        super(SpectralFractionalConv3d, self).__init__()

        """
        2D Complex Fractional Fourier layer. It does FFFT, linear transform, and Inverse FFFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.alpha1 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
    
        self.layer = cn.Conv3d(self.in_channels, self.out_channels, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fractional Fourier coeffcients up to factor of e^(- something constant)
        x_ft = dfrft(dfrft(dfrft(x, self.alpha1, dim=-1), self.alpha2, dim=-2), self.alpha3, dim=-3)
        
        # 1x1 convolutionl in complex domain
        out_ft = self.layer(x_ft)

        # Return to physical space
        x = dfrft(dfrft(dfrft(out_ft, -self.alpha3, dim=-3), -self.alpha2, dim=-2), -self.alpha1, dim=-1)
        return x
    
################################################################
# Learning Imaginary Part
################################################################
class ComplexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        return self.block(x)
    
################################################################
# Complex Channel Mixing
################################################################
class ComplexChannelMixing(cn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexChannelMixing, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = ComplexGELU()
        self.linear = cn.Linear(self.in_channels, self.out_channels)
        
    def forward(self, x):
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x
    
################################################################
# Complex block
################################################################
class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride=1, padding=0, activation=ComplexGELU()):
        super(ConvBNActivation, self).__init__()
        
        self.conv = cn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = cn.BatchNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        self.modes1 = args.num_basis
        self.modes2 = args.num_basis
        self.modes3 = args.num_basis // 2
        self.width = args.d_model
        self.act = ComplexGELU()
        self.padding = [int(x) for x in args.padding.split(',')]

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.fconv0 = SpectralFractionalConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.fconv1 = SpectralFractionalConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.fconv2 = SpectralFractionalConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.fconv3 = SpectralFractionalConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = cn.Conv3d(self.width, self.width, 1)
        self.w1 = cn.Conv3d(self.width, self.width, 1)
        self.w2 = cn.Conv3d(self.width, self.width, 1)
        self.w3 = cn.Conv3d(self.width, self.width, 1)
        
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc0 = nn.Linear(in_channels + 3, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self.residual = ConvBNActivation(in_channels=self.width, out_channels=self.width)
        self.complexblock = ComplexBlock(in_channels=self.width, out_channels=self.width)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1], 0, self.padding[2]])

        # Real to complex conversion
        x_r = x
        x_i = self.complexblock(x)
        x = torch.complex(x_r, x_i)
        x = self.residual(x)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.fconv0(x)
        x = x1 + x2 + x3
        x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.fconv1(x)
        x = x1 + x2 + x3
        x = self.act(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.fconv2(x)
        x = x1 + x2 + x3
        x = self.act(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.fconv3(x)
        x = x1 + x2 + x3
        
        # Complex to Real Conversion
        x = x.real

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[2], :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
from math import pi

import torch

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
# complex SELU
################################################################
class ComplexSELU(nn.Module):
    def __init__(self):
        super(ComplexSELU, self).__init__()

    def forward(self, input):
        real_selu = F.selu(input.real)
        imag_selu = F.selu(input.imag)

        return torch.complex(real_selu, imag_selu)
    
################################################################
# complex fourier layer
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
        x_ft = torch.fft.fft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
################################################################
# complex fractional fourier layer
################################################################

class SpectralFractionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, alpha: float = 0.5):
        super(SpectralFractionalConv2d, self).__init__()

        """
        2D Complex Fractional Fourier layer. It does FFFT, linear transform, and Inverse FFFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        
        # self.alpha1 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=True)
        self.alpha1 = alpha
        self.alpha2 = alpha
    
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
        x = dfrft(dfrft(out_ft, -self.alpha2, dim=-2), -self.alpha1, dim=-1)
        return x
    
################################################################
# Real to Complex Conversion
################################################################
def real_to_complex(x, width):
    x_r = x[:, :width, :, :]
    x_i = x[:, width:, :, :]
    x_complex = torch.complex(x_r, x_i)

    return x_complex

################################################################
# Complex to Real Conversion
################################################################
def complex_to_real(x):
    # Ensure that the input tensor is complex
    if not torch.is_complex(x):
        raise ValueError("Input tensor must be complex.")

    # Separate real and imaginary parts
    x_r = x.real
    x_i = x.imag

    # Concatenate real and imaginary parts along the specified dimension
    result = torch.cat((x_r, x_i), dim=-3)

    return result

################################################################
# Learning Imaginary Part
################################################################
class ComplexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ComplexBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
        # self.norm = cn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

################################################################
# Complex block
################################################################
class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride=1, padding=0, activation=ComplexGELU()):
        super(ConvBNActivation, self).__init__()
        
        self.conv = cn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = cn.BatchNorm2d(out_channels)
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
        self.width = args.d_model
        self.act = ComplexGELU()
        
        self.padding = [int(x) for x in args.padding.split(',')]

        self.conv0 = SpectralConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        
        self.fconv0 = SpectralFractionalConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.fconv1 = SpectralFractionalConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.fconv2 = SpectralFractionalConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        self.fconv3 = SpectralFractionalConv2d(self.width//3, self.width//3, self.modes1, self.modes2)
        
        self.w0 = cn.Conv2d(self.width//3, self.width//3, 3, 1, 1)
        self.w1 = cn.Conv2d(self.width//3, self.width//3, 3, 1, 1)
        self.w2 = cn.Conv2d(self.width//3, self.width//3, 3, 1, 1)
        self.w3 = cn.Conv2d(self.width//3, self.width//3, 3, 1, 1)

        self.fc0 = nn.Linear(in_channels + 2, self.width)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self.residual = ConvBNActivation(in_channels=self.width, out_channels=self.width)
        self.complexblock = ComplexBlock(in_channels=self.width, out_channels=self.width)
        
        self.channelmixing0 = cn.Conv2d(self.width, self.width, 1)
        self.channelmixing1 = cn.Conv2d(self.width, self.width, 1)
        self.channelmixing2 = cn.Conv2d(self.width, self.width, 1)
        self.channelmixing3 = cn.Conv2d(self.width, self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        # Real to complex conversion
        x_r = x
        x_i = self.complexblock(x)
        x = torch.complex(x_r, x_i)
        x = self.residual(x)
        
        dim = self.width//3
        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:2*dim, :, :]
        x3 = x[:, 2*dim:, :, :]
        
        x1 = self.conv0(x1)
        x2 = self.w0(x2)
        x3 = self.fconv0(x3)
        x = torch.cat([x1, x2, x3], axis = 1)
        x = self.channelmixing0(x)
        x = self.act(x)

        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:2*dim, :, :]
        x3 = x[:, 2*dim:, :, :]
        
        x1 = self.conv1(x1)
        x2 = self.w1(x2)
        x3 = self.fconv1(x3)
        x = torch.cat([x1, x2, x3], axis = 1)
        x = self.channelmixing0(x)
        x = self.act(x)

        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:2*dim, :, :]
        x3 = x[:, 2*dim:, :, :]
        
        x1 = self.conv2(x1)
        x2 = self.w2(x2)
        x3 = self.fconv2(x3)
        x = torch.cat([x1, x2, x3], axis = 1)
        x = self.channelmixing0(x)
        x = self.act(x)

        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:2*dim, :, :]
        x3 = x[:, 2*dim:, :, :]
        
        x1 = self.conv3(x1)
        x2 = self.w3(x2)
        x3 = self.fconv3(x3)
        x = torch.cat([x1, x2, x3], axis = 1)
        x = self.channelmixing0(x)
        
        # Complex to Real Conversion
        x = x.real
        # x = complex_to_real(x)

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
    
# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         in_channels = args.in_dim
#         out_channels = args.out_dim
#         self.modes1 = args.num_basis
#         self.modes2 = args.num_basis
#         # width//2 as it is converted into complex
#         self.width = args.d_model//2
#         self.w = args.d_model
#         self.act = ComplexGELU()
        
#         self.padding = [int(x) for x in args.padding.split(',')]

#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
#         self.fconv0 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.fconv1 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.fconv2 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.fconv3 = SpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        
#         self.w0 = cn.Conv2d(self.width, self.width, 1)
#         self.w1 = cn.Conv2d(self.width, self.width, 1)
#         self.w2 = cn.Conv2d(self.width, self.width, 1)
#         self.w3 = cn.Conv2d(self.width, self.width, 1)

#         self.fc0 = nn.Linear(in_channels + 2, self.w)  # input channel is 3: (a(x, y), x, y)
#         self.fc1 = nn.Linear(self.w, 128)
#         self.fc2 = nn.Linear(128, out_channels)

#     def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)
        
#         if not all(item == 0 for item in self.padding):
#             x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

#         # Real to complex conversion
#         x = real_to_complex(x, self.width)
        
#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x3 = self.fconv0(x)
#         x = x1 + x2 + x3
#         x = self.act(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x3 = self.fconv1(x)
#         x = x1 + x2 + x3
#         x = self.act(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x3 = self.fconv2(x)
#         x = x1 + x2 + x3
#         x = self.act(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x3 = self.fconv3(x)
#         x = x1 + x2 + x3
        
#         # Complex to Real Conversion
#         x = complex_to_real(x)

#         if not all(item == 0 for item in self.padding):
#             x = x[..., :-self.padding[1], :-self.padding[0]]
        
#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         return x

#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)
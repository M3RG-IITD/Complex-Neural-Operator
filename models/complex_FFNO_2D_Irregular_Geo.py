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
# complex fourier layer
################################################################
class ComplexSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(ComplexSpectralConv2d, self).__init__()

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

class ComplexSpectralFractionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, alpha: float = 0.5):
        super(ComplexSpectralFractionalConv2d, self).__init__()

        """
        2D Complex Fractional Fourier layer. It does FFFT, linear transform, and Inverse FFFT.    
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
        x = dfrft(dfrft(out_ft, -self.alpha2, dim=-2), -self.alpha1, dim=-1)
        return x
    
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                          torch.arange(start=-(self.modes2 - 1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(
            device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                          torch.arange(start=-(self.modes2 - 1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(
            device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y
    
class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3 * self.width, 4 * self.width)
        self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc4 = nn.Linear(4 * self.width, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.0001, 0.0001], device="cuda").reshape(1, 1, 2)

        self.B = np.pi * torch.pow(2, torch.arange(0, self.width // 4, dtype=torch.float, device="cuda")).reshape(1, 1,
                                                                                                                  1,
                                                                                                                  self.width // 4)

    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:, :, 1] - self.center[:, :, 1], x[:, :, 0] - self.center[:, :, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:, :, 0], x[:, :, 1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        x_cos = torch.cos(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b, n, 3 * self.width)

        if code != None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1, xd.shape[1], 1)
            xd = torch.cat([cd, xd], dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd
    
################################################################
# Real to Complex Conversion
################################################################
def real_to_complex(x, width):
    """
    Convert a 2D tensor with real values to a complex tensor.

    Args:
        x (torch.Tensor): Input tensor with real values.
        width (int): Width of the complex tensor (half of the original width).

    Returns:
        torch.Tensor: Complex tensor.
    """
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
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        return self.block(x)
    
################################################################
# complex grid
################################################################
def generate_complex_grid(real_grid, bias_function, complex_block):
    # Apply bias to the real grid
    biased_real_grid = bias_function(real_grid)
    # Compute the imaginary part using the complex block
    imaginary_part = complex_block(biased_real_grid)
    # Combine the real and imaginary parts to form a complex-valued grid
    complex_grid = torch.complex(biased_real_grid, imaginary_part)
    
    return complex_grid

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
    def __init__(self, args, is_mesh=True, modes1=12, modes2=12, s1=96, s2=96):
        super(Model, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        width = args.d_model

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.act = ComplexGELU()
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.b0 = nn.Conv2d(2, self.width, 1)
        
        self.conv1 = ComplexSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = ComplexSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = ComplexSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.fconv1 = ComplexSpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fconv2 = ComplexSpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        self.fconv3 = ComplexSpectralFractionalConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w1 = cn.Conv2d(self.width, self.width, 1)
        self.w2 = cn.Conv2d(self.width, self.width, 1)
        self.w3 = cn.Conv2d(self.width, self.width, 1)
        
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self.residual = ConvBNActivation(in_channels=self.width, out_channels=self.width)
        self.complexblock = ComplexBlock(in_channels=self.width, out_channels=self.width)
        self.complexb1 = ComplexBlock(self.width, self.width, kernel_size=1, stride=1, padding=0)
        self.complexb2 = ComplexBlock(self.width, self.width, kernel_size=1, stride=1, padding=0)
        self.complexb3 = ComplexBlock(self.width, self.width, kernel_size=1, stride=1, padding=0)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)
        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        # Real to complex conversion
        uc_r = uc
        uc_i = self.complexblock(uc)
        uc = torch.complex(uc_r, uc_i)
        uc = self.residual(uc)
        
        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.fconv1(uc)
        uc4 = generate_complex_grid(grid, self.b1, self.complexb1)
        uc = uc1 + uc2 + uc3 + uc4
        uc = self.act(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.fconv2(uc)
        uc4 = generate_complex_grid(grid, self.b2, self.complexb2)
        uc = uc1 + uc2 + uc3 + uc4
        uc = self.act(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.fconv3(uc)
        uc4 = generate_complex_grid(grid, self.b3, self.complexb3)
        uc = uc1 + uc2 + uc3 + uc4
        uc = self.act(uc)
        
        # Complex to Real Conversion
        uc = uc.real

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
# class Model(nn.Module):
#     def __init__(self, args, is_mesh=True, modes1=12, modes2=12, s1=96, s2=96):
#         super(Model, self).__init__()
#         in_channels = args.in_dim
#         out_channels = args.out_dim
#         width = args.d_model

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
#         self.act = ComplexGELU()
#         self.is_mesh = is_mesh
#         self.s1 = s1
#         self.s2 = s2

#         self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
#         self.b0 = nn.Conv2d(2, self.width, 1)
        
#         self.conv1 = ComplexSpectralConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
#         self.conv2 = ComplexSpectralConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
#         self.conv3 = ComplexSpectralConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
        
#         self.fconv1 = ComplexSpectralFractionalConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
#         self.fconv2 = ComplexSpectralFractionalConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
#         self.fconv3 = ComplexSpectralFractionalConv2d(self.width//2, self.width//2, self.modes1, self.modes2)
        
#         self.w1 = cn.Conv2d(self.width//2, self.width//2, 1)
#         self.w2 = cn.Conv2d(self.width//2, self.width//2, 1)
#         self.w3 = cn.Conv2d(self.width//2, self.width//2, 1)
        
#         self.b1 = nn.Conv2d(2, self.width, 1)
#         self.b2 = nn.Conv2d(2, self.width, 1)
#         self.b3 = nn.Conv2d(2, self.width, 1)
        
#         self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
#         self.b4 = nn.Conv1d(2, self.width, 1)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, out_channels)

#     def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
#         # u (batch, Nx, d) the input value
#         # code (batch, Nx, d) the input features
#         # x_in (batch, Nx, 2) the input mesh (sampling mesh)
#         # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
#         # x_in (batch, Nx, 2) the input mesh (query mesh)
#         if self.is_mesh and x_in == None:
#             x_in = u
#         if self.is_mesh and x_out == None:
#             x_out = u
#         grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

#         u = self.fc0(u)
#         u = u.permute(0, 2, 1)

#         uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
#         uc3 = self.b0(grid)
#         uc = uc1 + uc3
#         uc = F.gelu(uc)

#         # Real to complex conversion
#         uc = real_to_complex(uc, self.width//2)
        
#         uc1 = self.conv1(uc)
#         uc2 = self.w1(uc)
#         uc3 = self.fconv1(uc)
#         uc4 = real_to_complex(self.b1(grid), self.width//2)
#         uc = uc1 + uc2 + uc3 + uc4
#         uc = self.act(uc)

#         uc1 = self.conv2(uc)
#         uc2 = self.w2(uc)
#         uc3 = self.fconv2(uc)
#         uc4 = real_to_complex(self.b2(grid), self.width//2)
#         uc = uc1 + uc2 + uc3 + uc4
#         uc = self.act(uc)

#         uc1 = self.conv3(uc)
#         uc2 = self.w3(uc)
#         uc3 = self.fconv3(uc)
#         uc4 = real_to_complex(self.b3(grid), self.width//2)
#         uc = uc1 + uc2 + uc3 + uc4
#         uc = self.act(uc)
        
#         # Complex to Real Conversion
#         uc = complex_to_real(uc)

#         u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
#         u3 = self.b4(x_out.permute(0, 2, 1))
#         u = u + u3

#         u = u.permute(0, 2, 1)
#         u = self.fc1(u)
#         u = F.gelu(u)
#         u = self.fc2(u)
#         return u

#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

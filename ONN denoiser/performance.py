import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import math
def norm(x):
    return torch.sqrt(torch.dot(x, x))
torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# ------------------Optical parameters---------------------
L = 48e-3  # layer 48mm 64lamda
z = 30e-3  # 30mm 40*lamda spacing
lamda = 0.75e-3  # 750um plane source
M = 112  # layer pixel
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)


# ----------------Define the onn------------------
class Spatial_propagation(nn.Module):
    def __init__(self, L, lamda, z):
        super(Spatial_propagation, self).__init__()
        self.L = L
        self.lamda = lamda
        self.z = z

    def forward(self, u1):
        # 菲涅尔衍射空间传递函数
        M, N = u1.shape[-2:]
        dx = self.L / M
        k = 2 * np.pi / self.lamda
        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, device=u1.device, dtype=u1.dtype).to(device)
        FX, FY = torch.meshgrid(fx, fx)
        H = torch.exp(-j * np.pi * self.lamda * self.z * (FX ** 2 + FY ** 2)).to(device) * torch.exp(j * k * self.z).to(
            device)
        H = torch.fft.fftshift(H, dim=(-2, -1))
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2, -1))).to(device)
        U2 = H * U1
        u2 = torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2, -1))
        return u2


class Phase_modulation(nn.Module):
    def __init__(self, M, N):
        super(Phase_modulation, self).__init__()
        self.phase_values = nn.Parameter(torch.zeros((M, N)))
        nn.init.uniform_(self.phase_values, a=0, b=2)

    def forward(self, input_tensor):
        modulation_matrix = torch.exp(j * 2 * np.pi * self.phase_values)
        modulated_tensor = input_tensor * modulation_matrix
        return modulated_tensor


class imaging_layer(nn.Module):
    def __init__(self):
        super(imaging_layer, self).__init__()

    def forward(self, u):
        # Calculate the intensity
        intensity = torch.abs(u) ** 2
        return intensity



class ONN(nn.Module):
    def __init__(self, M, L, lamda, z):
        super(ONN, self).__init__()

        # 5 Propagation and Modulation layers interleaved
        layers = []
        for _ in range(6):
            layers.append(Spatial_propagation(L, lamda, z))
            layers.append(Phase_modulation(M, M))

        self.optical_layers = nn.Sequential(*layers)
        self.image = imaging_layer()


    def forward(self, x):
        x = self.optical_layers(x)
        x = self.image(x)
        return x

# Initialize network, loss, and optimizer
model = ONN(M, L, lamda, z).to(device)
model.load_state_dict(torch.load("E:/Users/Winshare Tang/Desktop/results/denoiser_large_uniform1 (2).pt",map_location=torch.device(device)))
model.eval()
path = "E:/Users/Winshare Tang/Desktop/pythonProject/data/"
train_data = np.load(path + "train_noise.npy", allow_pickle=True)
train_label_mod = np.load(path + "train_112.npy", allow_pickle=True)
test_data = np.load(path + "test_noise.npy", allow_pickle=True)
test_label_mod = np.load(path + "test_112.npy", allow_pickle=True)

train_data = torch.from_numpy(train_data).double()
train_label_mod = torch.from_numpy(train_label_mod).double()
test_data = torch.from_numpy(test_data).double()
test_label_mod = torch.from_numpy(test_label_mod).double()

# Transpose the data
train_data_transposed = train_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
train_label_transposed = train_label_mod.permute(2, 0, 1)  # Now shape [1000, 10]
test_data_transposed = test_data.permute(2, 0, 1)  # Now shape [1000, 250, 250]
test_label_transposed = test_label_mod.permute(2, 0, 1)  # Now shape [1000, 10]

# Create the TensorDataset
train_data_transposed = train_data_transposed.to(device)
test_data_transposed = test_data_transposed.to(device)

# -----------------------SSIM INDEX------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand( channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel,size_average = True):

    img1 = torch.reshape(img1,[1, 1, M, M])
    img1 = img1.type(torch.cuda.FloatTensor)
    img2 = torch.reshape(img2, [1, 1, M, M])
    img2 = img2.type(torch.cuda.FloatTensor)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average==True :
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



# change this line to the data you want to predict
num = input("select the image num in range(200): ")

num = int(num)
u0 = test_data_transposed[num]
# add an extra dim before u0, value=1
print('u0',u0.shape)
plt.subplot(3, 2, 1)
plt.imshow(u0)
plt.title("noise image")
u1 = u0
u1 = u1.unsqueeze(0)
output = model(u1)
plt.subplot(3, 2, 2)
plt.imshow(output[0])
plt.title("denoise image")
plt.savefig("combined_figure.png")
index1 = _ssim(output,u1,create_window(11,1).to(device),6,torch.tensor(1))
print(index1)

u0 = test_data_transposed[num+1]
plt.subplot(3, 2, 3)
plt.imshow(u0)
plt.title("noise image")
u1 = u0
u1 = u1.unsqueeze(0)
output = model(u1)
plt.subplot(3, 2, 4)
plt.imshow(output[0])
plt.title("denoise image")
plt.savefig("combined_figure.png")
index2 = _ssim(output,u1,create_window(11,1).to(device),6,torch.tensor(1))
print(index2)

u0 = test_data_transposed[num+2]
plt.subplot(3, 2, 5)
plt.imshow(u0)
plt.title("noise image")
u1 = u0
u1 = u1.unsqueeze(0)
output = model(u1)
plt.subplot(3, 2, 6)
plt.imshow(output[0])
plt.title("denoise image")
plt.savefig("combined_figure.png")
index3 = _ssim(output,u1,create_window(11,1).to(device),6,torch.tensor(1))
print(index1,index2,index3)
plt.show()
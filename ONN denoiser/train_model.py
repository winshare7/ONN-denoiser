import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

import math

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} ")

# ------------------constant parameters---------------------
L = 48e-3  # layer 48mm 64lamda
z = 30e-3  # 30mm 40*lamda spacing
lamda = 0.75e-3  # 750um plane source
M = 112  # layer pixel
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)
batch_size = 128
learning_rate = 0.003
epochs = 35
# -------------------data loader----------------------------
path = ''  # file path
class MyCustomNpyDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data_file = np.load(data_file, mmap_mode='r')
        self.label_file = np.load(label_file, mmap_mode='r')
        self.data_len = np.load(data_file, mmap_mode='r').shape[2]  # Assuming data shape is [250, 250, num_samples]

    def __len__(self):
        return self.data_len

    def __getitem__(self, dx):
        data = self.data_file[:, :, dx]
        labels = self.label_file[:, :, dx]
        return torch.tensor(data).double().to(device), torch.tensor(labels).double().to(device)
    


train_dataset = MyCustomNpyDataset(path + "train_noise.npy", path + "train_112.npy")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MyCustomNpyDataset(path+"test_noise.npy", path+"test_112.npy")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = MyCustomNpyDataset(path+"val_noise.npy", path+"val_112.npy")
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
print("data loaded")


# ----------------Define the onn------------------
def norm(x):
    return torch.sqrt(torch.dot(x, x)+0.00000001)

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
        H = torch.exp(-j * np.pi * self.lamda * self.z * (FX**2 + FY**2)).to(device) * torch.exp(j * k * self.z).to(device)
        H = torch.fft.fftshift(H, dim=(-2,-1))
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2,-1))).to(device)
        U2 = H * U1
        u2 = torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2,-1))
        return u2

class Phase_modulation(nn.Module):
    def __init__(self, M, N):
        super(Phase_modulation, self).__init__()
        self.phase_values = nn.Parameter(torch.zeros((M, N)))
        nn.init.uniform_(self.phase_values, a = 0, b = 2)
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
    def __init__(self, M, L, lmbda, z):
        super(ONN, self).__init__()
        
        # 6 layer
        layers = []
        for _ in range(6):
            layers.append(Spatial_propagation(L, lmbda, z))
            layers.append(Phase_modulation(M, M))
        
        self.optical_layers = nn.Sequential(*layers)
        self.image = imaging_layer()

    def forward(self, x):
        x = self.optical_layers(x)
        x = self.image(x)
        return x

model = ONN(M, L, lamda, z).to(device)

# ----------------loss function---------------------
def denoiser_loss(outputs,labels):
    k = torch.sum(labels)/torch.sum(outputs)
    n = 100*torch.sum(outputs)/torch.sum(labels)
    N = torch.numel(outputs)
    l = torch.sum(abs(labels-k*outputs))/N + 0.05*math.exp(-n)
    return l
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        batch = outputs.shape[0]
        loss = denoiser_loss(outputs, batch_labels)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # Evaluation phase
    model.eval()
    # correct = 0
    # total = 0
    with torch.no_grad():
        for data, labels in train_loader:
            outputs = model(data)
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, labels in validation_loader:
            outputs = model(data)
            loss = denoiser_loss(outputs, labels)
            val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(validation_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, \nValidation Loss: {avg_val_loss:.4f}")
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_running_loss = 0
    for data, labels in test_loader:
        outputs = model(data)
        loss = denoiser_loss(outputs, labels)
        test_running_loss += loss.item()

print(f"denoiser performance: {test_running_loss:.2f}")
torch.save(model.state_dict(), "denoiser_large_uniform1.pt")

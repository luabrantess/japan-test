import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ==== Configuracoes ====
batch_size = 128
epochs = 10
latent_dim = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset MNIST ====
transform = transforms.ToTensor()
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# ==== Modelo CVAE ====
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x, y):
        y_onehot = F.one_hot(y, num_classes=10).float()
        x = torch.cat([x.view(-1, 28*28), y_onehot], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=10).float()
        z = torch.cat([z, y_onehot], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ==== Funcoes de perda ====
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ==== Treinamento ====
model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(e)
pochs):
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, target)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cvae_mnist.pth")

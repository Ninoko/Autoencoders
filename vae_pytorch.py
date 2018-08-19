from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
torch.manual_seed(101)

batch_size = 32
input_dim = 28 * 28
num_epochs = 10
torch.set_num_threads(1)

transform = transforms.Compose([transforms.Resize(input_dim),
                                transforms.ToTensor()])

dataset = dset.MNIST(root='./data/mnist/', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.xavier_normal_(0., 1.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.xavier_normal_(0., 1.)
        m.bias.data.fill_(0)

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x) * input_dim
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=16, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, h_dim)
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

vae = VAE()

optimizer = optim.Adam(vae.parameters(), lr=0.00005)

for epoch in range(num_epochs):
    for batch, data in enumerate(dataloader):
        x, _ = data
        x = Variable(x)

        output, mu, logvar = vae(x)
        loss = vae_loss(output, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch [{}/{}]\tBatch:{} \tloss:{:.4f}\tMSE:{:.4f}".format(epoch+1, num_epochs, batch+1, loss.data.item(), F.mse_loss(x, output.data)))
    print('epoch [{}/{}]\tloss:{:.4f}'.format(epoch + 1, num_epochs, loss.data.item()))

torch.save(vae.state_dict(), './vae.pth')
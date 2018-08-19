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
num_samples = 20
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

def iwae_loss(outputs, x, mu, logvar):
    loss = 0.
    costs = []
    for output in outputs:
        MSE = F.mse_loss(output, x) * input_dim
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        costs.append(MSE+KLD)
    weights = F.softmax(torch.Tensor(costs)).data
    weights = Variable(weights, requires_grad = False)
    for i, cost in enumerate(costs):
        loss += cost * weights[i]

    return loss

class IWAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=16, z_dim=2, num_samples=num_samples):
        super(IWAE, self).__init__()
        self.num_samples = num_samples
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
        z_samples = []
        for i in range(self.num_samples):
            esp = torch.randn(*mu.size())
            z_samples.append(mu + std * esp)
        return z_samples
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z_samples = self.reparameterize(mu, logvar)
        return z_samples, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z_samples, mu, logvar = self.bottleneck(h)
        return z_samples, mu, logvar

    def decode(self, z_samples):
        outputs = []
        for i, z in enumerate(z_samples):
            z = self.fc3(z)
            outputs.append(self.decoder(z))
        return outputs

    def forward(self, x):
        z_samples, mu, logvar = self.encode(x)
        outputs = self.decode(z_samples)
        return outputs, mu, logvar

iwae = IWAE()

optimizer = optim.Adam(iwae.parameters(), lr=0.00005)

for epoch in range(num_epochs):
    for batch, data in enumerate(dataloader):
        x, _ = data
        x = Variable(x)

        output, mu, logvar = iwae(x)
        loss = iwae_loss(output, x, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch [{}/{}]\tBatch:{} \tloss:{:.4f}\tMSE:{:.4f}".format(epoch+1, num_epochs, batch+1, loss.data.item(), F.mse_loss(x, output[0].data)))
    print('epoch [{}/{}]\tloss:{:.4f}'.format(epoch + 1, num_epochs, loss.data.item()))

torch.save(iwae.state_dict(), './iwae.pth')
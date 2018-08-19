from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

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

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(16, input_dim),
			nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

shallow_ae = ShallowAutoencoder(input_dim).cuda()
shallow_ae.apply(weights_init)

criterion = nn.BCELoss()

optimizer = optim.Adam(shallow_ae.parameters(), lr=0.0005)

for epoch in range(num_epochs):
    for data in dataloader:
        x, _ = data
        x = Variable(x).cuda()

        output = shallow_ae(x)
        loss = criterion(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data.item()))

torch.save(shallow_ae.state_dict(), './shallow_ae.pth')


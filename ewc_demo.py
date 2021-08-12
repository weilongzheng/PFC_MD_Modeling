import torch
import torch.nn as nn
import torch.nn.functional as F
from elastic_weight_consolidation import ElasticWeightConsolidationOriginal

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(self.f1(x))))

crit = nn.CrossEntropyLoss()
ewc = ElasticWeightConsolidationOriginal(BaseModel(28 * 28, 100, 10), crit=crit, lr=1e-4)
from tqdm import tqdm

for _ in range(4):
    for input, target in tqdm(train_loader):
        #import pdb;pdb.set_trace()
        ewc.forward_backward_update(input, target)    
        
ewc.register_ewc_params(mnist_train, 100, 300)
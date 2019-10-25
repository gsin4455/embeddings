'''simple cnn in Pytorch.'''

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHAN = 2

class OrthLinear(nn.Linear):
    def __init__(self, *args, alpha=None,manifold=geoopt.Euclidean(), **kwargs):
        super().__init__(*args, **kwargs)
        weight = self._parameters.pop("weight")
        self._weight_shape = weight.shape
        self.weight_orig = geoopt.ManifoldParameter(
            weight.data.reshape(weight.shape[0], -1), manifold=manifold)
        with torch.no_grad():
            self.weight_orig.proj_()

        if alpha is None:
            self.alpha_orig = nn.Parameter(torch.zeros(self._weight_shape[0]))
        else:
            self.alpha_orig = alpha

    @property
    def weight(self):
        return (self.alpha[:, None] * self.weight_orig).reshape(self._weight_shape)

    @property
    def alpha(self):
        return self.alpha_orig.exp()



class OrthConv1d(nn.Conv1d):
    def __init__(self, *args, alpha=None,manifold=geoopt.Euclidean(), **kwargs):
        super().__init__(*args, **kwargs)
        weight = self._parameters.pop("weight")
        self._weight_shape = weight.shape
        self.weight_orig = geoopt.ManifoldParameter(
            weight.data.reshape(weight.shape[0], -1), manifold=manifold)
        with torch.no_grad():
            self.weight_orig.proj_()
            
        if alpha is None:
            self.alpha_orig = nn.Parameter(torch.zeros(self._weight_shape[0]))
        else:
            self.alpha_orig = alpha

    @property
    def weight(self):
        return (self.alpha[:, None] * self.weight_orig).reshape(self._weight_shape)

    @property
    def alpha(self):
        return self.alpha_orig.exp()




class net(nn.Module):
    def __init__(self, classes,manifold):
        super(net, self).__init__()
        self.conv1 = OrthConv1d(IN_CHAN,64,kernel_size=4,stride=2,padding=0,manifold=manifold)
        self.conv2 = OrthConv1d(64,96,kernel_size=3,stride=2,padding=0,manifold=manifold)
        self.pool = nn.MaxPool1d(3,stride=2)

        self.conv3 = OrthConv1d(96,128, kernel_size=3, stride=1, padding=0,manifold=manifold)
        self.conv4 = OrthConv1d(128, 256, kernel_size=3, stride=1, padding=0,manifold=manifold)
        
        self.fc = OrthLinear(123*256, 48,manifold=manifold)
        self.do = nn.Dropout()
        self.out = OrthLinear(48, classes,manifold=manifold)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc(self.do(x)))
        #x = F.relu(self.fc(x))
            
        x = self.out(x)
        return x

if __name__=='__main__':
    #Test code
    nn = net(2,manifold=geoopt.manifolds.PoincareBall())
    #nn = net(2,manifold=geoopt.manifolds.Euclidean())
    #nn = net(2,manifold=geoopt.manifolds.Stiefel())
    
    x = torch.randn(128,1024,2)
    
    y = nn(x)
    
    opt = geoopt.optim.RiemannianAdam(nn.parameters(),lr=1,stabilize=1)
    opt.zero_grad()
    opt.step()
    print(nn)

    #print(y)
    

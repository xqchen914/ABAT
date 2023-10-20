import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(model, Chans, Samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, Chans, Samples).to(device)
    out = model(x)
    return out.shape[-1]


def LoadModel(model_name, Classes, Chans, Samples,F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.5, midDim=40,d1=25,d2=50,d3=100):
    if model_name == 'EEGNet':
        modelF = EEGNet(Chans=Chans,
                       Samples=Samples,
                       kernLenght=64,
                       F1=F1,
                       D=D,
                       F2=F2,
                       dropoutRate=dropoutRate)
    elif model_name == 'DeepCNN':
        modelF = DeepConvNet(Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,d1=d1,d2=d2,d3=d3)
    elif model_name == 'ShallowCNN':
        modelF = ShallowConvNet(Chans=Chans, Samples=Samples, dropoutRate=dropoutRate, midDim=midDim)
    else:
        raise 'No such model'
    embed_dim = CalculateOutSize(modelF, Chans, Samples)
    modelC = Classifier(embed_dim, Classes)
    modelD = DomainDiscriminator(embed_dim, hidden_dim=128)
    return modelF, modelC, modelD, embed_dim




class EEGNet(nn.Module):
    """
    :param
    """
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate: Optional[float] = 0.5,
                 SAP_frac: Optional[float] = None):
        super(EEGNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.SAP_frac = SAP_frac

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)

    def pred_ent(self, x):
        logits = self(x)
        lsm = nn.LogSoftmax(dim=-1)
        log_probs = lsm(logits)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        predictive_entropy = -p_log_p.sum(axis=1)
        return predictive_entropy



        
class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 dropoutRate: Optional[float] = 0.5,
                 d1: Optional[int] = 25,
                 d2: Optional[int] = 50,
                 d3: Optional[int] = 100):
        super(DeepConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=d1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d3), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5, midDim: Optional[int] = 40,
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=midDim, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=midDim,
                      out_channels=midDim,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=midDim), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'), 
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.block = nn.Sequential(
            nn.Linear(in_features=self.input_dim,
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, feature):
        output = self.block(feature)

        return output

    def MaxNormConstraint(self):
        for n, p in self.block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)



    
class DomainDiscriminator(nn.Module):
    """
    Domain discriminator module - 2 layers MLP

    Parameters:
        - input_dim (int): dim of input features
        - hidden_dim (int): dim of hidden features
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super(DomainDiscriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
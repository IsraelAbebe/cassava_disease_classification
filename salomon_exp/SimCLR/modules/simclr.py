import torch.nn as nn
import torchvision
import pretrainedmodels


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args

        self.encoder = self.get_resnet(args.resnet)

#         self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer       
#         self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer
        
        self.n_features = self.encoder.last_linear.in_features  # get dimensions of fc layer
        self.encoder.last_linear = Identity()  # remove fully-connected layer after pooling layer


        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, args.projection_dim, bias=False),
        )
        

    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(pretrained=True),
            "resnet50": torchvision.models.resnet50(pretrained=True),
            
            "se_resnet50" : pretrainedmodels.se_resnet50(num_classes=1000, pretrained="imagenet"),
            "se_resnext101_32x4d" : pretrainedmodels.se_resnext101_32x4d(num_classes=1000, pretrained="imagenet"),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid Model (ResNet) version")
        return resnets[name]


    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z

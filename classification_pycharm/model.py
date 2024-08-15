import torch
import timm
import torch.nn as nn
from timm import create_model
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights, resnet18, ResNet101_Weights, resnet101
from torchvision.models import resnet152, ResNet152_Weights
import torchvision.models as models
from transformers import ConvNextForImageClassification,  ConvNextConfig
from torchvision.models import convnext_large
from torchvision.models.convnext import ConvNeXt_Large_Weights

# first model tested
# ResNet152 (larger and better than ResNet18, ResNet50)
class CModel:
    def __init__(self):
        # use the latest weight enumeration type
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        return self.model(x)

    def to(self, device):
        self.model = self.model.to(device)
        return self.model

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        return self.model(x)


class GoogleModel:
    def __init__(self):
        # GoogleNet-v4
        self.model = models.googlenet(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(num_ftrs, 5)
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        return self.model(x)

    def to(self, device):
        self.model = self.model.to(device)
        return self.model

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        return self.model(x)



class ViTModel:
    def __init__(self, num_classes=5, pretrained=True):
        super(ViTModel, self).__init__()
        # pretrained Vision Transformer
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

    def load(self, path):
        self.vit.load_state_dict(torch.load(path))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.vit(x)

    def to(self, device):
        self.vit = self.vit.to(device)
        return self

    def parameters(self):
        return self.vit.parameters()

    def train(self):
        self.vit.train()

    def eval(self):
        self.vit.eval()

    def __call__(self, x):
        return self.vit(x)




class SwinTModel:
    def __init__(self, num_classes=5):
        super(SwinTModel, self).__init__()
        # pretrained Swin Transformer
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        num_ftrs = self.model.head.in_features

        self.model.head = nn.Linear(num_ftrs, num_classes)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, x):
        return self.model(x)

    def to(self, device):
        # move the model to the specified device (GPU or CPU)
        self.model = self.model.to(device)
        return self.model

    def __call__(self, x):
        return self.model(x)


class ConvNeXtModel(nn.Module):
    def __init__(self):
        super().__init__()
        # initialise ConvNeXt, use pretrained convnext_large
        # self.model = models.convnext_large(pretrained=True)
        self.model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # modify last layer of classifier to conduct 5-class-classification
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.15),  # add dropout layer
            nn.Linear(in_features, 5)
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    # default forward passing
    def forward(self, x):
        return self.model(x)

    def to(self, device):
        # moves the model to the specified device
        self.model = self.model.to(device)
        return self.model

    def __call__(self, x):
        # define the behavior of model calls
        return self.forward(x)

    # forward passing to make prediction
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self(x)
        return outputs


class ConvNeXtFModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # also use the convnext_large model
        self.model = models.convnext_large(pretrained=False)

        # freeze the weights for all layers except the top
        for name, parameter in self.model.named_parameters():
            #  we decide to fine-tune only the last Sequential block and classifier layer
            if not (name.startswith('features.6') or name.startswith('features.7')):
                parameter.requires_grad = False
        # define the weight path
        weights_path = 'convnext_large_22k_224.pth'
        pretrained_weights = torch.load(weights_path)
        # can use the pretrained wight
        # self.model.load_state_dict(pretrained_weights)


        # print the name of each model weight
        # (this is because we should change the name of which parameter to freeze, so we should get their names first)
        print("\nname os model wights:")
        if 'model' in pretrained_weights:
            pretrained_weights = pretrained_weights['model']
        for key in pretrained_weights.keys():
            print(key)

        # modify the classifier to accommodate the new classification task
        # get the number of input features for the original classifier
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(in_features, num_classes)
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        return self.model(x)

    def to(self, device):
        # moves the model to the specified device
        self.model = self.model.to(device)
        return self.model

    def __call__(self, x):
        # define the behavior of model calls
        return self.model(x)


# linear classifier might to simple in parameters are frozen
# so the complex classifier could be used to replace the linear classifier
class ComplexClassifier(nn.Module):
    # change the structure here
    # in_fearures to 256 to 64 to 5
    def __init__(self, in_features, output_features):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(64, output_features)

    # forward passing
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


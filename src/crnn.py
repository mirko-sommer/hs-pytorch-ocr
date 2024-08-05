import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0):
        super(SimpleCNN, self).__init__()
        
        self.layer1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3, dropout_rate=dropout_rate)  # Similar to the first conv layer of ResNet
        self.layer2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate)
        self.layer3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate)
        self.layer4 = ConvBlock(256, 256, kernel_size=3, stride=2, padding=1, dropout_rate=dropout_rate)

    def forward(self, batch):
        batch = self.layer1(batch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        return batch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dropout_rate=0.5):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,  
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        

    def forward(self, batch):
        batch = self.conv(batch)
        batch = self.bn(batch)
        batch = F.relu(batch)
        batch = self.dropout(batch)
        return batch

class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional):
        super(GRUBlock, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        
    def forward(self, batch, add_output=False):
        outputs, _ = self.gru(batch)
        out_size = int(outputs.size(2) / 2)
        if add_output:
            outputs = outputs[:, :, :out_size] + outputs[:, :, out_size:]
        return outputs
    

class CRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, bidirectional=True, dropout=0, use_resnet18=True, use_pretrained=True):
        super(CRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Convolutional Layers
        if use_resnet18:
            if use_pretrained:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                modules = list(resnet.children())[:-3]
                self.cnn = nn.Sequential(*modules)
            else:
                resnet = resnet18()
                modules = list(resnet.children())[:-3]
                self.cnn = nn.Sequential(*modules)
        else:
            if use_pretrained:
                raise NotImplementedError("There does not exist a pretrained simple CNN model.")
            else:
                self.cnn = SimpleCNN(dropout_rate=dropout)
        
        self.conv_block = ConvBlock(256, 256, kernel_size=3, padding=1, dropout_rate=dropout) # Maybe delete for simplecnn?
        
        # Linear and RNN Layers
        self.fc1 = nn.Linear(1024, 256)
        self.gru1 = GRUBlock(256, hidden_size, bidirectional=bidirectional)
        self.gru2 = GRUBlock(hidden_size, hidden_size, bidirectional=bidirectional)
        self.fc2 = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, batch: torch.Tensor):
        batch_size = batch.size(0)

        # Convolutional Layers
        batch = self.cnn(batch)
        batch = self.conv_block(batch)

        # Reshape and linear layer
        batch = batch.permute(0, 3, 1, 2)
        n_channels = batch.size(1)
        batch = batch.view(batch_size, n_channels, -1)
        batch = self.fc1(batch)

        # RNN layers
        batch = self.gru1(batch, add_output=True)
        batch = self.gru2(batch)

        # Final linear Layers
        batch = self.fc2(batch)
        batch = batch.permute(1, 0, 2)

        return batch
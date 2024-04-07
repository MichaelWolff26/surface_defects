# Manual small Conv Network with one residual connection for first testing
import torch.nn as nn


class conv_network(nn.Module):
    def __init__(self):
        super(conv_network, self).__init__()
        self.convlayer1 = nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=5, stride=2, padding=0
        )
        self.relu1=nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.convlayer2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=0
        )
        self.relu2=nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.drop_out = nn.Dropout(0.2)
        self.relu3=nn.ReLU()
        self.flatten=nn.Flatten()

        self.dense1 = nn.Linear(in_features=3168, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.relu4=nn.ReLU()
        self.dense3 = nn.Linear(in_features=512, out_features=128)
        self.relu5=nn.ReLU()
        self.dense4 = nn.Linear(in_features=128, out_features=2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, X):
        X = self.convlayer1(X)
        X = self.relu1(X)
        X = self.max_pool1(X)
        X = self.convlayer2(X)
        X = self.relu2(X)
        X = self.max_pool2(X)
        X = self.flatten(X)
        X = self.dense1(X)
        X = self.relu3(X)
        X = self.drop_out(X)
        residual = X
        X = self.dense2(X)
        X = self.relu4(X)
        X = self.drop_out(X)
        X = X + residual
        X = self.dense3(X)
        X = self.relu5(X)
        X = self.drop_out(X)
        X = self.dense4(X)
        X = self.sigmoid(X)
        return X
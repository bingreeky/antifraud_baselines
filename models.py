import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_data
from tqdm import tqdm


class cnn_max(nn.Module):

    def __init__(
        self,
        in_channels: int = 1
    ) -> None:
        super(cnn_max, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(2, 2)
        )
        # relu here
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(2, 2)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        # flatten here
        self.flatten = nn.Flatten()

        self.linears = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=2)
        )

        # self.linear1 = nn.LazyLinear(out_features=128)
        # # relu here
        # self.linear2 = nn.LazyLinear(out_features=64)
        # # relu here
        # self.linear3 = nn.LazyLinear(out_features=2)
        # # output logits

    def forward(self, x):
        # x shape be like: (|B|, C_in, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.conv2(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.flatten(x)
        logits = self.linears(x)

        return logits


class modelHandler():
    def __init__(
        self,
        model_name: str = "cnn-max",
        epochs: int = 30,
        batch_szie: int = 512,
        device: str = "cpu"
    ) -> None:

        # prepare model
        if model_name == "cnn-max":
            self.model = cnn_max()
        else:
            raise NotImplementedError

        # training info
        self.epochs = epochs
        self.batch_size = batch_szie

        # set device
        if device == "cpu" or (not torch.cuda.is_available()):
            self.device = "cpu"
        else:
            self.device = "cuda"

        # prepare data
        feat_train, feat_test, label_train, label_test = load_data(model_name)
        self.train_data = (feat_train, label_train)
        self.test_data = (feat_test, label_test)

    def train(self):
        # prepare data and model
        if self.device != "cpu":
            self.model.cuda()
            features = torch.from_numpy(self.train_data[0]).to(
                dtype=torch.float32).cuda()
            labels = torch.from_numpy(self.train_data[1]).to(
                dtype=torch.long).cuda()
        else:
            features = torch.from_numpy(
                self.train_data[0]).to(dtype=torch.float32)
            labels = torch.from_numpy(self.train_data[1]).to(dtype=torch.long)

        # normalize data
        features = F.normalize(features, dim=3)

        # optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-3,
            momentum=0.9,
            nesterov=True
        )

        # loss func
        loss_func = nn.CrossEntropyLoss()

        # begin training
        batch_num = math.floor(len(labels) / self.batch_size)
        for epoch in (range(self.epochs)):

            loss = 0.0

            # mini-batch trainig
            for batch in tqdm(range(batch_num)):
                # i do not want to use dataloader...
                batch_mask = list(
                    range(batch*self.batch_size, (batch+1)*self.batch_size))
                output = self.model(features[batch_mask])

                batch_loss = loss_func(output, labels[batch_mask])
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()

            print(f"Epoch: {epoch}, loss: {loss / batch_num}")

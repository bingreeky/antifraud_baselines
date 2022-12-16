from math import floor, ceil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score
from utils import load_data, to_pred


class cnn_max(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        cate_feat_num: int = 3,
        cate_embed_dim: int = 8,
        cate_unique_num: list = [1664, 216, 2500]
    ) -> None:
        super(cnn_max, self).__init__()

        self.cate_feat_num = cate_feat_num
        self.cate_embed_dim = cate_embed_dim
        self.cate_unique_num = cate_unique_num

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

        # cate embedding layer
        # ['Location','Type','Target']
        self.cate_emdeds = nn.ModuleList([nn.Embedding(
            cate_unique_num[idx] + 1, cate_embed_dim) for idx in range(cate_feat_num)])

    def cate_embedding_layer(self, X_cate):
        cate_embed_ret = []
        for idx in range(self.cate_feat_num):
            cate_embed_ret.append(self.cate_emdeds[idx](X_cate[:, idx]))
            # 3个(batch_size, emb_dim)

        cate_embed_ret = torch.concat(cate_embed_ret, dim=1)
        return cate_embed_ret  # (batch_size, emb_dim*3)

    def forward(self, x_nume, x_cate):
        # x shape be like: (batch_size, time_windows_dim, feat_dim)

        x = x_nume.unsqueeze(1)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.conv2(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.flatten(x)

        cate_embeds = self.cate_embedding_layer(x_cate)
        x = torch.concat([x, cate_embeds], dim=1)
        logits = self.linears(x)

        return logits


def cnn_model(
    cate_x_train,
    nume_x_train,
    y_train,
    cate_x_test,
    nume_x_test,
    y_test,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu"
):
    model = cnn_max()
    model.to(device)

    nume_feats = torch.from_numpy(nume_x_train).to(
        dtype=torch.float32).to(device)
    nume_feats.transpose_(1, 2)
    cate_feats = torch.from_numpy(cate_x_train).to(dtype=torch.long).to(device)
    labels = torch.from_numpy(y_train).to(dtype=torch.long).to(device)

    # anti label imbalance
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts)*len(labels)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    batch_num = ceil(len(labels) / batch_size)
    for epoch in range(epochs):

        loss = 0.
        pred = []

        for batch in tqdm(range(batch_num)):
            optimizer.zero_grad()

            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels))))

            output = model(nume_feats[batch_mask], cate_feats[batch_mask])

            batch_loss = loss_func(output, labels[batch_mask])
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            # print(to_pred(output))
            pred.extend(to_pred(output))

        true = labels.cpu().numpy()
        pred = np.array(pred)
        print(
            f"Epoch: {epoch}, loss: {loss / batch_num}, auc: {roc_auc_score(true, pred)}, F1: {f1_score(true, pred, average='macro')}, AP: {average_precision_score(true, pred)}")
        print(confusion_matrix(true, pred))

    cate_feats_test = torch.from_numpy(
        cate_x_test).to(dtype=torch.long).to(device)
    nume_feats_test = torch.from_numpy(
        nume_x_test).to(dtype=torch.float32).to(device)
    nume_feats_test.transpose_(1, 2)
    labels_test = torch.from_numpy(y_test).to(dtype=torch.long)

    batch_num_test = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in tqdm(range(batch_num)):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(
                nume_feats_test[batch_mask], cate_feats_test[batch_mask])
            pred.extend(to_pred(output))

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred)}, F1: {f1_score(true, pred, average='macro')}, AP: {average_precision_score(true, pred)}")
        print(confusion_matrix(true, pred))


if __name__ == "__main__":
    cate_feats_train, nume_feats_train, cate_feats_test, nume_feats_test, label_train, label_test = load_data(
        "att_cnn_2d")

    cnn_model(cate_feats_train, nume_feats_train, label_train,
              cate_feats_test, nume_feats_test, label_test, device="cuda:0")

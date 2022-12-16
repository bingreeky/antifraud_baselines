import numpy as np
import torch
import torch.nn.functional as F
from math import floor, ceil
from tqdm import tqdm
from att_cnn_2d import Att_cnn2d_model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score
from utils import load_data


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def att_train_2d(
    cate_x_train,
    nume_x_train,
    y_train,
    cate_x_test,
    nume_x_test,
    y_test,
    num_classes: int = 2,
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    model = Att_cnn2d_model(
        time_windows_dim=nume_x_train.shape[2],
        feat_dim=nume_x_train.shape[1],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )
    model.to(device)

    # train
    cate_feats = torch.from_numpy(cate_x_train).to(dtype=torch.long).to(device)

    nume_feats = torch.from_numpy(nume_x_train).to(
        dtype=torch.float32).to(device)
    # (sample_num, feat_dim, time_windows)
    nume_feats.transpose_(1, 2)
    # (sample_num, time_windows, feat_dim)
    labels = torch.from_numpy(y_train).to(dtype=torch.long)

    cate_feats.requires_grad = False
    nume_feats.requires_grad = False
    labels.requires_grad = False

    cate_feats.to(device)
    nume_feats.to(device)
    labels = labels.to(device)

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

    np.random.seed(22)
    torch.cuda.manual_seed_all(22)
    torch.manual_seed(22)

    # cat_f = np.load("./data/STRAD_2d_cate_features.npy", allow_pickle=True)
    # num_f = np.load("./data/STRAD_2d_nume_features.npy", allow_pickle=True)
    # labels = np.load("./data/STRAD_labels.npy", allow_pickle=True)

    cate_feats_train, nume_feats_train, cate_feats_test, nume_feats_test, label_train, label_test = load_data(
        "att_cnn_2d")
    att_train_2d(cate_feats_train, nume_feats_train, label_train,
                 cate_feats_test, nume_feats_test, label_test, device="cuda:0")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def load_data(
    model: str = "cnn_max",
    test_ratio: float = 0.25,
    valid_ratio: float = 0.15,
    rand_seed: int = 42
):
    """load data fro cnn-max or STAN
    merely consider train/test set

    Args:
        model (str, optional): 2D or 3D data. Defaults to "cnn_max".
        test_ratio (float, optional): Defaults to 0.25.
        valid_ratio (float, optional): Defaults to 0.15.
        rand_seed (int, optional): Defaults to 42.

    Returns:
        _type_: _description_
    """
    if model == "cnn_max":
        features = np.load("./data/STRAD_2d.npy")
        labels = np.load("./data/STRAD_labels.npy")

        # feat_train, feat_rest, label_train, label_rest = train_test_split(features, labels, train_size=(1-test_ratio-valid_ratio), random_state=rand_seed)
        # feat_valid, feat_test, label_valid, label_test = train_test_split(feat_rest, label_rest, train_size=(valid_ratio)/(test_ratio+valid_ratio), random_state=rand_seed)
        feat_train, feat_rest, label_train, label_rest = train_test_split(
            features, labels, train_size=1-test_ratio, stratify=labels, random_state=rand_seed, shuffle=True)
        return feat_train, feat_rest, label_train, label_rest

    elif model == "att_cnn_2d":
        cate_feats = np.load(
            "./data/STRAD_2d_cate_features.npy", allow_pickle=True)
        nume_feats = np.load(
            "./data/STRAD_2d_nume_features.npy", allow_pickle=True)
        labels = np.load("./data/STRAD_labels.npy", allow_pickle=True)

        feat_train, feat_rest, label_train, label_rest = train_test_split(
            list(zip(cate_feats, nume_feats)), labels, train_size=1-test_ratio, stratify=labels, random_state=rand_seed, shuffle=True)

        cate_feats_train = np.array([rec[0] for rec in feat_train])
        nume_feats_train = np.array([rec[1] for rec in feat_train])
        cate_feats_test = np.array([rec[0] for rec in feat_rest])
        nume_feats_test = np.array([rec[1] for rec in feat_rest])

        return cate_feats_train, nume_feats_train, cate_feats_test, nume_feats_test, label_train, label_rest

    else:
        raise NotImplementedError("Unsupported model.")

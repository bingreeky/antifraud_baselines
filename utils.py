import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(
    model:str = "cnn-max",
    test_ratio:float = 0.25,
    valid_ratio:float = 0.15,
    rand_seed:int = 42
):
    """load data fro cnn-max or STAN
    merely consider train/test set

    Args:
        model (str, optional): 2D or 3D data. Defaults to "cnn-max".
        test_ratio (float, optional): Defaults to 0.25.
        valid_ratio (float, optional): Defaults to 0.15.
        rand_seed (int, optional): Defaults to 42.

    Returns:
        _type_: _description_
    """
    if model == "cnn-max":
        features = np.load("./data/STRAD_2d.npy")
        labels = np.load("./data/STRAD_labels.npy")

    else: 
        raise NotImplementedError("Unsupported model.")

    # feat_train, feat_rest, label_train, label_rest = train_test_split(features, labels, train_size=(1-test_ratio-valid_ratio), random_state=rand_seed)
    # feat_valid, feat_test, label_valid, label_test = train_test_split(feat_rest, label_rest, train_size=(valid_ratio)/(test_ratio+valid_ratio), random_state=rand_seed)
    feat_train, feat_rest, label_train, label_rest = train_test_split(features, labels, train_size=1-test_ratio, random_state=rand_seed)
    return feat_train, feat_rest, label_train, label_rest


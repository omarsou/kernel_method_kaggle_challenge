import pandas as pd
import os


def complementary_letter(letter):
    if letter == 'A':
        return 'T'
    if letter == 'T':
        return 'A'
    if letter == 'C':
        return 'G'
    if letter == 'G':
        return 'C'


def reverse_complementaire(seq):
    rc = ''
    for basis_letter in seq:
        rc = complementary_letter(basis_letter) + rc
    return rc


def return_training_datasets(model_dir):
    Xtr0 = pd.read_csv(os.path.join(model_dir, "Xtr0.csv"))
    Xtr1 = pd.read_csv(os.path.join(model_dir, "Xtr1.csv"))
    Xtr2 = pd.read_csv(os.path.join(model_dir, "Xtr2.csv"))

    Ytr0 = pd.read_csv(os.path.join(model_dir, "Ytr0.csv"))
    Ytr1 = pd.read_csv(os.path.join(model_dir, "Ytr1.csv"))
    Ytr2 = pd.read_csv(os.path.join(model_dir, "Ytr2.csv"))

    Ytr0.loc[Ytr0.Bound == 0, "Bound"] = - 1
    Ytr1.loc[Ytr1.Bound == 0, "Bound"] = - 1
    Ytr2.loc[Ytr2.Bound == 0, "Bound"] = - 1

    X = [Xtr0, Xtr1, Xtr2]
    y = [Ytr0, Ytr1, Ytr2]

    return X, y

def return_inference_datasets(model_dir):
    Xte0 = pd.read_csv(os.path.join(model_dir, "Xte0.csv"))
    Xte1 = pd.read_csv(os.path.join(model_dir, "Xte1.csv"))
    Xte2 = pd.read_csv(os.path.join(model_dir, "Xte2.csv"))

    X = [Xte0, Xte1, Xte2]
    return X

def split_train_val(X, y, frac=0.8, rs=42):
    n = len(X)
    X_, y_ = X.sample(frac=1, random_state=rs), y.sample(frac=1, random_state=rs)
    assert all(X_.index == y_.index)
    X_, y_ = X_.reset_index(drop=True), y_.reset_index(drop=True)
    X_train, X_val = X_.loc[:int(n * frac), :], X_.loc[int(n * frac):, :]
    y_train, y_val = y_.loc[:int(n * frac), :], y_.loc[int(n * frac):, :]
    return X_train, X_val.reset_index(drop=True), y_train, y_val.reset_index(drop=True)


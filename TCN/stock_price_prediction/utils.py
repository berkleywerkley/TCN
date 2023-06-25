import torch
from torch.autograd import Variable
import random
import pandas as pd
import numpy as np

def get_real_test(fname, seq_length):
    df = pd.read_csv(fname)
    df = df.drop(columns=["Date", "Close",  "Volume"])
    df = df.rename(columns={"Adj Close": "Price"})
    df = df.tail(seq_length)
    as_array = df.to_numpy().transpose()
    X = torch.zeros([1, 4, seq_length])
    X[0] = torch.from_numpy(as_array)
    return X


def get_data(fname):
    df = pd.read_csv(fname)
    df.drop(df.tail(100).index, inplace=True)
    df = df.drop(columns=["Date", "Close",  "Volume"])
    df = df.rename(columns={"Adj Close": "Price"})

    df["5dayEWM"] = df["Price"].ewm(span=5, adjust=False).mean()
    df["20dayEWM"] = df["Price"].ewm(span=20, adjust=False).mean()
    df["50dayEWM"] = df["Price"].ewm(span=50, adjust=False).mean()
    df["macd"] = df["5dayEWM"] - df["20dayEWM"]
    df["macd_trigger"] = df["macd"].ewm(span=3, adjust=False).mean()
    df["macd_cd"] = df["macd"] - df["macd_trigger"]

    shift = 5
    df["Price T+5"] = df["Price"].shift(-1 * shift)
    print(df.head())
    df = df.dropna(axis=0)
    as_array = df.to_numpy().transpose()
    return np.array(as_array[0:-1]), as_array[-1]


def build_dataset(data_arrs, target_arr, num_test_cases, seq_length=300):
    X = torch.zeros([num_test_cases, 10, seq_length])
    y = torch.zeros([num_test_cases, 1])
    for example_idx in range(num_test_cases):
        starting_idx = random.randrange(0, len(data_arrs[0]) - seq_length)
        end_idx = starting_idx + seq_length
        random_sub_sequence = np.array([arr[starting_idx:end_idx] for arr in data_arrs])
        if len(random_sub_sequence[0]) != seq_length:
            raise ("incorrect seq length")
        X[example_idx] = torch.from_numpy(random_sub_sequence)
        y[example_idx] = target_arr[end_idx]
    return Variable(X), Variable(y)
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import numpy as np


sys.path.append("../../")
from TCN.stock_price_prediction.model import TCN
from TCN.stock_price_prediction.utils import build_dataset, get_data, get_real_test

BATCH_SIZE = 32
USE_CUDA = False
DROPOUT = 0.3
CLIP = 1
EPOCHS = 10
KSIZE = 2
LEVELS = 8
SEQ_LEN = 300
LOG_INTERVAL = 10
LEARNING_RATE = 3e-3
OPTIMISER = "Adam"
NHID = 150

input_channels = 10
n_classes = 1
batch_size = BATCH_SIZE
seq_length = SEQ_LEN
epochs = EPOCHS

fpath = "UNH.csv"

price, price_t_plus_1 = get_data(fpath)



data, targets = build_dataset(price, price_t_plus_1, 10000, seq_length=seq_length)
n1 = int(0.8 * len(data))
X_train, Y_train = data[:n1], targets[:n1]
X_test, Y_test = data[n1:], targets[n1:]



X_real = get_real_test(fpath, SEQ_LEN)

# TODO Balance data, i.e. 50% wins, 50% losses

channel_sizes = [NHID] * LEVELS
kernel_size = KSIZE
dropout = DROPOUT
model = TCN(
    input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout
)


lr = LEARNING_RATE
optimizer = getattr(optim, OPTIMISER)(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i : (i + batch_size)], Y_train[i : (i + batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            cur_loss = total_loss / LOG_INTERVAL
            processed = min(i + batch_size, X_train.size(0))
            print(
                "Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    processed,
                    X_train.size(0),
                    100.0 * processed / X_train.size(0),
                    cur_loss,
                )
            )
            total_loss = 0


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print("\nTest set: Average loss: {:.6f}\n".format(test_loss.item()))
        return test_loss.item()


for ep in range(1, epochs + 1):
    train(ep)
    tloss = evaluate()

torch.save(model, "UNH.t_plus_5.pt")

# with torch.no_grad():
#     model.eval()
#     to_predict= X_real
#     res = model(to_predict).numpy()[0][0]    
#     print(res)

# to do backtest with previous 30 days worth of data.
# a win is where the price went in the predicted direction
# a loss is when it did not
# record actual vs predicted
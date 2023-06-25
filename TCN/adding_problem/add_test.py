import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append("../../")
from TCN.adding_problem.model import TCN
from TCN.adding_problem.utils import data_generator

# parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
# parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                     help='batch size (default: 32)')
# parser.add_argument('--cuda', action='store_false',
#                     help='use CUDA (default: True)')
# parser.add_argument('--dropout', type=float, default=0.0,
#                     help='dropout applied to layers (default: 0.0)')
# parser.add_argument('--clip', type=float, default=-1,
#                     help='gradient clip, -1 means no clip (default: -1)')
# parser.add_argument('--epochs', type=int, default=10,
#                     help='upper epoch limit (default: 10)')
# parser.add_argument('--ksize', type=int, default=7,
#                     help='kernel size (default: 7)')
# parser.add_argument('--levels', type=int, default=8,
#                     help='# of levels (default: 8)')
# parser.add_argument('--seq_len', type=int, default=400,
#                     help='sequence length (default: 400)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='report interval (default: 100')
# parser.add_argument('--lr', type=float, default=4e-3,
#                     help='initial learning rate (default: 4e-3)')
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer to use (default: Adam)')
# parser.add_argument('--nhid', type=int, default=30,
#                     help='number of hidden units per layer (default: 30)')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed (default: 1111)')
# args = parser.parse_args()

BATCH_SIZE = 32
USE_CUDA = False
DROPOUT = 0.2
CLIP = -1
EPOCHS = 1
KSIZE = 7
LEVELS = 8
SEQ_LEN = 400
LOG_INTERVAL = 100
LEARNING_RATE = 4e-3
OPTIMISER = "Adam"
NHID = 30

input_channels = 2
n_classes = 1
batch_size = BATCH_SIZE
seq_length = SEQ_LEN
epochs = EPOCHS

print("Producing data...")
X_train, Y_train = data_generator(50000, seq_length)
X_test, Y_test = data_generator(1000, seq_length)


# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [NHID] * LEVELS
kernel_size = KSIZE
dropout = DROPOUT
model = TCN(
    input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout
)

lr = LEARNING_RATE
optimizer = getattr(optim, OPTIMISER)(model.parameters(), lr=lr)


def train(epoch):
    global lr
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
                "Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}".format(
                    epoch,
                    processed,
                    X_train.size(0),
                    100.0 * processed / X_train.size(0),
                    lr,
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

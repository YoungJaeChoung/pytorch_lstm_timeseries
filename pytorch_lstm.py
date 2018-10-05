# http://chandlerzuo.github.io/blog/2017/11/darnn
# https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
# https://gist.github.com/spro/ef26915065225df65c1187562eca7ec4

from torch.autograd import Variable as V
from tqdm import tqdm
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import copy

row_idx = 0
last_idx = -1


class Sequence(nn.Module):
    def __init__(self, hidden_num=50):
        super(Sequence, self).__init__()
        self.hidden_num = hidden_num

        self.lstm1 = nn.LSTMCell(1, self.hidden_num)
        self.lstm2 = nn.LSTMCell(self.hidden_num, self.hidden_num)
        self.lstm3 = nn.LSTMCell(self.hidden_num, self.hidden_num)
        self.linear = nn.Linear(self.hidden_num, 1)

    def forward(self, input, pred_len=0):
        h_t = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)
        c_t = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)
        h_t2 = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)
        c_t2 = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)
        # h_t3 = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)
        # c_t3 = th.zeros(input.size(0), self.hidden_num).type(th.DoubleTensor)

        output = None
        outputs = list()
        for _, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm2(h_t, (h_t3, c_t3))
            output = self.linear(h_t2)
        outputs.append(output[-1, :])

        for _ in range(pred_len-1):     # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm2(h_t, (h_t3, c_t3))
            output = self.linear(h_t2)
            # output = self.linear(h_t3)
            outputs.append(output[-1, :])

        outputs = th.cat(outputs)
        return outputs


def loss_fn(y, pred):
    loss = (y - pred)
    return loss**2


if __name__ == '__main__':

    # --- data --- #
    sin = np.sin(np.arange(0, 1000))

    sin = pd.DataFrame(sin)
    data_tmp = copy.deepcopy(sin)
    shifts = list()
    shift_len = 2*7*2
    for idx in range(shift_len):
        col_name = "shift_" + str(idx + 1)
        sin[col_name] = data_tmp.shift(idx+1)
    sin = sin.dropna()

    data = th.from_numpy(sin.values)

    # --- hyper param --- #
    epochs = 1  # 30
    input_len = 20
    output_len = 4
    row_len = input_len + output_len
    batch_size = 20
    col_to_pred = 0

    # --- model --- #
    seq = Sequence(hidden_num=64)
    seq.double()
    optimizer = optim.Adam(seq.parameters(), lr=0.01)

    cnt = 0
    batch_loss = V(th.zeros([1, 1]), requires_grad=True).type(th.float64)
    for epoch in tqdm(range(epochs)):

        n = data.shape[row_idx]-row_len
        shuffle_idxs = np.random.choice(np.arange(0, n, 1), size=n, replace=False).tolist()

        for idx in shuffle_idxs:
            print("*", end="")

            cnt += 1
            data_one_row = copy.deepcopy(data[idx:(idx+input_len+output_len)])
            input_one_row = data_one_row[:input_len]
            y = data_one_row[input_len:, col_to_pred]

            input_one_row, y = V(input_one_row, requires_grad=False), V(y, requires_grad=False)

            if output_len > 1:
                pred_arr = list()
                for idx in range(output_len):
                    pred = seq.forward(input_one_row, pred_len=1)
                    pred_arr.append(pred)
                    new_row = th.cat((pred, input_one_row[last_idx][:last_idx]))
                    input_one_row = th.cat((input_one_row[1:], new_row.view(1, -1)))

                    loss = loss_fn(y[idx], pred)
                    batch_loss += th.mean(loss)

            elif output_len == 1:
                pred = seq.forward(input_one_row, pred_len=output_len)

            if cnt % batch_size == 0:
                optimizer.zero_grad()
                batch_loss = batch_loss/(batch_size * output_len)
                batch_loss.backward(retain_graph=True)
                optimizer.step()

                print("loss:", round(math.sqrt(float(batch_loss)), 2))
                batch_loss = V(th.zeros([1, 1]), requires_grad=True).type(th.float64)

            # for name, parameter in seq.named_parameters():
            #     print("name:", name, " / param:", parameter)

        # --- print loss --- #
        # if epoch % 1 == 0:
            # print("loss:", round(math.sqrt(float(loss)), 2))

pred = th.Tensor(np.array([float(x) for x in pred_arr]))
nans = th.Tensor([np.nan for _ in range(input_len)]).type(th.FloatTensor)
output = th.cat((nans, pred.type(th.FloatTensor)))
plt.plot(data_one_row[:, col_to_pred].numpy(), 'o-', label="real", color="blue")
plt.plot(output.data.numpy(), 'o-', label="pred", color="red")
plt.legend()
plt.show()


# --- predict --- #
pred_length = 300

pred_arr = list()
for idx in range(pred_length):
    pred = seq.forward(input_one_row, pred_len=1)
    new_row = th.cat((pred, input_one_row[last_idx][:last_idx]))
    input_one_row = th.cat((input_one_row[1:], new_row.view(1, -1)))
    pred_arr.append(pred)

pred_arr = [float(x) for x in pred_arr]
plt.plot(pred_arr, 'o-')
plt.show()

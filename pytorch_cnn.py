import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.datasets import mnist
from keras.utils import np_utils
import IPython


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x))

net = Net()
print(net)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:, np.newaxis].astype(float)/255.0
X_test = X_test[:, np.newaxis].astype(float)/255.0
y_train = np_utils.to_categorical(y_train).astype(float)
y_test= np_utils.to_categorical(y_test).astype(float)

criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

batch_size = 60

for batch in range(1, 10000):
	# in your training loop:
	optimizer.zero_grad()   # zero the gradient buffers

    X_batch = [batch*batch_size%60000:(batch*(batch_size + 1)%60000)]
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()    # Does the update


	print('Train Batch: {} \tLoss: {:.6f}'.format(
			batch, loss.data[0]))
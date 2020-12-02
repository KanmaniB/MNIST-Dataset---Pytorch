import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.autograd import Variable

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/Kanmani/Code/kaggle/input/digit-recognizer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("C:/Kanmani/Code/kaggle/input/digit-recognizer/train.csv")
train_data

Y_data = np.array(train_data["label"])
X_data = np.array(train_data.loc[:,"pixel0":]).reshape(-1,1,28,28)/255
print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)

from torch.utils.data import TensorDataset, DataLoader

tensor_x = torch.from_numpy(X_data).float()
tensor_y = torch.from_numpy(Y_data)

my_dataset = TensorDataset(tensor_x,tensor_y)
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, (int(len(X_data)*0.99),int(len(X_data)*0.01)))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),  # in_channels,out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=2),  # in_channels,out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=2),  # don't have to set padding, PyTorch can handle it.
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(256, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()  # training mode
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        del output
        training_loss /= len(train_loader.dataset)

        model.eval()  # evaluation mode for test.
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        del output
        valid_loss /= len(val_loader.dataset)
        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))

from torch import optim

cnn = CNN()
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")
cnn.to(device)

optimizer = optim.Adam(cnn.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
train(cnn, optimizer, loss_fn, train_loader=train_loader, val_loader=val_loader, device=device)

# Make predictions

test_data = pd.read_csv("C:/Kanmani/Code/kaggle/input/digit-recognizer/test.csv")
test_data

X_test = np.array(test_data).reshape(-1,1,28,28)/255
print("X_test shape:", X_test.shape)

tensor_test = torch.from_numpy(X_test).float()
tensor_test = tensor_test.to(device)

test_loader = DataLoader(tensor_test, batch_size = 64)

def prediction(model, data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data = Variable(data)
            if torch.cuda.is_available():
                data = data.cuda()
            output = model(data)
            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)
    return test_pred

test_pred = prediction(cnn, test_loader)

out_df = pd.DataFrame(np.c_[np.arange(1, len(test_data)+1)[:,None], test_pred.numpy()],
                      columns=['ImageId', 'Label'])
out_df.head()
out_df.to_csv('C:/Kanmani/Code/kaggle/output/digit-recognizer/sample_submission.csv', index=False)
# This code is used in the 'Run MATLAB from Python' example

# Copyright 2019-2021 The MathWorks, Inc.

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.onnx

import time
import os

cudaAvailable = torch.cuda.is_available()
if cudaAvailable:
    cuda = torch.device('cuda')

# start a MATLAB engine
import matlab.engine
MLEngine = matlab.engine.start_matlab()

miniBatchSize = 128.0

# Prepare training dataset
class TrainData(Dataset):
    def __init__(self):
        # Create persistent training dataset in MATLAB
        MLEngine.setupDatasets(miniBatchSize)
        # Set the dataset length to the number of minibatches
        # in the training dataset
        self.len = int(MLEngine.getNumIterationsPerEpoch())

    def __getitem__(self, index):
        # Call MATLAB to get a minibatch of features + labels
        minibatch = MLEngine.extractTrainingFeatures()
        x = torch.FloatTensor(minibatch.get('features'))
        y = torch.FloatTensor(minibatch.get('labels'))
        return x, y

    def __len__(self):
        return int(self.len)

print('Setting up datastores...')
trainDataset = TrainData()
print('Datastore setup complete')
print('Minibatch size: ', int(miniBatchSize))
print('Number of training files: ', int(trainDataset.len * miniBatchSize))
print('Number of minibatches per epoch: ', int(trainDataset.len))

trainLoader = DataLoader(dataset=trainDataset, batch_size=1)

print('Computing validation features...')
# Prepare validation dataset
# Call MATLAB to compute validation features
valFeatures = MLEngine.extractValidationFeatures()
XValidation = valFeatures["features"]
YValidation = valFeatures["labels"]

# Create Data Class
class ValData(Dataset):
    # Constructor
    def __init__(self):
        self.x = XValidation
        self.y = YValidation
        self.len = self.y.size[0]

    # Getter
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = torch.FloatTensor(self.y[index])
        return x, y

    # Get Length
    def __len__(self):
        return self.len

valDataset = ValData()
valLoader = DataLoader(dataset = valDataset, batch_size = valDataset.len)
print('Validation feature computation complete')

# Create the neural network
NumF = 12
numHops = 98
timePoolSize = 13
dropoutProb = 0.2
numClasses = 11

class CNN(nn.Module):

    # Contructor
    def __init__(self, out_1=NumF):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=2*out_1, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(2*out_1)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn3 = nn.Conv2d(in_channels=2*out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(4 * out_1)
        self.relu3 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn4 = nn.Conv2d(in_channels=4 * out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(4 * out_1)
        self.relu4 = nn.ReLU()
        self.cnn5 = nn.Conv2d(in_channels=4 * out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(4 * out_1)
        self.relu5 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(kernel_size=(timePoolSize, 1))

        self.dropout = nn.Dropout2d(dropoutProb)

        self.fc = nn.Linear(336, numClasses)

        #self.softmax = nn.Softmax(dim = 1)

    # Prediction
    def forward(self, x):
        out = self.cnn1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.batch3(out)
        out = self.relu3(out)

        out = self.maxpool3(out)

        out = self.cnn4(out)
        out = self.batch4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.batch5(out)
        out = self.relu5(out)

        out = self.maxpool4(out)

        out = self.dropout(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = self.softmax(out)

        return out

model = CNN()
if cudaAvailable:
    model.cuda()

# Define training parameters
n_epochs = 25
criterion = nn.CrossEntropyLoss()
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_list = []
accuracy_list = []
numValItems = len(valDataset)

doValidation = True

print('Training...')

for epoch in range(n_epochs):

    if epoch == 20:
        for g in optimizer.param_groups:
            g['lr'] = 3e-5

    count = 0
    for batch in trainLoader:
        count += 1
        print('Epoch ', epoch+1, ' Iteration', count, ' of ', trainDataset.len)
        if cudaAvailable:
            x = batch[0].cuda()
            y = batch[1].cuda()
        else:
            x = batch[0]
            y = batch[1]
        optimizer.zero_grad()
        z = model(torch.squeeze(x.float(), 0))
        loss = criterion(z, torch.squeeze(y).long())
        loss.backward()
        optimizer.step()

    if doValidation:
        correct = 0
        # perform a prediction on the validation  data
        for x_test, y_test in valLoader:
            if cudaAvailable:
                x_test = x_test.cuda()
                y_test = y_test.cuda()
            else:
                x_test = x_test
                y_test = y_test
            z = model(x_test.float())
            _ , yhat = torch.max(z.data, 1)
            correct += (yhat == y_test.squeeze()).sum().item()
        accuracy = correct / numValItems
        print('Validation accuracy: ', accuracy)
        accuracy_list.append(accuracy)
        loss_list.append(loss.data)

        # Export the trained model to ONXX format
        if cudaAvailable:
            x = torch.empty(1, 1, 98, 50).cuda()
        else:
            x = torch.empty(1, 1, 98, 50)

        torch.onnx.export(model,
                  x,
                  "cmdRecognition.onnx",
                  export_params=True,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'])

print('Training complete')
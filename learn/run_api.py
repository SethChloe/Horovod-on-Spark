import os
import sys

import horovod.spark
import horovod.torch as hvd
import numpy as np
import pandas as pd
import pyspark
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyspark.sql import SparkSession
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

os.environ['HOROVOD_GLOO_TIMEOUT_SECONDS'] = '3000'
# constants
NUM_PROC = 1
BATCH_SIZE = 16
EPOCHS = 10

# initialize spark
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/bin/python'
spark = SparkSession.builder \
    .config("spark.executor.memory", "6g").config("spark.driver.memory", "6g").getOrCreate()


# MODEL
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(model, optimizer, criterion, epoch, train_loader, val_loader):
    model.train()
    train_losses = []
    train_loader = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_loader):
        images, labels = data
        pred = model(images)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        # optimizer.synchronize()
        # with optimizer.skip_synchronize():
        optimizer.step()
            # scheduler.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        validation_losses = []
        val_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(val_loader):
            images, labels = data
            pred = model(images)
            loss = criterion(pred, labels)
            validation_losses.append(loss.item())
    training_loss = np.mean(train_losses)
    validation_loss = np.mean(validation_losses)
    # Horovod: average metric values across workers.
    validation_loss = metric_average(validation_loss, 'avg_validation_loss')
    training_loss = metric_average(training_loss, 'avg_training_loss')
    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print(
            f"hvd.rank() = {hvd.rank()}, epoch = {epoch},training loss = {training_loss}, validation loss = {validation_loss}")


def train_hvd():
    is_cuda = torch.cuda.is_available()
    number_of_gpus = torch.cuda.device_count()
    # Horovod: initialize library.
    hvd.init()
    if is_cuda:
        # Horovod: pin GPU to local rank.
        if number_of_gpus == 1:
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(42)
    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(1)

    # By default, Adasum doesn't need scaling up learning rate.
    # lr_scaler = hvd.size()

    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-3 * lr_scaler)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, eta_min=0, T_max=20 * lr_scaler)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # hvd.broadcast_object(scheduler, root_rank=0)
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average)

    dir = './data'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dir, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=False)
    for epoch in range(EPOCHS):
        train(model, optimizer, criterion, epoch, train_loader, val_loader)


if __name__ == "__main__":
    horovod.spark.run(train_hvd, verbose=1, num_proc=NUM_PROC)

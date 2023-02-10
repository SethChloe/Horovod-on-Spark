import os
import sys

import horovod.spark
import horovod.torch as hvd
import numpy as np
import pandas as pd
import petastorm
import pyspark
import torch
import torch.nn as nn
import torch.nn.functional as F
from petastorm.spark import make_spark_converter, SparkDatasetConverter
from pyspark.sql import SparkSession


os.environ['HOROVOD_GLOO_TIMEOUT_SECONDS'] = '3000'
# constants
NUM_PROC = 1
BATCH_SIZE = 20
EPOCHS = 10

# initialize spark
os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/bin/python'
spark = (SparkSession.builder
         .appName("app_name")
         .config("spark.driver.memory", "4g")
         .config("spark.driver.maxResultSize", "4g")
         .getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
               "file:///tmp/petastorm/cache")
# create petastorm dataset
iris_pdf = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
iris_pdf = iris_pdf.assign(
    **{'species': iris_pdf['species'].astype("category").cat.codes.astype(float)})
iris_pdf = pd.concat([iris_pdf] * 10)
iris_pdf = iris_pdf.sample(frac=1, replace=False).reset_index(drop=True)
iris_sdf = spark.createDataFrame(iris_pdf)
train_sdf, test_sdf = iris_sdf.randomSplit([0.5, 0.5], seed=23)
converter_train = make_spark_converter(train_sdf.repartition(NUM_PROC))
converter_test = make_spark_converter(test_sdf.repartition(NUM_PROC))


# MODEL

class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X


model_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(model, optimizer, scheduler, criterion, epoch, train_steps_per_epoch, train_dataloader_iter, test_steps_per_epoch, test_dataloader_iter):
    model.zero_grad()
    model.train()
    train_losses = []
    for train_step in range(train_steps_per_epoch):
        batch = next(train_dataloader_iter)
        optimizer.zero_grad()
        X = torch.stack([v for k, v in batch.items() if k in model_features]).T
        y = batch[target].long()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.synchronize()
        with optimizer.skip_synchronize():
            optimizer.step()
            scheduler.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        validation_losses = []
        for test_step in range(test_steps_per_epoch):
            batch = next(test_dataloader_iter)
            X = torch.stack([v for k, v in batch.items()
                             if k in model_features]).T
            y = batch[target].long()
            out = model(X)
            loss = criterion(out, y)
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
    torch.set_num_threads(1)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size()

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.e-3 * lr_scaler)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=0, T_max=20 * lr_scaler)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_object(scheduler, root_rank=0)
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average)
    with converter_train.make_torch_dataloader(batch_size=BATCH_SIZE, cur_shard=hvd.rank(), shard_count=hvd.size()) as train_dataloader, \
            converter_test.make_torch_dataloader(batch_size=BATCH_SIZE, cur_shard=hvd.rank(), shard_count=hvd.size()) as test_dataloader:
        train_dataloader_iter = iter(train_dataloader)
        train_steps_per_epoch = len(converter_train) // BATCH_SIZE

        test_dataloader_iter = iter(test_dataloader)
        test_steps_per_epoch = max(1, len(converter_test) // BATCH_SIZE)
        for epoch in range(EPOCHS):
            train(model, optimizer, scheduler, criterion, epoch, train_steps_per_epoch,
                  train_dataloader_iter, test_steps_per_epoch, test_dataloader_iter)


if __name__ == "__main__":
    horovod.spark.run(train_hvd, verbose=2, num_proc=NUM_PROC)
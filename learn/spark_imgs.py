import argparse
import os
import subprocess
import sys

from filelock import FileLock
from packaging import version

import numpy as np

import pyspark
import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from torchvision import transforms, datasets

if version.parse(pyspark.__version__) < version.parse('3.0.0'):
    from pyspark.ml.feature import OneHotEncoderEstimator as OneHotEncoder
else:
    from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import lit
import pyspark.sql.functions as F
from pyspark.ml.image import ImageSchema
from pyspark.ml.linalg import DenseVector, VectorUDT

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import horovod.spark.torch as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store
from horovod.spark.common import util

parser = argparse.ArgumentParser(description='PyTorch Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark master to connect to')
parser.add_argument('--num-proc', type=int, default=1,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--work-dir', default='/tmp/horovod/work',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp/horovod/data',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')
parser.add_argument('--backward-passes-per-step', type=int, default=1,
                    help='number of backward passes to perform before calling hvd.allreduce')

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
    os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/bin/python'

    # Initialize SparkSession
    conf = SparkConf().setAppName('pytorch_spark_mnist') \
        # .set('spark.sql.shuffle.partitions', '16')
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf)\
        .config("spark.executor.memory", "6g").config("spark.driver.memory", "6g").getOrCreate()

    # Setup our store for intermediate data
    store = Store.create(args.work_dir)

    # # Download MNIST dataset
    # data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
    # libsvm_path = os.path.join(args.data_dir, 'mnist.bz2')
    # if not os.path.exists(libsvm_path):
    #     subprocess.check_output(['wget', data_url, '-O', libsvm_path])
    #
    # Load dataset into a Spark DataFrame
    img_path = '/root/autodl-tmp/fer2013/train/'
    img_anger = spark.read.format('image').load(img_path + 'anger') \
        .withColumn("label", lit(0.0)).repartition(1)
    img_fear = spark.read.format('image').load(img_path + 'fear') \
        .withColumn("label", lit(1.0)).repartition(1)
    img_contempt = spark.read.format('image').load(img_path + 'contempt') \
        .withColumn("label", lit(2.0)).repartition(1)
    img_disgust = spark.read.format('image').load(img_path + 'disgust') \
        .withColumn("label", lit(3.0)).repartition(1)
    img_happiness = spark.read.format('image').load(img_path + 'happiness') \
        .withColumn("label", lit(4.0)).repartition(1)
    img_neutral = spark.read.format('image').load(img_path + 'neutral') \
        .withColumn("label", lit(5.0)).repartition(1)
    img_sadness = spark.read.format('image').load(img_path + 'sadness') \
        .withColumn("label", lit(6.0)).repartition(1)
    img_surprise = spark.read.format('image').load(img_path + 'surprise') \
        .withColumn("label", lit(7.0)).repartition(1)
    train_df = img_anger.unionAll(img_contempt).unionAll(img_disgust).unionAll(img_happiness) \
        .unionAll(img_fear).unionAll(img_neutral).unionAll(img_sadness).unionAll(img_surprise)

    ImageSchema.imageFields
    img2vec = F.udf(lambda x: DenseVector(ImageSchema.toNDArray(x).flatten()), VectorUDT())
    train_df = train_df.withColumn('features', img2vec("image"))

    # Define the PyTorch model without any Horovod-specific parameters
    class Deep_Emotion(nn.Module):
        def __init__(self):
            super(Deep_Emotion, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 0),  # 32*48*48
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32*23*23
                nn.Conv2d(32, 64, 3, 1, 0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64*10*10
                nn.Conv2d(64, 64, 3, 1, 0),

                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 8)
            )

        def forward(self, x):
            x = self.model(x)
            return x


    model = Deep_Emotion()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss = nn.CrossEntropyLoss()

    # Train a Horovod Spark Estimator on the DataFrame
    backend = SparkBackend(num_proc=args.num_proc,
                           stdout=sys.stdout, stderr=sys.stderr,
                           prefix_output_with_timestamp=True)

    torch_estimator = hvd.TorchEstimator(backend=backend,
                                         store=store,
                                         model=model,
                                         optimizer=optimizer,
                                         loss=lambda input, target: loss(input, target.long()),
                                         input_shapes=[[-1, 3, 48, 48]],
                                         feature_cols=['features'],
                                         label_cols=['label'],
                                         batch_size=args.batch_size,
                                         epochs=args.epochs,
                                         #validation=0.1,
                                         # backward_passes_per_step=args.backward_passes_per_step,
                                         verbose=1)

    torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])

    # Evaluate the model on the held-out test DataFrame
    # pred_df = torch_model.transform(test_df)
    #
    # argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    # pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
    # evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
    # print('Test accuracy:', evaluator.evaluate(pred_df))

    spark.stop()

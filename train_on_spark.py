import os
import time
import datetime
import os
import shutil
import sys

from pyspark.sql import SparkSession

import torch.multiprocessing as mp
import torch
from torchvision import transforms
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from model import Deeplab_Torch_Multisource

import horovod.torch as hvd
import horovod.spark

from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric


def main(args):
    def train_one_epoch(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader),
                  desc='Train Epoch     #{}'.format(epoch + 1),
                  disable=not verbose) as t:
            for batch_idx, (images, targets, imgs_feats) in enumerate(train_loader):
                images = images.cuda()
                imgs_feats = imgs_feats.cuda()
                targets = targets.cuda().long()

                outputs = model(images, lbr=imgs_feats)

                loss_dict = criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                lr_scheduler.step()
                t.set_postfix({'lr': optimizer.param_groups[0]['lr'],
                               'loss': losses_reduced.item()})
                t.update(1)


    def validation(epoch):
        metric.reset()
        torch.cuda.empty_cache()
        model.eval()
        with tqdm(total=len(val_loader),
                  desc='Validate Epoch  #{}'.format(epoch + 1),
                  disable=not verbose) as t:
            for batch_idx, (images, targets, imgs_feats) in enumerate(val_loader):
                images = images.cuda()
                imgs_feats = imgs_feats.cuda()
                targets = targets.cuda().long()
                with torch.no_grad():
                    outputs = model(images, lbr=imgs_feats)
                metric.update(outputs[0], targets)
                t.update(1)
        pixAcc, mIoU = metric.get()
        logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(batch_idx + 1, pixAcc, mIoU))
        global order
        global best_pred
        metric.confusion_plot(id=order)
        order += 1
        # TODO
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > best_pred:
            best_pred = new_pred
            save_checkpoint(model, args)
        synchronize()
        print("***********************************")

    def save_checkpoint(model, args):
        """Save Checkpoint"""
        directory = os.path.expanduser(args.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = 'AMSDF_model.pth'
        filename = os.path.join(directory, filename)
        torch.save(model, filename)

    def create_lr_scheduler(optimizer,
                            num_step: int,
                            epochs: int,
                            warmup=True,
                            warmup_epochs=1,
                            warmup_factor=1e-3):
        assert num_step > 0 and epochs > 0
        if warmup is False:
            warmup_epochs = 0

        def f(x):
            """
            根据step数返回一个学习率倍率因子，
            注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
            """
            if warmup is True and x <= (warmup_epochs * num_step):
                alpha = float(x) / (warmup_epochs * num_step)
                # warmup过程中lr倍率因子从warmup_factor -> 1
                return warmup_factor * (1 - alpha) + alpha
            else:
                # warmup后lr倍率因子从1 -> 0
                # 参考deeplab_v2: Learning rate policy
                return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

    # cudnn.benchmark = True #TODO benchmark

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    logger = setup_logger("semantic_segmentation", args.log_dir, hvd.rank(),
                          filename='{}_{}_log.txt'.format('fcn', 'wlkdata'))
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    args.base_size = 224
    args.crop_size = args.base_size
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(8)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # 加载数据集
    train_dataset = get_segmentation_dataset(args.data_set, root=args.data_dir, split='train', mode='train',
                                             **data_kwargs)

    val_dataset = get_segmentation_dataset(args.data_set, root=args.data_dir, split='val', mode='val',
                                           **data_kwargs)
    # # Horovod: use DistributedSampler to partition the training data.
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.allreduce_batch_size,
    #     sampler=train_sampler, **kwargs)
    # # Horovod: use DistributedSampler to partition the test data.
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    #                                          sampler=val_sampler, **kwargs)

    iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
    max_iters = args.epochs * iters_per_epoch
    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, max_iters)
    val_sampler = make_data_sampler(val_dataset, False, args.distributed)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=8,
                                   pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_sampler=val_batch_sampler,
                                 num_workers=8,
                                 pin_memory=True)


    model = Deeplab_Torch_Multisource(13)
    # Move model to GPU.
    model.cuda()

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         name, ext = os.path.splitext(args.resume)
    #         # assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'{AttributeError}'Deeplab_Torch_Multisource' object has no attribute 'paramters'
    #         print('Resuming training, loading {}...'.format(args.resume))
    #         model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

    # create criterion
    criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                      aux_weight=args.aux_weight, ignore_index=-1).cuda()

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr * lr_scaler,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    # lr scheduling
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    metric = SegmentationMetric(train_dataset.num_class)

    # 开始训练
    iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
    max_iters = args.epochs * iters_per_epoch
    start_time = time.time()
    global order
    global best_pred
    order=0
    best_pred = 0.0
    logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(args.epochs, max_iters))

    for epoch in range(args.epochs):
        train_one_epoch(epoch)
        validation(epoch)

    total_training_time = time.time() - start_time
    total_training_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f}s / it)".format(
            total_training_str, total_training_time / max_iters))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")
    parser.add_argument('--model', type=str, default='fcn')
    parser.add_argument("--data-dir", default="/root/autodl-tmp/WLKdata_1111/WLKdataset", help="WLKdata root")
    parser.add_argument("--data-set", default="yjs")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--log-dir', default='./runs/logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--save-dir', default='./models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--use-ohem', type=bool, default=True,  # 提出的模型跑的时候要设置这个
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    # TODO use mixed precision for training
    parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                        help='use mixed precision for training')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')

    # Arguments when not run through horovodrun
    parser.add_argument("--distributed", default=True)
    parser.add_argument("--num-gpus", default=1)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
    parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    # initialize spark
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/jdk1.8.0_162'
    os.environ['PYSPARK_PYTHON'] = '/root/miniconda3/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/miniconda3/bin/python'
    spark = SparkSession.builder \
        .config("spark.executor.memory", "6g").config("spark.driver.memory", "6g").getOrCreate()

    if args.num_proc:
        # run training through horovod.run
        print('Running training through horovod.spark.run')
        horovod.spark.run(main,
                          args=(args,),
                          num_proc=args.num_proc,
                          use_gloo=args.communication == 'gloo',
                          use_mpi=args.communication == 'mpi')

import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from segment_anything.build_Litho import build_litho, build_light_litho, build_source_litho, build_litho_one
from util import  train_one_epoch, evaluate

from dataset import *

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler



def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tb_writer = SummaryWriter("runs")

    # 实例化训练数据集

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_dataset.collate_fn)

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_dataset.collate_fn)

    Benchmark = "organizedData"
    ImageSize = (1024, 1024)
    BatchSize = args.batch_size
    NJobs = 8

    
    # train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    train_dataset, val_dataset = lithosim(Benchmark, ImageSize)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    model = build_litho()

    
    parameter = model.parameters()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    # model_weight_path = "./saved/model-4.pth"
    
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))

    optimizer = optim.AdamW(parameter, lr=args.lr, weight_decay=0.01, betas = [0.9, 0.999])
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x 
    * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    for epoch in range(args.epochs):
        # train
        train_loss =  train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        # val_loss  = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--gpu_id', type=str, default='0,1')

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

# torchrun  --nproc_per_node=2 new_train.py --batchSize 64 --epochs 10



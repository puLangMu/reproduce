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

from datetime import datetime
import torch.multiprocessing as mp
import torchvision
import torch.nn as nn
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp


def train(gpu, args):

    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )  

    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    


    tb_writer = SummaryWriter("runs")
    model = build_litho()
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])





    Benchmark = "organizedData"
    ImageSize = (1024, 1024)
    BatchSize = args.batch_size
    NJobs = 8



    # train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)
    train_dataset, val_dataset = lithosim(Benchmark, ImageSize)


    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=BatchSize,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=0,
       pin_memory=True,
    #############################
      sampler=train_sampler)    # 
    #############################

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
       batch_size=BatchSize,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=0,
       pin_memory=True,
    #############################
      sampler=train_sampler)   




    parameter = model.parameters()

    # model_weight_path = "./saved/model-4.pth"
    
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))

    optimizer = optim.AdamW(parameter, lr=args.lr, weight_decay=0.01, betas = [0.9, 0.999])
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x 
    * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)



    for epoch in range(args.epochs):
        # train
        train_sampler.set_epoch(epoch)
        train_loss =  train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss  = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')


    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,)) 

    train(args)

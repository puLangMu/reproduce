import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from segment_anything.build_Litho import build_litho, build_light_litho
from utils import  train_one_epoch, evaluate

from dataset import *


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


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

    train_loader, val_loader = loadersLitho(Benchmark, ImageSize, BatchSize, NJobs)


    model = build_litho()

    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除不需要的权重
    #     del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #         else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))

    # if args.freeze_layers: 
    #     for name, para in model.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]

    parameter = model.parameters()
    optimizer = optim.AdamW(parameter, lr=args.lr, weight_decay=0.01, betas = [0.9, 0.999])
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)


    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

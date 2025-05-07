import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt


from utils import calculate_loss
from utils import show_images




def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list



def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    # accu_loss = torch.zeros(1).to(device)  # 累计损失
    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    k = 0.7
    alpha = 2
    beta = 1
    gamma = 3


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        # images, labels = data
        # sample_num += images.shape[0]

        # pred = model(images.to(device))
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # loss = loss_function(pred, labels.to(device))
        # loss.backward()
        # accu_loss += loss.detach()

        mask, source, resist = data
        source = torch.nn.functional.interpolate(source, size=(256,256), mode='bilinear', align_corners=False)

        source.requires_grad = True
        mask.requires_grad = True
        resist.requires_grad = True

        source1 = source.to(device)
        mask1 = mask.to(device)
        pred = model(mask1, source1)

       

    
        loss, BCE_loss, dice_loss, ssim_loss = calculate_loss(pred, resist.to(device), alpha = alpha, beta = beta, gamma = gamma, k= k)

        # with torch.no_grad():
        #     single_resist =  resist[:1,:,:,:].to(device) # 第一张 resist 图像
        #     single_mask = mask[:1,:,:,:].to(device) # 第一张 mask 图像
        #     single_source = source[:1,:,:,:].to(device) # 第一张 source 图像
        #     test_pred = pred[:1,:,:,:] # 第一张预测图像

        #     print("Loss:", loss.item())
        #     print("BCE Loss:", BCE_loss.item())
        #     print("Dice Loss:", dice_loss.item())
        #     print("SSIM Loss:", ssim_loss.item())

            # if(loss < 2.5):
            #       show_images(test_pred, single_mask, single_source, single_resist, save_dir="pictures", name = "loss_{:.3f}".format(loss.item()))

                
        
        loss.backward()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, loss.item() )
                                                                               

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        
        # for name, parms in reversed(list(model.named_parameters())):
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        
        # torch.nn.utils.clip_grad_value_(mask, clip_value=0.5)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        
        # if step % 3 == 1:
        #     for name, parms in reversed(list(model.named_parameters())):
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

        #     optimizer.step()
            
        #     optimizer.zero_grad()

        optimizer.step()
            
        optimizer.zero_grad()

    return loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    model.eval()


    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
    #     images, labels = data
    #     sample_num += images.shape[0]

    #     pred = model(images.to(device))
    #     pred_classes = torch.max(pred, dim=1)[1]
    #     accu_num += torch.eq(pred_classes, labels.to(device)).sum()

    #     loss = loss_function(pred, labels.to(device))
    #     accu_loss += loss

    #     data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
    #                                                                            accu_loss.item() / (step + 1),
    #                                                                            accu_num.item() / sample_num)

    # return accu_loss.item() / (step + 1), accu_num.item() / sample_num

        mask, source, resist = data
        source = torch.nn.functional.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)

        pred = model(mask.to(device), source.to(device))

        loss, _, _, _ = calculate_loss(pred, resist.to(device), alpha=2, beta=1, gamma=3, k=0.9)


        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch, loss.item() )
                                                                               


    return loss.item()



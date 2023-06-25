import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
from torch.optim import Adam
from Unet_LSTM import Unet_Lstm, SeqModel
from DataLoader import DataContainer
import matplotlib.pyplot as plt
from utils import LoadModel
from ConvLstm_v3 import Seq2Seq

import time
if  __name__ == "__main__":
    time_start = time.time()
    data_root = "./Data/sample_data"
    file_names = {'latitude':'latitude_matrix.txt', 'longitude':'longitude_matrix.txt',
                  'ssc_data':'ssc_data_176.9_177.25_-39.5_-39.75.txt',
                  'day':'day.txt', 'month':'month.txt', 'year':'year.txt'}
    sample_indices = {'train':(0, 232), 'val':(232, 304), 'test':(304, 380)}
    layer_num = 5
    num_epochs = 100
    lstm_layer_num = 1
    batch_size = 16
    loaderoffset = 0.5

    reuse = True
    #best_3lstm_100_epochs.pth   best_5lstm_100_epochs.pth
    checkPath = "CheckPoints/unet/best.pth"

    data_container = DataContainer(data_root=data_root, file_names=file_names, sample_indices=sample_indices,
                                   batch_size=batch_size)
    _, _, test_loader = data_container.getLoader()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Unet_Lstm(layer_num=layer_num, input_dim=1, base_dim=32,
                      lstm_mode='all', lstm_layer_num=lstm_layer_num).to(device)
    # model = SeqModel(layer_num=1, input_dim=1, base_dim=32, lstm_mode='all').to(device)
    #model = Seq2Seq(num_channels=1, num_kernels=64,
     #                               kernel_size=(3, 3), padding=(1, 1), activation="relu",
      #                                                  frame_size=(96, 135), num_layers=layer_num, device=device).to(device)
    if reuse:
        _, _, modelDict, _ = LoadModel(checkPath)
        model.load_state_dict(modelDict)
    print(sum(p.numel() for p in model.parameters() if  p.requires_grad))

    criterion = nn.L1Loss(reduction='none')

    model.eval()
    for batch_num, (inputs, target) in enumerate(test_loader):
        if (batch_num == 0):
            continue
        print(inputs.size())
        # inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        inputs = torch.nn.functional.pad(inputs, pad=(12, 13))
        inputs = inputs.to(device)
        target = target.to(device)
        output = model(inputs)
        output = output[:, :, :, 12:-13]
        # output = output.clamp(min=-1., max=1.)

        inputs = inputs[:, :, :, :, 12:-13]
        inputs = (inputs + 1.) / 2.
        inputs = inputs.cpu()
        output = (output + 1.) / 2.
        output = output.cpu()
        target = (target + 1.) / 2.
        target = target.cpu()
        break
    sLong, eLong, sLati, eLati = data_container.getLocation()
    fig, axs = plt.subplots(9, 9, figsize=(30, 90), constrained_layout=True)
    from PIL import Image
    import numpy as np
    outs = inputs.detach().numpy() * 255.
    outs = outs.astype(np.uint8)
    num = 0
    # for i in range(len(output)):
    #     for j in range(7):
    #         num += 1
    #         img = Image.fromarray(outs[i][0][j])
    #         img.save('%s.jpg' %(num))
    for i in range(len(output)):
        if i == 0:
            for j in range(7):
                axs[0, j].set_title('img_%s' % (j))
            axs[0, 7].set_title('ground-truth')
            axs[0, 8].set_title('prediction')
        for j in range(7):
            axs[i, j].imshow(inputs[i][0][j].detach().numpy(), cmap='jet', extent=[sLong, eLong, eLati, sLati])
        axs[i, 7].imshow(target[i][0].detach().numpy(), cmap='jet', extent=[sLong, eLong, eLati, sLati])
        axs[i, 8].imshow(output[i][0].detach().numpy(), cmap='jet', extent=[sLong, eLong, eLati, sLati])

    for ax in axs.flat:
        ax.set(xlabel='Longitude', ylabel='Latitude')
        # plt.figure(figsize=(20,10))
        # .xlabel("Longitude")
        # plt.ylabel("Latitude")
        # plt.imshow(output[i][0].detach().numpy(),cmap='jet',extent = [sLong,eLong,eLati,sLati])
        # cbar = plt.colorbar()
        # cbar.set_label("mg/l")
        
    plt.savefig('./training_log_Unet_'+ str(num_epochs) + '_' + str(layer_num) + '_' + str(loaderoffset) + '.jpg')
    plt.show()
    print("total test time:" , time.time()-time_start)
    with open('training_log_Unet_'+ str(num_epochs) + '_' + str(layer_num) + '_' + str(loaderoffset) + '.txt', 'a') as f:
        f.write("total test time:" + str(time.time()-time_start) + "\n")

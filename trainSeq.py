import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from torch.optim import Adam

from ConvLstm_v3 import Seq2Seq
from DataLoader import DataContainer
from utils import SaveModel, LoadModel

import time
# dataset = SscDataset('./Data/sample_data/ssc_data_176.9_177.25_-39.5_-39.75.txt')
# ssc,day,month,year=dataset[0]
# ssc = ssc.reshape(eLatiIndex-sLatiIndex,eLongIndex-sLongIndex)
# plt.figure(figsize=(20,10))
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.imshow(ssc,cmap='jet',extent = [sLong,eLong,eLati,sLati])
# cbar = plt.colorbar()
# cbar.set_label("mg/l")
# plt.show()

if  __name__ == "__main__":
    time_start = time.time()

    data_root = "./Data/sample_data"
    file_names = {'latitude':'latitude_matrix.txt', 'longitude':'longitude_matrix.txt',
                  'ssc_data':'ssc_data_176.9_177.25_-39.5_-39.75.txt',
                  'day':'day.txt', 'month':'month.txt', 'year':'year.txt'}
    sample_indices = {'train':(0, 232), 'val':(232, 304), 'test':(304, 380)}
    num_epochs = 100
    start_epoch = 0
    batch_size = 16
    lr = 1e-4
    num_layers = 5 #3
    best_performance = float('inf')
    save_root = "CheckPoints/seq"
    pixel_loss = True # True: pixel or False: image
    loaderoffset = 0.5


    reuse = False
    reuseType = 'latest'  # 'latest' or 'best'
    checkPath = os.path.join('CheckPoints', '%s.pth' % (reuseType))

    data_container = DataContainer(data_root=data_root, file_names=file_names, batch_size=batch_size,
                                   sample_indices=sample_indices)
    train_loader, val_loader, test_loader = data_container.getLoader()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(num_channels=1, num_kernels=64,
                    kernel_size=(3, 3), padding=(1, 1), activation="relu",
                    frame_size=(96, 135), num_layers=num_layers, device=device).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    if reuse:
        pre_epoch, pre_performance, modelDict, optDict = LoadModel(checkPath)
        start_epoch = pre_epoch + 1
        best_performance = pre_performance
        model.load_state_dict(modelDict)
        optimizer.load_state_dict(optDict)
        for para in optimizer.param_groups:
            para['lr'] = lr

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.L1Loss(reduction='none')

    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        model.train()
        for batch_num, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            target = torch.cat([inputs[:, :, 1:], target.unsqueeze(dim=2)], dim=2)

            output = output.squeeze(dim=1)
            target = target.squeeze(dim=1)
            B, T, H, W = target.size()
            loss = criterion(output, target)
            loss = loss.mean() if pixel_loss else (loss.sum() / (B * T))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += (loss.detach().item() * B)
        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, target in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                output = output[:, :, -1]
                output = output.squeeze(dim=1)
                target = target.squeeze(dim=1)
                B, H, W = target.size()
                loss = criterion(output, target)
                loss = loss.mean() if pixel_loss else (loss.sum() / B)
                val_loss += (loss.detach().item() * B)
        val_loss /= len(val_loader.dataset)

        SaveModel(epoch, val_loss, model, optimizer, save_root, best=False)

        if val_loss <= best_performance:
            SaveModel(epoch, val_loss, model, optimizer, save_root, best=True)
            best_performance = val_loss

        print("Epoch:{} Training Loss:{} Validation Loss:{}; Best Perfomance:{}"
              .format(epoch, train_loss, val_loss, best_performance))
        with open('training_log_Seq_'+ str(num_epochs) + '_' + str(num_layers) + '_' + str(loaderoffset) + '.txt', 'a') as f:
            f.write('epoch: %s; train_loss: %s, val_loss: %s, best_performance: %s\n'
                    % (epoch, train_loss, val_loss, best_performance))

    with open('training_log_Seq_'+ str(num_epochs) + '_' + str(num_layers) + '_' + str(loaderoffset) + '.txt', 'a') as f:
        f.write("total train time:" + str(time.time()-time_start) + "\n")
    print("total train time:" , time.time()-time_start)




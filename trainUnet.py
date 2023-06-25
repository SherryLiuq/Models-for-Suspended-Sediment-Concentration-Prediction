import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
from torch.optim import Adam

from Unet_LSTM import Unet_Lstm, SeqModel
from DataLoader import DataContainer
from utils import SaveModel, LoadModel

from ConvLstm_v3 import Seq2Seq
import time
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
    layer_num = 5
    lstm_layer_num = 1
    best_performance = float('inf')
    save_root = "CheckPoints/unet"
    loaderoffset = 0.5


    reuse = False
    reuseType = 'latest'  # 'latest' or 'best'
    checkPath = os.path.join('CheckPoints', '%s.pth' % (reuseType))

    data_container = DataContainer(data_root=data_root, file_names=file_names, batch_size=batch_size,
                                   sample_indices=sample_indices)
    train_loader, val_loader, test_loader = data_container.getLoader()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Unet_Lstm(layer_num=layer_num, input_dim=1, base_dim=32,
                      lstm_mode='all', lstm_layer_num=lstm_layer_num).to(device)
   # model = SeqModel(layer_num=3, input_dim=1, base_dim=32, lstm_mode='all').to(device)
    #model = Seq2Seq(num_channels=1, num_kernels=64,
     #               kernel_size=(3, 3), padding=(1, 1), activation="relu",
      #              frame_size=(96, 135), num_layers=layer_num, device=device).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    if reuse:
        pre_epoch, pre_performance, modelDict, optDict = LoadModel(checkPath)
        start_epoch = pre_epoch + 1
        best_performance = pre_performance
        model.load_state_dict(modelDict)
        optimizer.load_state_dict(optDict)
        for para in optimizer.param_groups:
            para['lr'] = lr

    # criterion = nn.L1Loss(reduction='mean')
    criterion = nn.MSELoss(reduction='mean')

    for epoch in range(start_epoch, num_epochs):
        train_loss = 0
        model.train()
        for batch_num, (inputs, target) in enumerate(train_loader):
            B, _, _, _, _ = inputs.size()
            # inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
            inputs = torch.nn.functional.pad(inputs, pad=(12,13))

            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            output = output[:, :, :, 12:-13]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += (loss.detach().item() * B)
        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, target in val_loader:
                B, _, _, _, _ = inputs.size()
                # inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
                inputs = torch.nn.functional.pad(inputs, pad=(12, 13))
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                output = output[:, :, :, 12:-13]
                # output = output.clamp(min=-1., max=1.)
                loss = criterion(output, target)
                val_loss += (loss.detach().item() * B)
        val_loss /= len(val_loader.dataset)

        SaveModel(epoch, val_loss, model, optimizer, save_root, best=False)

        if val_loss <= best_performance:
            SaveModel(epoch, val_loss, model, optimizer, save_root, best=True)
            best_performance = val_loss

        print("Epoch:{} Training Loss:{} Validation Loss:{}; Best Perfomance:{}"
              .format(epoch, train_loss, val_loss, best_performance))
        with open('training_log_Unet_'+ str(num_epochs) + '_' + str(layer_num) + '_' + str(loaderoffset) + '.txt', 'a') as f:
            f.write('epoch: %s; train_loss: %s, val_loss: %s, best_performance: %s\n'
                    % (epoch, train_loss, val_loss, best_performance))

    with open('training_log_Unet_'+ str(num_epochs) + '_' + str(layer_num) + '_' + str(loaderoffset) + '.txt', 'a') as f:
        f.write("total train time:" + str(time.time()-time_start) + "\n")
    print("total train time:" , time.time()-time_start)

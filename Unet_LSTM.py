import torch
import torch.nn as  nn

class LstmCell(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(LstmCell, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.forget_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.input_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.cell_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, stride, padding),
            # nn.Tanh(),
            nn.ReLU()
        )
        self.output_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, stride, padding),
            nn.Sigmoid()
        )
        self.act_for_c = nn.Tanh()
        
    def forward(self, input_t, h_prev, c_prev):
        
        input_for_gates = torch.cat([input_t, h_prev], dim=1)

        # fg: forget gate; ig: input gate; cg: cell gate; og: output gate
        f_t = self.forget_gate(input_for_gates)
        i_t = self.input_gate(input_for_gates)
        c_bar_t = self.cell_gate(input_for_gates)
        o_t = self.output_gate(input_for_gates)
        
        c_t = f_t * c_prev + i_t * c_bar_t
        h_t = o_t * self.act_for_c(c_t)
        
        return h_t, c_t

class ConvLstm(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3),
                 stride=(1,1), padding=(1,1), return_mode='all'):
        super(ConvLstm, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mode = return_mode

        self.lstm_cell = LstmCell(self.input_dim, hidden_dim, self.kernel_size, self.stride, self.padding)

    def forward(self, input_seq):

        B, C, T, H, W = input_seq.size()
        h_collection = []
        c_collection = []
        h_t = torch.zeros((B, self.hidden_dim, H, W)).to(input_seq.device)
        c_t = torch.zeros((B, self.hidden_dim, H, W)).to(input_seq.device)
        for t in range(T):
            input_t = input_seq[:, :, t, ...]
            h_t, c_t = self.lstm_cell(input_t, h_t, c_t)
            h_collection.append(h_t.view(B, -1, 1, H, W))
            c_collection.append(c_t.view(B, -1, 1, H, W))

        h_all = torch.cat(h_collection, dim=2)
        # c_all = torch.cat(c_collection, dim=1)
        if self.return_mode == "final":
            return h_t
        else: #all
            return h_all

class Unet_Lstm(nn.Module):

    def __init__(self, layer_num=3, input_dim=1, base_dim=32, lstm_mode='all', lstm_layer_num=1):
        super(Unet_Lstm, self).__init__()

        self.layer_num = layer_num
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.lstm_mode = lstm_mode
        self.lstm_layer_num = lstm_layer_num

        self.downLayers = nn.ModuleList([])
        self.catLayers = nn.ModuleList([])
        self.upLayers = nn.ModuleList([])
        self.lstmLayers = nn.ModuleList([])
        self.dims = [self.input_dim]
        for i in range(self.layer_num):
            self.dims.append(self.base_dim * (2**i))

        for i in range(self.layer_num):
            dim_in, dim_out = self.dims[i], self.dims[i + 1]
            convLstms = nn.Sequential()
            if i == 0:
                downConvs = self.DownBlock(dim_in, dim_out, (3, 3), (1, 1), (1, 1))
                catConvs = self.DownBlock(dim_out * 2, dim_out, (3, 3), (1, 1), (1, 1))
                upConvs = self.PredictBlock(dim_out, dim_in)
            elif i == self.layer_num - 1:
                downConvs = self.DownBlock(dim_in, dim_out, (3, 3), (2, 2), (1, 1))
                catConvs = nn.Sequential()
                upConvs = self.UpBlock(dim_out, dim_in, (4, 4), (2, 2), (1, 1))
            else:
                downConvs = self.DownBlock(dim_in, dim_out, (3, 3), (2, 2), (1, 1))
                catConvs = self.DownBlock(dim_out * 2, dim_out, (3, 3), (1, 1), (1, 1))
                upConvs = self.UpBlock(dim_out, dim_in, (4, 4), (2, 2), (1, 1))

            for j in range(self.lstm_layer_num):
                lstm = ConvLstm(dim_out, dim_out, return_mode=self.lstm_mode)
                batchNorm = nn.BatchNorm3d(num_features=dim_out)
                convLstms.append(lstm)
                convLstms.append(batchNorm)


            self.downLayers.append(downConvs)
            self.catLayers.append(catConvs)
            self.upLayers.append(upConvs)
            self.lstmLayers.append(convLstms)

    def DownBlock(self, input_dim, output_dim, kernel_size, stride, padding):
        convBlock = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        return convBlock

    def UpBlock(self, input_dim, output_dim, kernel_size, stride, padding):
        convBlock = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        return convBlock

    def PredictBlock(self, input_dim, output_dim):
        convBlock = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//2, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(input_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim//2, input_dim//2, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(input_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim//2, output_dim, (1, 1), (1, 1), (0, 0)),
            nn.Tanh()
        )
        return convBlock

    def forward(self, input_seq):

        B, C, T, H, W = input_seq.size()
        feat_layer = input_seq.permute(0, 2, 1, 3, 4).contiguous()
        feat_layer = feat_layer.view(B*T, C, H, W)
        downFeats = []
        for layer in range(self.layer_num):
            feat_layer = self.downLayers[layer](feat_layer)
            _, c, h, w = feat_layer.size()
            downFeats.append(feat_layer.view(B, T, c, h, w).permute(0, 2, 1, 3, 4).contiguous())

        lstmFeats = [self.lstmLayers[layer](feat_layer) for layer, feat_layer in enumerate(downFeats)]

        for layer in range(self.layer_num-1, -1, -1):
            lstm_final_feat = lstmFeats[layer][:, :, -1, ...]
            if layer == self.layer_num - 1:
                feat_layer = self.upLayers[layer](lstm_final_feat)
            else:
                # print('1:',lstm_final_feat.size(), feat_layer.size())
                concat_feat = torch.cat([lstm_final_feat, feat_layer], dim=1)
                # print('2:', concat_feat.size())
                feat_layer = self.catLayers[layer](concat_feat)
                # print('3:', feat_layer.size())
                feat_layer = self.upLayers[layer](feat_layer)

        prediction = feat_layer

        return prediction


class SeqModel(nn.Module):

    def __init__(self, layer_num=3, input_dim=1, base_dim=32, lstm_mode='all'):
        super(SeqModel, self).__init__()
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.lstm_mode = lstm_mode

        # self.conv = nn.Sequential([
        #     nn.Conv2d(self.input_dim, self.base_dim, (3,3), (1,1), (1,1)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.base_dim, self.base_dim*2, (3,3), (1,1), (1,1)),
        #     nn.ReLU(inplace=True),
        # ])

        self.lstm = nn.Sequential()
        for i in range(self.layer_num):
            self.lstm.append(ConvLstm(self.input_dim, self.base_dim, return_mode=self.lstm_mode))

        self.predict = nn.Sequential(
            nn.Conv2d(self.base_dim,self.base_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.base_dim, self.base_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.base_dim // 2, self.input_dim, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, seq):
        output = self.lstm(seq)
        B, C, T, H, W = output.size()
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = output.view(B * T, C, H, W)

        # Return only the last output frame
        # output = self.conv(output[:, :, -1])
        output = self.predict(output)
        BT, C, H, W = output.size()
        output = output.view(B, T, C, H, W)[:, -1]
        return output




        
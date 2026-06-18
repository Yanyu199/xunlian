# tem_model_factory/core/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hid_dim, bi_layer_size, dropout, INPUT_DIM=1):
        super().__init__()
        self.layer_num = bi_layer_size
        self.embedding = nn.Sequential(
            nn.Linear(INPUT_DIM, hid_dim),
            nn.Tanh()
        )
        self.first_rnn = nn.LSTM(hid_dim, hid_dim, bidirectional=True)
        self.first_l_norm = nn.LayerNorm(hid_dim * 2)

        if self.layer_num > 1:
            self.rnns = nn.ModuleList(
                [nn.LSTM(hid_dim * 2, hid_dim, bidirectional=True) for _ in range(self.layer_num - 1)])
            self.l_norm = nn.ModuleList([nn.LayerNorm(hid_dim * 2) for _ in range(self.layer_num - 1)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src)
        src = self.dropout(src)

        hs, cs = [], []
        output, (h, c) = self.first_rnn(src)
        output = self.dropout(self.first_l_norm(output))
        hs.append(h)
        cs.append(c)

        for i in range(self.layer_num - 1):
            output, (h, c) = self.rnns[i](output)
            output = self.dropout(self.l_norm[i](output))
            hs.append(h)
            cs.append(c)

        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        return output, (hs, cs)


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim * 2, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, hid_dim, en_layer_num, dropout, attention):
        super().__init__()
        self.layer_num = en_layer_num * 2
        self.attention = attention
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(hid_dim * 2, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

        if self.layer_num > 1:
            self.rnns = nn.ModuleList([nn.LSTM(hid_dim, hid_dim) for _ in range(self.layer_num - 1)])
            self.l_norm = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(self.layer_num - 1)])

        self.fcn = nn.Sequential(nn.Linear(hid_dim, 10), nn.Tanh(), self.dropout, nn.Linear(10, 1))
        self.fcn_thick = nn.Sequential(nn.Linear(hid_dim, 10), nn.Tanh(), self.dropout, nn.Linear(10, 1))

    def forward(self, hs, cs, encoder_outputs):
        a = self.attention(hs[-1], encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)

        hs_new, cs_new = [], []
        h, c = hs[0].unsqueeze(0), cs[0].unsqueeze(0)

        output, (h, c) = self.rnn(weighted, (h, c))
        hs_new.append(h)
        cs_new.append(c)
        output = self.dropout(self.ln(output))

        for i in range(self.layer_num - 1):
            h, c = hs[i + 1].unsqueeze(0), cs[i + 1].unsqueeze(0)
            output, (h, c) = self.rnns[i](output, (h, c))
            hs_new.append(h)
            cs_new.append(c)
            output = self.dropout(self.l_norm[i](output))

        prediction = self.fcn(output).squeeze(0)
        prediction_thick = self.fcn_thick(output).squeeze(0)
        prediction = torch.cat([prediction, prediction_thick], dim=1)

        # 修复 4: 激活层偏移对齐 (严格保证输出为正，并和数据集预处理的+10呼应)
        prediction = F.relu(prediction + 10.0)

        hs_new = torch.cat(hs_new, dim=0)
        cs_new = torch.cat(cs_new, dim=0)
        return prediction, (hs_new, cs_new)


class TEM_Seq2Seq_Net(nn.Module):
    def __init__(self, layer_num=5, hid_dim=128, rnn_layers=2, dropout=0.1):
        super().__init__()
        self.layer_num = layer_num
        self.output_dim = layer_num * 2 - 1  # 例如5层模型：5个电阻率 + 4个厚度 = 9

        # 特征维度始终是1 (因为输入的是时间序列的值)
        self.encoder = Encoder(hid_dim=hid_dim, bi_layer_size=rnn_layers, dropout=dropout, INPUT_DIM=1)
        self.attention = Attention(hid_dim=hid_dim)
        self.decoder = Decoder(hid_dim=hid_dim, en_layer_num=rnn_layers, dropout=dropout, attention=self.attention)

    def forward(self, src):
        # src shape: [batch_size, seq_len]
        batch_size = src.shape[0]

        # 转换为 RNN 所需的 [seq_len, batch_size, feature_dim=1]
        src = src.transpose(0, 1).unsqueeze(-1)

        # 容器: [解码步长(即地层数), batch_size, 2(电阻率和厚度)]
        outputs = torch.zeros(self.layer_num, batch_size, 2).to(src.device)

        encoder_outputs, (hs, cs) = self.encoder(src)

        # 修复 6: 严格按地层数量进行解码，每次生成当前层的 (电阻率, 厚度)
        for t in range(self.layer_num):
            output, (hs, cs) = self.decoder(hs, cs, encoder_outputs)
            outputs[t] = output

        # 维度转换: [batch_size, layer_num, 2] -> [batch_size, layer_num * 2]
        outputs = outputs.permute(1, 0, 2).reshape(batch_size, -1)

        # 截断最后一个元素的厚度 (因为最底层是半空间，无厚度)
        return outputs[:, :self.output_dim]
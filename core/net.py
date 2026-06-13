# tem_model_factory/core/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, hid_dim, bi_layer_size, dropout, INPUT_DIM=1):
        """
        编码器：将输入的瞬变电磁时间序列数据编码为高维隐藏状态
        """
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
        # src: [src_len, batch_size, INPUT_DIM]
        src = self.embedding(src)  # [src_len, batch_size, hid_dim]
        src = self.dropout(src)

        hs = []
        cs = []

        # 第一层双向 LSTM
        output, (h, c) = self.first_rnn(src)
        output = self.first_l_norm(output)
        output = self.dropout(output)
        hs.append(h)
        cs.append(c)

        # 叠加更深层的 LSTM
        for i in range(self.layer_num - 1):
            output, (h, c) = self.rnns[i](output)
            output = self.l_norm[i](output)
            output = self.dropout(output)
            hs.append(h)
            cs.append(c)

        # 将所有层的 hidden state 拼接
        hs = torch.cat(hs, dim=0)  # [layer_num * 2, batch, enc_dim]
        cs = torch.cat(cs, dim=0)  # [layer_num * 2, batch, enc_dim]

        # output: [src_len, batch_size, hid_dim * 2]
        return output, (hs, cs)


class Attention(nn.Module):
    """
    Bahdanau 注意力机制：帮助解码器在预测某一地层时，自动聚焦到输入时间道中最相关的部分
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim * 2, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]
        src_len = encoder_outputs.shape[0]

        # 扩展 hidden 维度以匹配序列长度
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, dec_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, enc_hid_dim * 2]

        # 计算注意力能量
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))  # [batch_size, src_len, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, hid_dim, en_layer_num, dropout, attention):
        """
        解码器：根据编码器的上下文和注意力，逐步解码出每一层的电阻率和厚度
        """
        super().__init__()
        self.layer_num = en_layer_num * 2
        self.attention = attention
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(hid_dim * 2, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

        if self.layer_num > 1:
            self.rnns = nn.ModuleList([nn.LSTM(hid_dim, hid_dim) for _ in range(self.layer_num - 1)])
            self.l_norm = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(self.layer_num - 1)])

        # 预测电阻率的全连接层
        self.fcn = nn.Sequential(
            nn.Linear(hid_dim, 10),
            nn.Tanh(),
            self.dropout,
            nn.Linear(10, 1)
        )
        # 预测层厚的全连接层
        self.fcn_thick = nn.Sequential(
            nn.Linear(hid_dim, 10),
            nn.Tanh(),
            self.dropout,
            nn.Linear(10, 1)
        )

    def forward(self, hs, cs, encoder_outputs):
        # hs/cs: [layers_num, batch_size, dec_hid_dim]
        # encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]

        # 1. 计算注意力权重并加权上下文
        a = self.attention(hs[-1], encoder_outputs).unsqueeze(1)  # [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, enc_hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, enc_hid_dim * 2]

        hs_new = []
        cs_new = []

        # 2. LSTM 层层递进 (带 LayerNorm)
        h = hs[0].unsqueeze(0)  # [1, batch, hid_dim]
        c = cs[0].unsqueeze(0)
        output, (h, c) = self.rnn(weighted, (h, c))
        hs_new.append(h)
        cs_new.append(c)
        output = self.ln(output)
        output = self.dropout(output)

        for i in range(self.layer_num - 1):
            h = hs[i + 1].unsqueeze(0)
            c = cs[i + 1].unsqueeze(0)
            output, (h, c) = self.rnns[i](output, (h, c))
            hs_new.append(h)
            cs_new.append(c)
            output = self.l_norm[i](output)
            output = self.dropout(output)

        # 3. 输出层计算 (使用 ReLU 确保电阻率和厚度严格为正数)
        prediction = self.fcn(output).squeeze(0)
        prediction_thick = self.fcn_thick(output).squeeze(0)

        # 拼接成 [batch_size, 2] 的输出 (即本层的电阻率与厚度)
        prediction = torch.cat([prediction, prediction_thick], dim=1)
        prediction = F.relu(prediction)

        hs_new = torch.cat(hs_new, dim=0)
        cs_new = torch.cat(cs_new, dim=0)

        return prediction, (hs_new, cs_new)


class TEM_Seq2Seq_Net(nn.Module):
    """
    高度封装的顶层网络接口 (完美适配你的 train.py)
    """

    def __init__(self, input_dim=30, output_dim=9, hid_dim=128, rnn_layers=2, dropout=0.1):
        """
        :param input_dim: 输入的时间道数量 (默认 30)
        :param output_dim: 需要反演输出的目标数量 (如 5层模型 = 5个电阻率 + 4个厚度 = 9)
        :param hid_dim: RNN 隐藏层维度
        :param rnn_layers: RNN 层数
        """
        super().__init__()
        self.output_dim = output_dim
        # 计算需要解码的循环次数 (由于每次解码输出一组 [电阻率, 厚度] 共 2 个值)
        # 如果 output_dim 是 9, 我们需要循环 5 次解码出 10 个值，最后裁掉多余的一个厚度
        self.decode_steps = (output_dim + 1) // 2

        # 实例化各个组件 (将原本的输入特征数当作 1，时间道序列长度当作 input_dim)
        self.encoder = Encoder(hid_dim=hid_dim, bi_layer_size=rnn_layers, dropout=dropout, INPUT_DIM=1)
        self.attention = Attention(hid_dim=hid_dim)
        self.decoder = Decoder(hid_dim=hid_dim, en_layer_num=rnn_layers, dropout=dropout, attention=self.attention)

    def forward(self, src):
        """
        :param src: [batch_size, input_dim] (比如 [128, 30])
        :return: [batch_size, output_dim] (比如 [128, 9])
        """
        batch_size = src.shape[0]

        # 1. 维度转换以适配 RNN:
        # [batch_size, 30] -> [30(seq_len), batch_size, 1(feature_dim)]
        src = src.transpose(0, 1).unsqueeze(-1)

        # 2. 准备容器存放解码输出
        # outputs shape: [decode_steps, batch_size, 2]
        outputs = torch.zeros(self.decode_steps, batch_size, 2).to(src.device)

        # 3. 编码器计算
        encoder_outputs, (hs, cs) = self.encoder(src)

        # 4. 循环逐步解码
        for t in range(self.decode_steps):
            output, (hs, cs) = self.decoder(hs, cs, encoder_outputs)
            outputs[t] = output

        # 5. 后处理输出格式
        # outputs: [decode_steps, batch_size, 2] -> [batch_size, decode_steps, 2]
        outputs = outputs.permute(1, 0, 2)
        # 展平为 [batch_size, decode_steps * 2]
        outputs = outputs.reshape(batch_size, -1)

        # 6. 切片返回精确所需的维度 (例如 10 个值中截取前 9 个)
        return outputs[:, :self.output_dim]
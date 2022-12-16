import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Att_cnn2d_model(nn.Module):
    """
    1.attribute embeddig(dimension reduction for one-hot features)
    2.attention
    3.cnn
    4.linear layer
    """

    def __init__(
        self,
        time_windows_dim: int,
        feat_dim: int,
        num_classes: int,
        attention_hidden_dim: int,
        cate_feat_num: int = 3,
        cate_embed_dim: int = 8,
        cate_unique_num: list = [1664, 216, 2500],
        filter_sizes: tuple = (2, 2),
        num_filters: int = 64,
        in_channels: int = 1
    ) -> None:
        super().__init__()
        self.time_windows_dim = time_windows_dim
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.attention_hidden_dim = attention_hidden_dim

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.cate_feat_num = cate_feat_num
        self.cate_unique_num = cate_unique_num

        # cate embedding layer
        # ['Location','Type','Target']
        self.cate_emdeds = nn.ModuleList([nn.Embedding(
            cate_unique_num[idx] + 1, cate_embed_dim) for idx in range(cate_feat_num)])

        # attention layer
        self.attention_W = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_V = nn.Parameter(torch.Tensor(
            self.attention_hidden_dim, 1).uniform_(0., 1.))

        # cnn layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_sizes
        )

        # FC layer
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(self.num_classes)
        )

    def attention_layer(
        self,
        X: torch.Tensor
    ):
        self.output_att = []
        # input_att = torch.split(X, self.time_windows_dim, dim=1)
        input_att = torch.split(X, 1, dim=1)  # 第二个参数是split_size!
        for index, x_i in enumerate(input_att):
            x_i = x_i.reshape(-1, self.feat_dim)
            c_i = self.attention(x_i, input_att, index)
            # combine two feature maps to one map
            # print("shape of x_i: {}".format(x_i.shape))
            # print("shape of c_i: {}".format(c_i.shape))

            inp = torch.concat([x_i, c_i], axis=1)
            self.output_att.append(inp)

        input_conv = torch.reshape(torch.concat(self.output_att, axis=1),
                                   [-1, self.time_windows_dim, self.feat_dim*2])

        self.input_conv_expanded = torch.unsqueeze(input_conv, 1)

        return self.input_conv_expanded

    def cnn_layer(
        self,
        input: torch.Tensor
    ):
        if len(input.shape) == 3:
            self.input_conv_expanded = torch.unsqueeze(input, 1)
        elif len(input.shape) == 4:
            self.input_conv_expanded = input
        else:
            print("Wrong conv input shape!")

        self.input_conv_expanded = F.relu(self.conv(input))

        return self.input_conv_expanded

    def attention(self, x_i, x, index):
        e_i = []
        c_i = []

        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.feat_dim)
            att_hidden = torch.tanh(torch.add(torch.matmul(
                x_i, self.attention_W), torch.matmul(output, self.attention_U)))
            e_i_j = torch.matmul(att_hidden, self.attention_V)
            e_i.append(e_i_j)

        e_i = torch.concat(e_i, axis=1)
        # print(f"e_i shape: {e_i.shape}")
        alpha_i = F.softmax(e_i, dim=1)
        alpha_i = torch.split(alpha_i, 1, 1)  # !!!

        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = output.reshape(-1, self.feat_dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)

        c_i = torch.reshape(torch.concat(c_i, axis=1),
                            [-1, self.time_windows_dim-1, self.feat_dim])
        c_i = torch.sum(c_i, dim=1)
        return c_i

    def cate_embedding_layer(self, X_cate):
        cate_embed_ret = []
        for idx in range(self.cate_feat_num):
            cate_embed_ret.append(self.cate_emdeds[idx](X_cate[:, idx]))
            # 3个(batch_size, emb_dim)

        cate_embed_ret = torch.concat(cate_embed_ret, dim=1)
        return cate_embed_ret  # (batch_size, emb_dim*3)

    def forward(self, X_nume, X_cate):
        # X shape be like: (batch_size, time_windows_dim, feat_dim)
        out = self.attention_layer(X_nume)

        out = self.cnn_layer(out)
        out = self.flatten(out)

        cate_embeds = self.cate_embedding_layer(X_cate)
        out = torch.concat([out, cate_embeds], dim=1)
        out = self.linears(out)

        return out

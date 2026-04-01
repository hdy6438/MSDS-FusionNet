import torch
import torch.nn as nn
import torch.nn.functional as F


class MutilStreamConvAblation(nn.Module):
    def __init__(self, d_model, output_len):
        super(MutilStreamConvAblation, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1)

        self.max_pool = nn.AdaptiveMaxPool1d(output_len)

        self.norm1 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        conv1 = F.leaky_relu(self.conv1(x))
        conv1 = F.dropout(conv1, p=0.2, training=self.training)
        conv1 = self.max_pool(conv1)

        x = self.norm1(conv1).permute(0, 2, 1)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return x


class BAFMAbtion(nn.Module):
    def __init__(self, d_model):
        super(BAFMAbtion, self).__init__()
        self.conv = nn.Conv1d(d_model * 2, d_model, kernel_size=1, stride=1)

    def forward(self, x1, x2, subject_feature=None):
        x = torch.cat([x1, x2], dim=-1).permute(0, 2, 1)
        x = self.conv(x)
        return x

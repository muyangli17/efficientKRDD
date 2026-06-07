import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticFun(nn.Module):
    def __init__(
        self, input_dim=2048, hidden_dim=1024, output_dim=1, act=F.relu, SN=False
    ):
        super(CriticFun, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = act
        if SN:
            self.fc1 = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim))
            self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim))
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def truncate_parameters(self, c=0.1):
        # 截断所有参数
        with torch.no_grad():
            for param in self.parameters():
                param.clamp_(-c, c)

    def set_parameters(self, value=1.0):
        # 设置所有参数为指定值
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(value)

    def get_parameters(self):
        # 获取所有参数
        params = []
        for param in self.parameters():
            params.append(param.data.clone())
        return params

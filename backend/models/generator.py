# backend/models/generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int = 7, ngf: int = 64, output_channels: int = 3) -> None:
        """
        latent_dim : 噪声向量维度
        n_classes  : 条件总类别数（例如角色类型3类 + 颜色主题4类 = 7）
        ngf        : 生成器基础通道数
        output_channels: 输出图像通道数（RGB：3）
        输出图像尺寸：64x64
        """
        super().__init__()
        cond_dim = 10  # 可调，条件信息经过 embedding 后的维度
        self.condition_embedding = nn.Linear(n_classes, cond_dim)
        # 总输入维度 = latent_dim + cond_dim
        in_dim = latent_dim + cond_dim

        self.net = nn.Sequential(
            # 输入: (batch, in_dim, 1, 1)
            nn.ConvTranspose2d(in_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 最终输出: (output_channels) x 64 x 64
        )

    def forward(self, noise, condition):
        # noise: (batch, latent_dim)
        # condition: (batch, n_classes) —— one-hot 编码
        cond_emb = self.condition_embedding(condition)  # (batch, cond_dim)
        x = torch.cat((noise, cond_emb), dim=1)  # (batch, latent_dim + cond_dim)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.net(x)

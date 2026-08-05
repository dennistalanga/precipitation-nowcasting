import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(
        self,
        sequence_length=6,
        predict_steps=6,
        input_channels=2,
        base_channels=32,
        num_groups=4
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        in_channels = sequence_length * input_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 256 -> 128

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 128 -> 64

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c3),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),  # 64 -> 128
            nn.Conv2d(c3, c2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c2),
            nn.ReLU(inplace=True),

            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),  # 128 -> 256
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1, predict_steps, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (batch, seq, data_ch, H, W)
        batch_size, _, _, h, w = x.shape

        # Permute to group like-channels together: (batch, data_ch, seq, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Flatten to 4D using automatic sizing: (batch, data_ch x seq, H, W)
        x = x.reshape(batch_size, -1, h, w)
        x = self.encoder(x)
        x = self.decoder(x)

        return torch.sigmoid(x)

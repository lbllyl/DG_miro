from torchvision import transforms as T
import torch
import torch.nn as nn

basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class SaltPeperNoise(nn.Module):
    def __init__(self, snr=0.7, image_range=(0, 255)):
        """
        Salt and pepper noise
        Args:
            snr: signal to noise ratio
            p: probability of noise
        """
        super().__init__()
        assert 0 <= snr <= 1, "snr should be in [0, 1]"
        self.snr = snr
        self.image_range = image_range

    def forward(self, x):
        """
        Args:
            x: input tensor
        """
        B, C, H, W = x.shape

        pepper_p = (1 - self.snr) / 2  # pepper noise probability
        salt_p = (1 - self.snr) / 2  # salt noise probability

        # generate noise
        noise_mask = torch.rand(B, 1, H, W)  # 3-channels should be the same
        noise_mask = noise_mask.expand(B, C, H, W)
        pepper_mask = noise_mask < pepper_p
        salt_mask = (noise_mask >= pepper_p) & (noise_mask < (pepper_p + salt_p))

        x = x.clone()  # TODO check if clone is necessary

        x[pepper_mask] = self.image_range[0]
        x[salt_mask] = self.image_range[1]

        return x

    def __repr__(self):
        return self.__class__.__name__ + f"(snr={self.snr})"

class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1, image_range=(0, 255)):
        """

        Args:
            mean:
            std:
            image_range: the range of the image
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.image_range = image_range

    def forward(self, x):
        noise = torch.randn_like(x) * self.std + self.mean
        return torch.clamp(x + noise, self.image_range[0], self.image_range[1])

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"

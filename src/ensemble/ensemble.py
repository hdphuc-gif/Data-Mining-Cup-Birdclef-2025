import torch
import torch.nn as nn
from src.model_a.model import EfficientNetWrapper
from src.model_b.model import ConvNeXtImageWrapper
from src.model_c.model import ConvNeXtBirdWrapper
from timm import create_model


class EnsembleModel(nn.Module):
    def __init__(self, model_a, model_b, model_c, weights=(0.2, 0.6, 0.2)):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.model_c = model_c
        self.wa, self.wb, self.wc = weights

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        out_c = self.model_c(x)
        return self.wa * out_a + self.wb * out_b + self.wc * out_c


if __name__ == "__main__":
    x = torch.randn(2, 1, 224, 224)
    model_a = EfficientNetWrapper()
    model_b = ConvNeXtImageWrapper()
    model_c = ConvNeXtBirdWrapper()
    ensemble = EnsembleModel(model_a, model_b, model_c)
    y = ensemble(x)
    print(y.shape)  # Expected: (2, 206)


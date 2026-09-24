from typing import Union, List
import torch
from torch import nn, Size

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)


        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.normalized_shape == input.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = input.mean(dim=dims, keepdim=True)
        mean_x2 = (input ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2

        x_norm = (input - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.gain + self.bias
        return x_norm

#simple test
def _test():
    layer_norm = LayerNorm(normalized_shape=3)
    x = torch.randn(2, 3)
    output = layer_norm(x)
    print("Input:", x)
    print("Output:", output)

if __name__ == "__main__":
    _test()
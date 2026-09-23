import torch
from torch import nn

class BatchNorm(nn.Module):
    # channels is the number of features in the input
    # eps is ϵ, used in  Var[x (k)]+ϵ​for numerical stability
    # momentum is the momentum in taking the exponential moving average
    # affine is whether to scale and shift the normalized value
    # track_running_stats is whether to calculate the moving averages or mean and variance */

    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    # Create parameters for γ and β for scale and shift
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))
    # Create buffers for the exponential moving averages of mean and variance
        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x.shape[1]

        x = x.view(batch_size, self.channels, -1)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2], keepdim=True)
            # Compute the variance using the formula Var[x] = E[x^2] - (E[x])^2
            mean_x2 = (x ** 2).mean(dim=[0, 2], keepdim=True)
            var = mean_x2 - mean ** 2

            if self.training and self.track_running_stats:
                # Update the exponential moving averages of mean and variance
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean.squeeze()
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var.squeeze()
            else:
                mean = self.exp_mean
                var = self.exp_var

        # Normalize the input using the mean and variance
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        # Scale and shift the normalized input if affine is True
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        # Reshape the output back to the original shape
        return x_norm.view(x_shape)


def _test():
    # Create a BatchNorm layer with 3 channels
    bn = BatchNorm(3)
    # Create a random input tensor with shape [2, 3, 4, 4]
    x = torch.randn(2, 3, 4, 4)
    # Pass the input through the BatchNorm layer
    output = bn(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    _test()



import torch
import torch.nn as nn
import math


class DynamicBiasCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        velocity_init=0.1,
        bias_decay=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        self.bias_decay = bias_decay
        self.tanh = nn.Tanh()

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def _initialize_parameters(self, *args, **kwargs):
        # self.bias = self.bias.expand(args[0].shape)
        batch_size, in_channels, height, width = args[0].shape

        output_height = (
            (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]
        ) + 1
        output_width = (
            (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]
        ) + 1
        
        self.register_buffer(
            "dynamic_bias",
            0.5
            + torch.zeros(
                batch_size,
                self.out_channels,
                output_height,
                output_width,
                device=args[0].device,
            ),
        )

        self.velocity = nn.Parameter(
            torch.zeros(
                batch_size,
                self.out_channels,
                output_height,
                output_width,
                device=args[0].device,
            )
        )

        return batch_size, in_channels, output_height, output_width

    def forward(self, x):
        
        # Apply the bias adjustment as in the NN implementation
        current_bias = self.velocity * self.dynamic_bias
        z = self.conv(x) + current_bias

        # Update bias in-place with no_grad to prevent tracking in backward pass
        if not torch.is_grad_enabled():
            self.dynamic_bias = current_bias.detach().clone() + 1e-3

        return self.tanh(z)

    def reset_bias(self):
        """Reset the current bias state"""
        nn.init.zeros_(self.bias)

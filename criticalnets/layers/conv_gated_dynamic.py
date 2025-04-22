import torch
import torch.nn as nn
import math
from torch.autograd import grad

# Import the parent class (assuming it's in the same module)
from criticalnets.layers import DynamicBiasCNN


class GatedDynamicBiasCNN(DynamicBiasCNN):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        velocity_init=0.1,
        bias_decay=1.0,
    ):
        """
        Initialize the Gated Dynamic Bias Convolutional Neural Network

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int or tuple): Size of the convolutional kernel
            stride (int or tuple): Stride of the convolution (default: 1)
            padding (int or tuple): Padding added to all sides of the input (default: 0)
            velocity_init (float): Initial velocity value (default: 0.1)
            bias_decay (float): Bias decay factor (default: 1.0)
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            velocity_init,
            bias_decay,
        )

        self.register_buffer("dynamic_bias", None)
        self.velocity_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        # Initialize velocity_conv weights
        nn.init.kaiming_uniform_(self.velocity_conv.weight, a=math.sqrt(5))

        self.Tanh = nn.Tanh()

    def _initialize_parameters(self, *args, **kwargs):
        batch_size, in_channels, output_height, output_width = (
            super()._initialize_parameters(*args, **kwargs)
        )
        # Use register_buffer instead of nn.Parameter to avoid gradient issues
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
        return batch_size, in_channels, output_height, output_width

    def forward(self, x):
        # Store current dynamic bias for criticality calculations
        self._current_dynamic_bias = None
        # Calculate velocity using the velocity convolution
        velocity = self.Tanh(self.velocity_conv(x))
        current_bias = self.dynamic_bias * velocity
        self._current_dynamic_bias = current_bias
        # Use the current bias for this forward pass
        a = self.conv(x) + current_bias
        # Update the dynamic bias buffer after computation
        if not torch.is_grad_enabled():
            # Only update when not computing gradients
            self.dynamic_bias = current_bias.detach().clone() + 1e-3

        return self.tanh(a)

    def update_bias(self):
        """
        This method should be called after backward() to update the dynamic bias.
        Call this in your training loop after loss.backward() but before optimizer.step()
        """
        if self._current_dynamic_bias is not None:
            # Update the dynamic bias with the last computed value
            # Since we're outside of autograd context now, this won't affect gradients
            self.dynamic_bias = self._current_dynamic_bias.detach().clone()
            self._current_dynamic_bias = None

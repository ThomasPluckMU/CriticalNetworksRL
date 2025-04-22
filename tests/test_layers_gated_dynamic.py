import pytest
import torch
import math
from criticalnets.layers import gated_dynamic


def test_layer_initialization():
    """Test GatedDynamicBiasNN initialization"""
    layer = gated_dynamic.GatedDynamicBiasNN(5, 10)

    # Verify parameters
    assert layer.input_size == 5
    assert layer.hidden_size == 10
    assert layer.weight.shape == (10, 5)
    assert layer.velocity_weight.shape == (10, 5)

    # Verify weights are initialized with Kaiming uniform
    fan_in = 5
    bound = math.sqrt(3.0) * math.sqrt(2.0 / fan_in)
    assert torch.all(layer.weight >= -bound) and torch.all(layer.weight <= bound)
    assert torch.all(layer.velocity_weight >= -bound) and torch.all(
        layer.velocity_weight <= bound
    )


def test_forward_pass():
    """Test forward pass with sample input"""
    layer = gated_dynamic.GatedDynamicBiasNN(5, 10)
    x = torch.randn(3, 5)  # batch_size=3, input_size=5

    # First pass initializes bias
    output = layer(x)
    assert output.shape == (3, 10)
    assert layer.current_bias is not None
    assert layer.current_bias.shape == (3, 10)

    # Second pass should show bias effects
    output2 = layer(x)
    assert not torch.equal(output, output2)


def test_velocity_effects():
    """Test velocity computation effects"""
    layer = gated_dynamic.GatedDynamicBiasNN(5, 10)
    x = torch.randn(3, 5)

    # Get initial bias
    layer(x)
    initial_bias = layer.current_bias.clone()

    # Run forward pass to trigger update
    layer(x)
    updated_bias = layer.current_bias

    # Verify bias was updated
    assert not torch.equal(initial_bias, updated_bias)


def test_bias_reset():
    """Test bias reset functionality"""
    layer = gated_dynamic.GatedDynamicBiasNN(5, 10)
    x = torch.randn(3, 5)

    # First pass initializes bias
    layer(x)
    bias1 = layer.current_bias.clone()

    # Reset bias
    layer.reset_bias()
    bias_reset = layer.current_bias

    # Verify bias was properly reset
    assert not torch.equal(bias1, bias_reset)

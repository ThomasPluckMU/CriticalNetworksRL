import pytest
import torch
from criticalnets.layers import conv_dynamic

def test_conv_dynamic_import():
    """Test that conv_dynamic module imports correctly"""
    assert conv_dynamic is not None

def test_layer_initialization():
    """Test DynamicBiasCNN initialization"""
    layer = conv_dynamic.DynamicBiasCNN(
        in_channels=3, 
        out_channels=16,
        kernel_size=3
    )
    assert layer is not None
    assert isinstance(layer.conv, torch.nn.Conv2d)
    assert layer.conv.in_channels == 3
    assert layer.conv.out_channels == 16
    assert layer.conv.kernel_size == (3, 3)

def test_forward_pass():
    """Test forward pass with sample input"""
    layer = conv_dynamic.DynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, channels=3, height=32, width=32
    output = layer(x)
    
    assert output.shape == (2, 16, 32, 32)
    assert layer.current_bias_maps is not None
    assert layer.current_bias_maps.shape == (2, 16, 32, 32)

def test_reset_bias():
    """Test bias reset functionality"""
    layer = conv_dynamic.DynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    output = layer(x)  # Initialize with output shape
    
    original_bias = layer.current_bias_maps.clone()
    # Store output shape before resetting
    out_shape = output.shape
    layer.reset_bias()
    # Reinitialize with stored shape
    layer._initialize_parameters(out_shape)
    new_bias = layer.current_bias_maps
    
    assert not torch.equal(original_bias, new_bias)
    assert torch.all(new_bias == 0)

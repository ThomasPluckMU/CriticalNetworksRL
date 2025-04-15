import pytest
import torch
from criticalnets.layers import conv_gated_dynamic

def test_layer_initialization():
    """Test GatedDynamicBiasCNN initialization"""
    layer = conv_gated_dynamic.GatedDynamicBiasCNN(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        ema_factor=0.95
    )
    
    # Verify parameters
    assert layer.in_channels == 3
    assert layer.out_channels == 16
    assert layer.kernel_size == (3, 3)
    assert layer.ema_factor == 0.95
    
    # Verify weights are initialized
    assert layer.conv.weight is not None
    assert layer.velocity_conv.weight is not None
    
    # Verify initial state
    assert layer.base is None
    assert layer.current_bias_maps is None

def test_forward_pass():
    """Test forward pass with sample input"""
    layer = conv_gated_dynamic.GatedDynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, channels=3, height=32, width=32
    
    # First pass initializes parameters
    output = layer(x)
    assert output.shape == (2, 16, 32, 32)
    assert layer.base is not None
    assert layer.current_bias_maps is not None
    assert layer.current_bias_maps.shape == (2, 16, 32, 32)
    
    # Second pass should show bias effects
    output2 = layer(x)
    assert not torch.equal(output, output2)

def test_ema_update():
    """Test EMA update mechanism"""
    layer = conv_gated_dynamic.GatedDynamicBiasCNN(3, 16, 3, padding=1, ema_factor=0.9)
    x = torch.randn(2, 3, 32, 32)
    
    # Get initial bias maps
    layer(x)
    initial_bias = layer.current_bias_maps.clone()
    
    # Run forward pass to trigger update
    layer(x)
    updated_bias = layer.current_bias_maps
    
    # Verify EMA update occurred
    assert not torch.equal(initial_bias, updated_bias)

def test_bias_reset():
    """Test bias reset functionality"""
    layer = conv_gated_dynamic.GatedDynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    
    # First pass initializes bias
    output1 = layer(x)
    bias1 = layer.current_bias_maps.clone()
    
    # Reset bias
    layer.reset_bias()
    bias_reset = layer.current_bias_maps
    
    # Verify bias was properly reset
    assert not torch.equal(bias1, bias_reset)

def test_batch_size_change():
    """Test handling of batch size changes"""
    layer = conv_gated_dynamic.GatedDynamicBiasCNN(3, 16, 3, padding=1)
    
    # First pass with batch_size=2
    x1 = torch.randn(2, 3, 32, 32)
    output1 = layer(x1)
    assert layer.current_bias_maps.shape[0] == 2
    
    # Second pass with batch_size=4
    x2 = torch.randn(4, 3, 32, 32)
    output2 = layer(x2)
    assert layer.current_bias_maps.shape[0] == 4

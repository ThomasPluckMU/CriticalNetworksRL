import pytest
import torch
import numpy as np
from criticalnets.layers import conv_deadweight_dynamic

def test_layer_initialization():
    """Test DeadWeightDynamicBiasCNN initialization"""
    layer = conv_deadweight_dynamic.DeadWeightDynamicBiasCNN(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        spectral_radius=0.9,
        connectivity=0.8,
        seed=42
    )
    
    # Verify parameters
    assert layer.in_channels == 3
    assert layer.out_channels == 16
    assert layer.kernel_size == (3, 3)
    
    # Verify weights are frozen
    assert not layer.conv.weight.requires_grad
    assert layer.velocity_conv.weight.requires_grad
    
    # Verify weight initialization
    weights = layer.conv.weight.detach().numpy()
    assert np.any(weights != 0)  # Should have some non-zero weights
    assert np.mean(weights != 0) < 0.9  # Should have sparsity from connectivity=0.8

def test_forward_pass():
    """Test forward pass with sample input"""
    layer = conv_deadweight_dynamic.DeadWeightDynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, channels=3, height=32, width=32
    
    # First pass initializes bias
    output = layer(x)
    assert output.shape == (2, 16, 32, 32)
    assert layer.current_bias_maps is not None
    assert layer.current_bias_maps.shape == (2, 16, 32, 32)
    
    # Second pass should show bias effects
    output2 = layer(x)
    assert not torch.equal(output, output2)

def test_bias_reset():
    """Test bias reset functionality"""
    layer = conv_deadweight_dynamic.DeadWeightDynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    
    # First pass initializes bias
    output1 = layer(x)
    bias1 = layer.current_bias_maps.clone()
    
    # Reset bias
    layer.reset_bias()
    assert layer.current_bias_maps is None
    
    # Second pass reinitializes bias
    output2 = layer(x)
    bias2 = layer.current_bias_maps
    
    # Verify bias was properly reset
    assert not torch.equal(bias1, bias2)

def test_velocity_computation():
    """Test velocity computation effects"""
    layer = conv_deadweight_dynamic.DeadWeightDynamicBiasCNN(3, 16, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    
    # Get initial output
    output1 = layer(x)
    
    # Modify velocity weights to see effect
    with torch.no_grad():
        layer.velocity_conv.weight.fill_(0.5)
        layer.velocity_conv.bias.fill_(0.1)
    
    output2 = layer(x)
    assert not torch.equal(output1, output2)

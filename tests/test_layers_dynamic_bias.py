import pytest
import torch
from criticalnets.layers import dynamic_bias

def test_dynamic_bias_import():
    """Test that dynamic_bias module imports correctly"""
    assert dynamic_bias is not None

def test_base_class_initialization():
    """Test DynamicBiasBase initialization"""
    layer = dynamic_bias.DynamicBiasBase(input_size=10, hidden_size=20)
    assert layer.input_size == 10
    assert layer.hidden_size == 20
    assert layer.current_bias is None

def test_nn_layer_initialization():
    """Test DynamicBiasNN initialization"""
    layer = dynamic_bias.DynamicBiasNN(input_size=10, hidden_size=20)
    assert layer.weight.shape == (20, 10)
    assert layer.velocity.shape == (20,)
    assert layer.current_bias is None

def test_forward_pass():
    """Test forward pass with sample input"""
    layer = dynamic_bias.DynamicBiasNN(input_size=5, hidden_size=3)
    x = torch.randn(2, 5)  # batch_size=2, input_size=5
    
    output = layer(x)
    assert output.shape == (2, 3)
    assert layer.current_bias is not None
    assert layer.current_bias.shape == (2, 3)

def test_bias_reset():
    """Test bias reset functionality"""
    layer = dynamic_bias.DynamicBiasNN(input_size=5, hidden_size=3)
    x = torch.randn(2, 5)
    
    # First pass - initializes bias to zeros
    output1 = layer(x)
    bias1 = layer.current_bias.clone()
    
    # Second pass - should show bias effects
    output2 = layer(x) 
    bias2 = layer.current_bias.clone()
    
    # Verify bias is changing
    assert not torch.equal(bias1, bias2)
    
    # Reset bias
    layer.reset_bias()
    assert layer.current_bias is None
    
    # After reset, first pass should be same as original first pass
    output_reset = layer(x)
    assert torch.allclose(output1, output_reset, atol=1e-6)

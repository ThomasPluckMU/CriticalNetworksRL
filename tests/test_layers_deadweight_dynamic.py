import pytest
import torch
import numpy as np
from criticalnets.layers import deadweight_dynamic


def test_layer_initialization():
    """Test DeadWeightDynamicBiasNN initialization"""
    layer = deadweight_dynamic.DeadWeightDynamicBiasNN(
        input_size=10, hidden_size=20, spectral_radius=0.95, connectivity=0.8, seed=42
    )

    # Verify parameters
    assert layer.input_size == 10
    assert layer.hidden_size == 20
    assert layer.weight.shape == (20, 10)
    assert layer.velocity_weight.shape == (10, 1)
    assert layer.velocity_bias.shape == (1,)

    # Verify weights are frozen
    assert not layer.weight.requires_grad
    assert layer.velocity_weight.requires_grad
    assert layer.velocity_bias.requires_grad


def test_forward_pass():
    """Test forward pass with sample input"""
    layer = deadweight_dynamic.DeadWeightDynamicBiasNN(5, 10)
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
    layer = deadweight_dynamic.DeadWeightDynamicBiasNN(5, 10)
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
    layer = deadweight_dynamic.DeadWeightDynamicBiasNN(5, 10)
    x = torch.randn(3, 5)

    # First pass initializes bias
    layer(x)
    bias1 = layer.current_bias.clone()

    # Reset bias
    layer.reset_bias()
    bias_reset = layer.current_bias

    # Verify bias was properly reset
    assert not torch.equal(bias1, bias_reset)


def test_weight_initialization():
    """Test reservoir weight initialization"""
    layer = deadweight_dynamic.DeadWeightDynamicBiasNN(
        10,
        10,  # square matrix for eigenvalue test
        spectral_radius=0.9,
        connectivity=0.5,
        seed=42,
    )

    # Compute eigenvalues of weight matrix
    eigenvalues = np.linalg.eigvals(layer.weight.detach().numpy())
    max_eigen = max(abs(eigenvalues))

    # Verify spectral radius
    assert pytest.approx(max_eigen, abs=1e-2) == 0.9

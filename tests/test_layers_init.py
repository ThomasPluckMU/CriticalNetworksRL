import pytest
from criticalnets.layers import __init__ as layers_init


def test_layers_init_imports():
    """Test that layers module imports correctly"""
    assert layers_init is not None

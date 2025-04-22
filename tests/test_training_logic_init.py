import pytest
from criticalnets.training.logic import __init__ as logic_init


def test_logic_init_imports():
    """Test that training.logic module imports correctly"""
    assert logic_init is not None

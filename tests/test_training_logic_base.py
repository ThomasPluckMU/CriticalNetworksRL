import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch
from criticalnets.training.logic.base import TrainingLogic


class ConcreteTestLogic(TrainingLogic):
    """Concrete implementation for testing"""

    def run_episode(self, env, agent, memory, episode_idx):
        return 0.0, {}  # Dummy implementation


class TestTrainingLogic:
    def test_configure_optimizer(self):
        """Test default optimizer configuration"""
        logic = ConcreteTestLogic()
        network = nn.Linear(10, 2)

        optimizer = logic.configure_optimizer(network)

        assert isinstance(optimizer, optim.SGD)
        assert optimizer.defaults["lr"] == 0.001
        assert len(optimizer.param_groups[0]["params"]) == 2  # weight + bias

    def test_on_checkpoint(self):
        """Test checkpoint callback can be overridden"""

        class TestLogic(ConcreteTestLogic):
            def on_checkpoint(self, episode):
                self.called = True
                self.episode = episode

        logic = TestLogic()
        logic.on_checkpoint(42)

        assert hasattr(logic, "called")
        assert logic.called is True
        assert logic.episode == 42

    def test_run_episode_abstract(self):
        """Test run_episode is abstract and must be implemented"""
        with pytest.raises(TypeError):

            class InvalidLogic(TrainingLogic):
                pass  # Doesn't implement run_episode

            InvalidLogic()  # Should raise TypeError

    @patch("torch.optim.SGD")
    def test_configure_optimizer_params(self, mock_sgd):
        """Test optimizer is created with network parameters"""
        logic = ConcreteTestLogic()
        network = Mock()
        network.parameters.return_value = ["param1", "param2"]

        logic.configure_optimizer(network)

        mock_sgd.assert_called_once_with(["param1", "param2"], lr=0.001)

    def test_configure_optimizer_empty_params(self):
        """Test optimizer with network that returns empty parameters"""
        logic = ConcreteTestLogic()
        network = Mock()
        network.parameters.return_value = []

        with patch("builtins.list") as mock_list:
            mock_list.return_value = []
            optimizer = logic.configure_optimizer(network)
            assert optimizer is None
            mock_list.assert_called_once_with([])

    def test_configure_optimizer_with_convnet(self):
        """Test optimizer works with convolutional network"""
        logic = ConcreteTestLogic()
        network = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 10),
        )

        optimizer = logic.configure_optimizer(network)
        assert isinstance(optimizer, optim.SGD)
        assert (
            len(optimizer.param_groups[0]["params"]) == 4
        )  # conv + bias, linear + bias

    def test_configure_optimizer_empty_network(self):
        """Test optimizer with network that has no parameters"""
        logic = ConcreteTestLogic()
        network = nn.Sequential()  # Empty network

        # Verify network really has no parameters
        assert len(list(network.parameters())) == 0

        optimizer = logic.configure_optimizer(network)
        assert optimizer is None

    def test_on_checkpoint_no_override(self):
        """Test default on_checkpoint does nothing"""
        logic = ConcreteTestLogic()
        # Should not raise any exceptions
        logic.on_checkpoint(42)

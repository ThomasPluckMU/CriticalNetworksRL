import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from criticalnets.agents.gated_atari_udqn import GatedAtariUDQN

class TestGatedAtariUDQN:
    @pytest.fixture
    def agent(self):
        config = {
            'frame_stack': 4,
            'device': 'cpu'
        }
        
        # Create mock convolutional layers
        mock_conv1 = MagicMock()
        mock_conv2 = MagicMock()
        mock_conv3 = MagicMock()
        
        # Mock the forward passes to return properly shaped and flattened tensors
        mock_conv1.return_value = torch.rand(1, 1920)  # Flattened conv output
        mock_conv2.return_value = torch.rand(1, 1920)  # Flattened conv output  
        mock_conv3.return_value = torch.rand(1, 1920)  # Flattened conv output
        
        # Create agent with patched convolutional layers
        with patch('criticalnets.agents.gated_atari_udqn.GatedDynamicBiasCNN', 
                 side_effect=[mock_conv1, mock_conv2, mock_conv3]):
            agent = GatedAtariUDQN(config, action_space=6)
            # Replace the actual conv layers with our mocks
            agent.conv1 = mock_conv1
            agent.conv2 = mock_conv2
            agent.conv3 = mock_conv3
            return agent

    def test_initialization(self, agent):
        assert agent.frame_stack == 4
        assert agent.fc.out_features == 512
        assert agent.head.out_features == 6

    def test_forward_pass(self, agent):
        x = torch.rand(1, 4, 84, 84)  # batch, channels, height, width
        output = agent(x)
        assert output.shape == (1, 6)  # batch, actions
        agent.conv1.assert_called_once()
        agent.conv2.assert_called_once()
        agent.conv3.assert_called_once()

    @patch('torch.sigmoid')
    def test_activation_functions(self, mock_sigmoid, agent):
        mock_sigmoid.return_value = torch.ones(1, 512)
        x = torch.rand(1, 4, 84, 84)
        output = agent(x)
        mock_sigmoid.assert_called_once()
        assert output.shape == (1, 6)

    def test_device_handling(self):
        config = {
            'frame_stack': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        agent = GatedAtariUDQN(config, action_space=6)
        assert str(next(agent.parameters()).device) == config['device']

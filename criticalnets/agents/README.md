# Agents Module

## Available Agents

### Value-Based Agents
- `CriticalAtariDQN`: DQN with critical network regularization
  - 3 CNN layers + FC
  - Tanh activations
  - Jacobian norm regularization
- `DynamicAtariUDQN`: Uncertainty-aware DQN
- `GatedAtariUDQN`: Gated version with uncertainty
- `StandardDQN`: Basic Deep Q-Network

### Policy-Based Agents
- `PongA2CAgent`: Advantage Actor-Critic for Pong
- `PPOAgent`: Proximal Policy Optimization
- `CriticalGatedAgent`: Gated policy with critical features

## BaseAtariAgent Features
```mermaid
classDiagram
    class BaseAtariAgent {
        +activations: dict
        +gradients: dict
        +loss: float
        +register_activation_probes()
        +set_loss()
        +get_metrics()
        +get_param_schema()
        +validate_config()
    }
```

## CriticalAtariDQN Architecture
```mermaid
graph TD
    Input[84x84x4 Frame] --> Conv1[8x8 Conv, stride 4]
    Conv1 --> Tanh1[Tanh]
    Tanh1 --> Conv2[4x4 Conv, stride 2] 
    Conv2 --> Tanh2[Tanh]
    Tanh2 --> Conv3[3x3 Conv, stride 1]
    Conv3 --> Tanh3[Tanh]
    Tanh3 --> Flatten
    Flatten --> FC[512-unit FC]
    FC --> Sigmoid
    Sigmoid --> Head[Action Q-values]
```

## Usage Examples

### Basic Usage
```python
from criticalnets.agents import get_agent_class

agent_class = get_agent_class('CriticalAtariDQN')
agent = agent_class(config={
    'device': 'cuda',
    'frame_stack': 4,
    'reg_strength': 1.0
}, action_space=6)

q_values = agent(state_tensor)  # Get action values
```

### Monitoring
```python
# Track specific layer activations
agent.register_activation_probes(['conv1', 'conv2'])

# After forward pass:
print(agent.activations['conv1'].shape)  # Activation tensor
print(agent.gradients['conv1.weight'])  # Parameter gradients
```


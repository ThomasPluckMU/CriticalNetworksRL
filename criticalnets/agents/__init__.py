from typing import Dict, Type
import torch
import torch.nn as nn
from importlib import import_module
from pathlib import Path

class BaseAtariAgent(nn.Module):
    """Base class that all Atari agents must implement"""
    def __init__(self, config: Dict, action_space: int):
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.activations = {}  # Stores layer_name -> output tensors
        self.gradients = {}    # Stores param_name -> gradient tensors 
        self.loss = None       # Stores latest loss value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Agents must implement forward()")

    def register_activation_probes(self, layer_names: list):
        """Add forward hooks to specified layers"""
        for name, module in self.named_modules():
            if name in layer_names:
                module.register_forward_hook(self._save_activation(name))

    def _save_activation(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def set_loss(self, loss: torch.Tensor):
        """Capture loss and register gradient hooks"""
        self.loss = loss.detach()
        self.loss.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.gradients[name] = param.grad.detach()

    def get_metrics(self) -> dict:
        """Return copy of current metrics"""
        return {
            'activations': {k: v.clone() for k,v in self.activations.items()},
            'gradients': {k: v.clone() for k,v in self.gradients.items()},
            'loss': self.loss.clone() if self.loss is not None else None
        }

# Auto-discover and register agents
AGENT_REGISTRY = {}

agents_dir = Path(__file__).parent
for file in agents_dir.glob("*.py"):
    if file.name.startswith("_"):
        continue
    module = import_module(f"criticalnets.agents.{file.stem}")
    for name, cls in vars(module).items():
        if isinstance(cls, type) and issubclass(cls, BaseAtariAgent) and cls != BaseAtariAgent:
            AGENT_REGISTRY[name] = cls

def get_agent_class(name: str) -> Type[BaseAtariAgent]:
    """Get agent class by name from registry"""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available agents: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name]

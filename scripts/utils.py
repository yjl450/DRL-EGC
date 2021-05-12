import torch
from torch import nn
import pfrl

# For Rainbow
class DistributionalDuelingHead(nn.Module):
    """Head module for defining a distributional dueling network.

    This module expects a (batch_size, in_size)-shaped `torch.Tensor` as input
    and returns `pfrl.action_value.DistributionalDiscreteActionValue`.

    Args:
        in_size (int): Input size.
        n_actions (int): Number of actions.
        n_atoms (int): Number of atoms.
        v_min (float): Minimum value represented by atoms.
        v_max (float): Maximum value represented by atoms.
    """

    def __init__(self, in_size, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        assert in_size % 2 == 0
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            "z_values", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float)
        )
        self.a_stream = nn.Linear(in_size // 2, n_actions * n_atoms)
        self.v_stream = nn.Linear(in_size // 2, n_atoms)

    def forward(self, h):
        h_a, h_v = torch.chunk(h, 2, dim=1)
        a_logits = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        a_logits = a_logits - a_logits.mean(dim=1, keepdim=True)
        v_logits = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        probs = nn.functional.softmax(a_logits + v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)
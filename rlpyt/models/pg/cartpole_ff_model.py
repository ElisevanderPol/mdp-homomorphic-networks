"""
Adaptation of the Atari model class to suit CartPole-v1 and enable the use of
basis networks.
"""

import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel

from symmetrizer.nn import BasisCartpoleNetworkWrapper, BasisCartpoleLayer, \
    BasisLinear


class CartpoleFfModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[64, 64],
            basis=None,
            gain_type="xavier",
            out=None,
            ):
        super().__init__()

        input_size = image_shape[0]
        # Main body
        self.head = MlpModel(input_size, fc_sizes)
        # Policy output
        self.pi = torch.nn.Linear(fc_sizes[-1], output_size)
        # Value output
        self.value = torch.nn.Linear(fc_sizes[-1], 1)

        if gain_type == "xavier":
            self.head.apply(weight_init)
            self.pi.apply(weight_init)
            self.value.apply(weight_init)


    def forward(self, in_state, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        state = in_state.type(torch.float)  # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, state_shape = infer_leading_dims(state, 1)

        base = self.head(state.view(T * B, -1))
        pi = F.softmax(self.pi(base), dim=-1)
        v = self.value(base).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


def weight_init(layer):
    """
    Xavier initialization
    """
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)


class CartpoleBasisModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[45, 45],
            basis="equivariant",
            gain_type="default",
            ):
        super(CartpoleBasisModel, self).__init__()
        input_size = image_shape[0]
        input_size = 1


        self.head = BasisCartpoleNetworkWrapper(input_size, fc_sizes,
                                             gain_type=gain_type,
                                             basis=basis)
        self.pi = BasisCartpoleLayer(fc_sizes[-1], 1,
                                         gain_type=gain_type,
                                         basis=basis)
        self.value = BasisCartpoleLayer(fc_sizes[-1], 1,
                                            gain_type=gain_type,
                                            basis=basis, out="invariant")

    def forward(self, in_state, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        state = in_state.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, state_shape = infer_leading_dims(state, 1)

        base = self.head(state.view(T * B, -1))
        pi = F.softmax(self.pi(base), dim=-1).squeeze()
        v = self.value(base).squeeze()

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v



import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class CartpoleDqnModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[64, 64],
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        super().__init__()
        self.dueling = dueling
        input_size = image_shape[0]
        # self.mlp = MlpModel(input_size, fc_sizes, output_size)
        if dueling:
            self.head = DuelingHeadModel(input_size, fc_sizes, output_size)
        else:
            self.head = MlpModel(input_size, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 1)

        # conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        q = self.head(img.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

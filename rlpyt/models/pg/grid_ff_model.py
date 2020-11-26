import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from symmetrizer.nn.grid_networks import BasisGridNetwork



class GridFfModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=[512],
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            basis=None,
            ):
        super().__init__()
        use_avgpool=True
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 5],
            strides=strides or [4, 2],
            paddings=paddings or [0, 0],
            use_avgpool=use_avgpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.pi = torch.nn.Linear(self.conv.output_size, output_size)
        self.value = torch.nn.Linear(self.conv.output_size, 1)


    def forward(self, image, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v



class GridBasisModel(torch.nn.Module):

    def __init__(self, image_shape, output_size, fc_sizes=[512],
                 channels=[16, 32], kernel_sizes=[8, 5], strides=[4, 2],
                 paddings=[0, 0], basis="equivariant", gain_type="he"):
        super().__init__()
        self.conv = BasisGridNetwork(1, channels=channels,
                                     filters=kernel_sizes, strides=strides,
                                     paddings=paddings, hidden_sizes=fc_sizes,
                                     gain_type=gain_type, basis=basis)


    def forward(self, image, prev_action, prev_reward, imshow=False):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(fc_out[0], dim=-1).squeeze(-2)
        v = fc_out[1].squeeze(-1).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v



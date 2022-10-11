import numpy as np
import os
# os.environ['CUDA_PATH']='/home/aoli/tools/cuda10.0'
import torch
from pykeops.torch import LazyTensor
import torch.nn as nn
from math import sqrt

def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices

def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x

def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv

#  Fast tangent convolution layer ===============================================
class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases. 
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()

class dMaSIFConv(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, radius=1.0, hidden_units=None, cheap=False
    ):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features
        O = out_channels is the dimension of the output features
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)


        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:

          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)

          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 1 linear layer if "cheap" is True;
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.

          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)


        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.

        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
            cheap (bool, optional): shall we use a 1-layer deep Filter,
                instead of a 2-layer deep MLP? Defaults to False.
        """

        super(dMaSIFConv, self).__init__()

        self.Input = in_channels
        self.Output = out_channels
        self.Radius = radius
        self.Hidden = self.Output if hidden_units is None else hidden_units
        self.Cuts = 8  # Number of hidden units for the 3D MLP Filter.
        self.cheap = cheap

        # For performance reasons, we cut our "hidden" vectors
        # in n_heads "independent heads" of dimension 8.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "Hidden" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if self.Hidden < self.heads_dim:
            self.heads_dim = self.Hidden

        if self.Hidden % self.heads_dim != 0:
            raise ValueError(f"The dimension of the hidden units ({self.Hidden})"\
                    + f"should be a multiple of the heads dimension ({self.heads_dim}).")
        else:
            self.n_heads = self.Hidden // self.heads_dim


        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Hidden, self.Hidden),  # (H, H) + (H,)
            nn.LeakyReLU(negative_slope=0.2),)  #  (H,)
        self.norm_in = nn.GroupNorm(4, self.Hidden)


        # 3D convolution filters, encoded as an MLP:
        if cheap:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Hidden), nn.ReLU()  # (H, 3) + (H,)
            )  # KeOps does not support well LeakyReLu
        else:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
                nn.ReLU(),  # KeOps does not support well LeakyReLu
                nn.Linear(self.Cuts, self.Hidden),
            )  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
        )  #  (O,)

        self.norm_out = nn.GroupNorm(4, self.Output)
        # self.norm_out = nn.LayerNorm(self.Output)
        # self.norm_out = nn.Identity()

        # Custom initialization for the MLP convolution filters:
        # we get interesting piecewise affine cuts on a normalized neighborhood.
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()

            if not cheap:
                nn.init.uniform_(
                    self.conv[2].weight,
                    a=-1 / np.sqrt(self.Cuts),
                    b=1 / np.sqrt(self.Cuts),
                )
                nn.init.normal_(self.conv[2].bias)
                self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()


    def forward(self, points, nuv, features, ranges=None):
        """Performs a quasi-geodesic interaction step.

        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)

        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.

        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): low-level format
                to support batch processing, as described in the KeOps documentation.
                In practice, this will be built by a higher-level object
                to encode the relevant "batch vectors" in a way that is convenient
                for the KeOps CUDA engine. Defaults to None.

        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # 1. Transform the input features: -------------------------------------
        features = self.net_in(features)  # (N, I) -> (N, H)
        features = features.transpose(1, 0)[None, :, :]  # (1,H,N)
        features = self.norm_in(features)
        features = features[0].transpose(1, 0).contiguous()  # (1, H, N) -> (N, H)

        # 2. Compute the local "shape contexts": -------------------------------

        # 2.a Normalize the kernel radius:
        points = points / (sqrt(2.0) * self.Radius)  # (N, 3)

        # 2.b Encode the variables as KeOps LazyTensors

        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)

        # WARNING - Here, we assume that the normals are fixed:
        normals = (
            nuv[:, 0, :].contiguous().detach()
        )  # (N, 3) - remove the .detach() if needed

        # Local bases:
        nuv_i = LazyTensor(nuv.view(-1, 1, 9))  # (N, 1, 9)
        # Normals:
        n_i = nuv_i[:3]  # (N, 1, 3)

        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)

        # To avoid register spilling when using large embeddings, we perform our KeOps reduction
        # over the vector of length "self.Hidden = self.n_heads * self.heads_dim"
        # as self.n_heads reduction over vectors of length self.heads_dim (= "Hd" in the comments).
        head_out_features = []
        for head in range(self.n_heads):

            # Extract a slice of width Hd from the feature array
            head_start = head * self.heads_dim
            head_end = head_start + self.heads_dim
            head_features = features[:, head_start:head_end].contiguous()  # (N, H) -> (N, Hd)

            # Features:
            f_j = LazyTensor(head_features[None, :, :])  # (1, N, Hd)

            # Convolution parameters:
            if self.cheap:
                # Extract a slice of Hd lines: (H, 3) -> (Hd, 3)
                A = self.conv[0].weight[head_start:head_end, :].contiguous()  
                # Extract a slice of Hd coefficients: (H,) -> (Hd,)
                B = self.conv[0].bias[head_start:head_end].contiguous() 
                AB = torch.cat((A, B[:, None]), dim=1)  # (Hd, 4)
                ab = LazyTensor(AB.view(1, 1, -1))  # (1, 1, Hd*4)
            else:
                A_1, B_1 = self.conv[0].weight, self.conv[0].bias  # (C, 3), (C,)
                # Extract a slice of Hd lines: (H, C) -> (Hd, C)
                A_2 = self.conv[2].weight[head_start:head_end, :].contiguous()
                # Extract a slice of Hd coefficients: (H,) -> (Hd,)
                B_2 = self.conv[2].bias[head_start:head_end].contiguous()
                a_1 = LazyTensor(A_1.view(1, 1, -1))  # (1, 1, C*3)
                b_1 = LazyTensor(B_1.view(1, 1, -1))  # (1, 1, C)
                a_2 = LazyTensor(A_2.view(1, 1, -1))  # (1, 1, Hd*C)
                b_2 = LazyTensor(B_2.view(1, 1, -1))  # (1, 1, Hd)

            # 2.c Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij = (-d2_ij).exp()  # (N, N, 1)

            # 2.d Local MLP:
            # Local coordinates:
            # xij = x_j - x_i
            # # xij = xij / xij.norm2()
            X_ij =  nuv_i.matvecmult(x_j - x_i)  # (N, N, 3)
            # X_ij = nuv_i.matvecmult(x_j - x_i)  # (N, N, 9) "@" (N, N, 3) = (N, N, 3)
            # MLP:
            if self.cheap:
                X_ij = ab.matvecmult(
                    X_ij.concat(LazyTensor(1))
                )  # (N, N, Hd*4) @ (N, N, 3+1) = (N, N, Hd)
                X_ij = X_ij.relu()  # (N, N, Hd)
            else:
                X_ij = a_1.matvecmult(X_ij) + b_1  # (N, N, C)
                X_ij = X_ij.relu()  # (N, N, C)
                X_ij = a_2.matvecmult(X_ij) + b_2  # (N, N, Hd)
                X_ij = X_ij.relu()
            # 2.e Actual computation:

            F_ij =  window_ij *X_ij * f_j  # (N, N, Hd)
            F_ij.ranges = ranges  # Support for batches and/or block-sparsity

            head_out_features.append(ContiguousBackward().apply(F_ij.sum(dim=1)))  # (N, Hd)

        # Concatenate the result of our n_heads "attention heads":
        features = torch.cat(head_out_features, dim=1)  # n_heads * (N, Hd) -> (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.transpose(1, 0)[None, :, :]  # (1,O,N)
        features = self.norm_out(features)
        features = features[0].transpose(1, 0).contiguous()

        return features
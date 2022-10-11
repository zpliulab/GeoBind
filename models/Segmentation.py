import torch
import torch.nn.functional as F
import torch.nn as nn
from pykeops.torch import LazyTensor

from .geometry_processing import dMaSIFConv, tangent_vectors
from .geometry_processing import diagonal_ranges

#Origined from dMaSIF.
class Conv(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels, n_layers, radius=12.0):
        super(Conv, self).__init__()

        self.name = "conv"
        self.radius = radius
        self.I, self.O = in_channels, out_channels

        self.layers = nn.ModuleList(
            [dMaSIFConv(self.I, self.O, radius, self.O)]
            + [dMaSIFConv(self.O, self.O, radius, self.O) for i in range(n_layers - 1)]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for i in range(n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for i in range(n_layers - 1)]
        )

    def forward(self, features):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        points, nuv, ranges = self.points, self.nuv, self.ranges
        x = features
        for i, layer in enumerate(self.layers):
            x_i = layer(points, nuv, x, ranges)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i
        return x

    def load_mesh(self, xyz, triangles=None, normals=None, nuv=None, weights=None, batch=None):
        """Loads the geometry of a triangle mesh.

        Input arguments:
        - xyz, a point cloud encoded as an (N, 3) Tensor.
        - triangles, a connectivity matrix encoded as an (N, 3) integer tensor.
        - weights, importance weights for the orientation estimation, encoded as an (N, 1) Tensor.
        - radius, the scale used to estimate the local normals.
        - a batch vector, following PyTorch_Geometric's conventions.

        The routine updates the model attributes:
        - points, i.e. the point cloud itself,
        - nuv, a local oriented basis in R^3 for every point,
        - ranges, custom KeOps syntax to implement batch processing.
        """

        # 1. Save the vertices for later use in the convolutions ---------------
        self.points = xyz
        self.batch = batch
        self.ranges = diagonal_ranges(
            batch
        )  # KeOps support for heterogeneous batch processing
        self.triangles = triangles
        self.normals = normals
        if nuv ==None and weights!=None:
            points=xyz / self.radius
            tangent_bases=tangent_vectors(normals)  # Tangent basis (N, 2, 3)
            # 3. Steer the tangent bases according to the gradient of "weights" ----

            # 3.a) Encoding as KeOps LazyTensors:
            # Orientation scores:
            weights_j=LazyTensor(weights.view(1, -1, 1))  # (1, N, 1)
            # Vertices:
            x_i=LazyTensor(points[:, None, :])  # (N, 1, 3)
            x_j=LazyTensor(points[None, :, :])  # (1, N, 3)
            # Normals:
            n_i=LazyTensor(normals[:, None, :])  # (N, 1, 3)
            n_j=LazyTensor(normals[None, :, :])  # (1, N, 3)
            # Tangent basis:
            uv_i=LazyTensor(tangent_bases.view(-1, 1, 6))  # (N, 1, 6)

            # 3.b) Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            rho2_ij=((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij=(-rho2_ij).exp()  # (N, N, 1)

            # 3.c) Coordinates in the (u, v) basis - not oriented yet:
            X_ij=uv_i.matvecmult(x_j - x_i)  # (N, N, 2)

            # 3.d) Local average in the tangent plane:
            orientation_weight_ij=window_ij * weights_j  # (N, N, 1) （1, N, 1）-> (N,N)
            orientation_vector_ij=orientation_weight_ij * X_ij  # (N, N, 2)

            # Support for heterogeneous batch processing:
            orientation_vector_ij.ranges=self.ranges  # Block-diagonal sparsity mask

            orientation_vector_i=orientation_vector_ij.sum(dim=1)  # (N, 2)
            orientation_vector_i=(
                    orientation_vector_i + 1e-5
            )  # Just in case someone's alone...

            # 3.e) Normalize stuff:
            orientation_vector_i=F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
            ex_i, ey_i=(
                orientation_vector_i[:, 0][:, None],
                orientation_vector_i[:, 1][:, None],
            )  # (N,1)

            # 3.f) Re-orient the (u,v) basis:
            uv_i=tangent_bases  # (N, 2, 3)
            u_i, v_i=uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
            tangent_bases=torch.cat(
                (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=1
            ).contiguous()  # (N, 6)

            # 4. Store the local 3D frame as an attribute --------------------------
            self.nuv=torch.cat(
                (normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)), dim=1
            )
            pass
        else:
            self.nuv = nuv

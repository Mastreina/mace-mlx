"""Radial basis functions and cutoff envelopes for MACE-MLX.

Implements BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis,
RadialEmbeddingBlock, and make_radial_mlp — all as MLX nn.Module subclasses.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from mace_mlx.utils import scatter_sum


def _polynomial_envelope(u: mx.array, p: int) -> mx.array:
    """Smooth polynomial envelope on u = r/r_max: 1 at u=0, 0 at u>=1.

    envelope(u) = 1 - C1*u^p + C2*u^(p+1) - C3*u^(p+2)
    where C1=(p+1)(p+2)/2, C2=p(p+2), C3=p(p+1)/2
    """
    pf = float(p)
    u_p = u ** p
    u_p1 = u_p * u
    u_p2 = u_p1 * u
    envelope = (
        1.0
        - ((pf + 1.0) * (pf + 2.0) / 2.0) * u_p
        + pf * (pf + 2.0) * u_p1
        - (pf * (pf + 1.0) / 2.0) * u_p2
    )
    return envelope * (u < 1.0)


class BesselBasis(nn.Module):
    """Radial Bessel basis functions (Eq. 7 in MACE paper).

    B_n(r) = sqrt(2/r_max) * sin(n*pi*r / r_max) / r
    """

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        self._prefactor = math.sqrt(2.0 / r_max)

        bessel_weights = (
            mx.array([float(i) for i in range(1, num_basis + 1)]) * math.pi / r_max
        )

        if trainable:
            self.bessel_weights = bessel_weights
        else:
            self.bessel_weights = bessel_weights
            self.freeze(keys=["bessel_weights"])

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N,) or (N, 1)
        if x.ndim == 1:
            x = x[:, None]  # (N, 1)
        bw = mx.stop_gradient(self.bessel_weights)
        numerator = mx.sin(bw * x)  # (N, num_basis)
        # Handle x=0: use L'Hopital limit sin(w*x)/x -> w as x->0
        safe_x = mx.where(x == 0.0, mx.ones_like(x), x)
        result = self._prefactor * numerator / safe_x
        # At x=0, sin(w*0)/0 -> w (by L'Hopital), so result = prefactor * w
        at_zero = self._prefactor * bw
        result = mx.where(x == 0.0, at_zero, result)
        return result  # (N, num_basis)

    def __repr__(self) -> str:
        return (
            f"BesselBasis(r_max={self.r_max}, num_basis={self.num_basis})"
        )


class GaussianBasis(nn.Module):
    """Gaussian radial basis functions.

    G_n(r) = exp(coeff * (r - center_n)^2)
    where coeff = -0.5 / (r_max / (num_basis - 1))^2
    """

    def __init__(self, r_max: float, num_basis: int = 128):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis

        self.centers = mx.linspace(0.0, r_max, num_basis)
        spacing = r_max / (num_basis - 1) if num_basis > 1 else r_max
        self._coeff = -0.5 / (spacing ** 2)

        self.freeze(keys=["centers"])

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N,) or (N, 1)
        if x.ndim == 1:
            x = x[:, None]  # (N, 1)
        diff = x - self.centers  # (N, num_basis)
        return mx.exp(self._coeff * (diff ** 2))

    def __repr__(self) -> str:
        return f"GaussianBasis(r_max={self.r_max}, num_basis={self.num_basis})"


class PolynomialCutoff(nn.Module):
    """Smooth polynomial envelope: 1 at r=0, 0 at r=r_max.

    envelope(x) = 1 - C1 * u^p + C2 * u^(p+1) - C3 * u^(p+2)
    where u = x/r_max, C1=(p+1)(p+2)/2, C2=p(p+2), C3=p(p+1)/2
    """

    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.r_max = r_max
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        return _polynomial_envelope(x / self.r_max, self.p)

    def __repr__(self) -> str:
        return f"PolynomialCutoff(r_max={self.r_max}, p={self.p})"


class ZBLBasis(nn.Module):
    """Ziegler-Biersack-Littmark (ZBL) pair repulsion with polynomial cutoff.

    Matches PyTorch MACE's ZBLBasis: the cutoff envelope uses an edge-wise
    r_max = covalent_radii[Z_u] + covalent_radii[Z_v], so the repulsion only
    acts at bonding distances and below. Used by the mpa-0/0b/0b2/0b3 model
    family (pair_repulsion=True checkpoints).

    All quantities are computed in float32 (part of the geometric front-end).
    """

    def __init__(
        self,
        p: int = 6,
        a_exp: float = 0.300,
        a_prefactor: float = 0.4543,
    ):
        super().__init__()
        import ase.data

        self._p = int(p)
        self._a_exp = float(a_exp)
        self._a_prefactor = float(a_prefactor)
        # Universal screening function coefficients (Ziegler et al.)
        self._c = (0.1818, 0.5099, 0.2802, 0.02817)
        self._covalent_radii = mx.array(
            ase.data.covalent_radii.tolist(), dtype=mx.float32
        )

    def __call__(
        self,
        lengths: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        atomic_numbers: mx.array,
    ) -> mx.array:
        """
        Args:
            lengths: (num_edges, 1) interatomic distances (float32)
            node_attrs: (num_atoms, num_elements) one-hot encoding
            edge_index: (2, num_edges) [sender, receiver]
            atomic_numbers: (num_elements,) atomic number per element index

        Returns:
            (num_atoms,) per-node ZBL repulsion energy (float32)
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        node_z = atomic_numbers[mx.argmax(node_attrs, axis=1)]  # (num_atoms,)
        Z_u = node_z[sender].astype(mx.int32)
        Z_v = node_z[receiver].astype(mx.int32)
        Zu_f = Z_u.astype(mx.float32)
        Zv_f = Z_v.astype(mx.float32)

        r = lengths.squeeze(-1).astype(mx.float32)  # (num_edges,)
        a = (
            self._a_prefactor
            * 0.529
            / (Zu_f ** self._a_exp + Zv_f ** self._a_exp)
        )
        r_over_a = r / a
        c0, c1, c2, c3 = self._c
        phi = (
            c0 * mx.exp(-3.2 * r_over_a)
            + c1 * mx.exp(-0.9423 * r_over_a)
            + c2 * mx.exp(-0.4028 * r_over_a)
            + c3 * mx.exp(-0.2016 * r_over_a)
        )
        # e^2 / (4*pi*eps0) = 14.3996 eV*Angstrom
        v_edges = (14.3996 * Zu_f * Zv_f) / r * phi
        r_max = self._covalent_radii[Z_u] + self._covalent_radii[Z_v]
        envelope = _polynomial_envelope(r / r_max, self._p)
        v_edges = 0.5 * v_edges * envelope
        v_nodes = scatter_sum(v_edges[:, None], receiver, node_attrs.shape[0])
        return v_nodes.squeeze(-1)

    def __repr__(self) -> str:
        return f"ZBLBasis(p={self._p}, a_exp={self._a_exp}, a_prefactor={self._a_prefactor})"


class AgnesiTransform(nn.Module):
    """Agnesi radial distance transform (element-pair dependent).

    Transforms interatomic distances based on covalent radii of the
    atom pair.  Used by the MACE 0b/0b2/mpa-0 model family.

    Reference: ACEpotentials.jl, JCP 2023 (doi:10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        covalent_radii: list[float] | None = None,
    ):
        super().__init__()
        self._q = q
        self._p = p
        self._a = a
        if covalent_radii is None:
            import ase.data
            covalent_radii = ase.data.covalent_radii.tolist()
        self._covalent_radii = mx.array(covalent_radii, dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        node_attrs: mx.array,
        edge_index: mx.array,
        atomic_numbers: mx.array,
    ) -> mx.array:
        """Transform distances using element-pair covalent radii.

        Args:
            x: (num_edges, 1) interatomic distances.
            node_attrs: (num_atoms, num_elements) one-hot encoding.
            edge_index: (2, num_edges) [sender, receiver].
            atomic_numbers: (num_elements,) atomic numbers for each element index.

        Returns:
            (num_edges, 1) transformed distances.
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        # Map one-hot -> element index -> atomic number -> covalent radius
        node_z = atomic_numbers[mx.argmax(node_attrs, axis=1)]  # (num_atoms,)
        Z_u = node_z[sender].astype(mx.int32)  # (num_edges,)
        Z_v = node_z[receiver].astype(mx.int32)  # (num_edges,)
        r_0 = 0.5 * (self._covalent_radii[Z_u] + self._covalent_radii[Z_v])
        r_0 = r_0[:, None]  # (num_edges, 1)
        # Clamp: the fractional powers below have infinite gradients at 0,
        # which would NaN the backward pass for zero-length edges.
        r_over_r_0 = mx.maximum(x / r_0, 1e-12)
        q, p, a = self._q, self._p, self._a
        return 1.0 / (1.0 + a * mx.power(r_over_r_0, q) / (1.0 + mx.power(r_over_r_0, q - p)))


class RadialEmbeddingBlock(nn.Module):
    """Combines BesselBasis with PolynomialCutoff and optional distance transform.

    For original models: output = bessel(r) * cutoff(r)
    For 0b/0b2/mpa-0 models: output = bessel(transform(r)) * cutoff(r)

    When apply_cutoff=False (mh-1 family), returns (basis, cutoff) separately.
    The cutoff is then applied in the interaction block to TP weights and density.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 6,
        distance_transform: dict | None = None,
        apply_cutoff: bool = True,
    ):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self._has_transform = distance_transform is not None
        self._apply_cutoff = apply_cutoff
        if self._has_transform:
            self.distance_transform = AgnesiTransform(**distance_transform)

    def __call__(
        self,
        edge_lengths: mx.array,
        node_attrs: mx.array | None = None,
        edge_index: mx.array | None = None,
        atomic_numbers: mx.array | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        # edge_lengths: (num_edges,) or (num_edges, 1)
        if edge_lengths.ndim == 2:
            edge_lengths_1d = edge_lengths.squeeze(-1)
        else:
            edge_lengths_1d = edge_lengths

        cutoff = self.cutoff_fn(edge_lengths_1d)  # (num_edges,)

        # Apply distance transform if present (0b/0b2/mpa-0 models)
        if self._has_transform:
            el_2d = edge_lengths_1d[:, None] if edge_lengths_1d.ndim == 1 else edge_lengths
            transformed = self.distance_transform(
                el_2d, node_attrs, edge_index, atomic_numbers
            )
            basis = self.bessel_fn(transformed.squeeze(-1))
        else:
            basis = self.bessel_fn(edge_lengths_1d)

        if not self._apply_cutoff:
            # Return basis and cutoff separately; cutoff applied in interaction block
            return basis, cutoff[:, None]  # (num_edges, num_bessel), (num_edges, 1)

        return basis * cutoff[:, None]  # (num_edges, num_bessel)

    def __repr__(self) -> str:
        return (
            f"RadialEmbeddingBlock(bessel={self.bessel_fn}, cutoff={self.cutoff_fn})"
        )


def make_radial_mlp(channels_list: Sequence[int]) -> nn.Module:
    """Build a radial MLP: Linear -> SiLU -> Linear -> SiLU -> ... -> Linear.

    The last layer is a plain Linear (no activation).
    No bias is used (matching e3nn's FullyConnectedNet convention).
    For inference with pre-trained weights, no LayerNorm is needed.
    """
    layers: list[nn.Module] = []
    for i in range(len(channels_list) - 1):
        layers.append(nn.Linear(channels_list[i], channels_list[i + 1], bias=False))
        if i < len(channels_list) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def make_radial_mlp_with_layernorm(channels_list: Sequence[int]) -> nn.Module:
    """Build a RadialMLP: Linear(bias) -> LayerNorm -> SiLU -> ... -> Linear(bias).

    Used by RealAgnosticResidualNonLinearInteractionBlock (mh-1 family).
    Unlike FullyConnectedNet, this uses nn.Linear with bias and LayerNorm.
    The last layer is a plain Linear (no activation, no LayerNorm).
    """
    layers: list[nn.Module] = []
    for i in range(len(channels_list) - 1):
        layers.append(nn.Linear(channels_list[i], channels_list[i + 1], bias=True))
        if i < len(channels_list) - 2:
            layers.append(nn.LayerNorm(channels_list[i + 1]))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)

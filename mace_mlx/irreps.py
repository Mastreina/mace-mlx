"""Irreducible representations of O(3) for equivariant neural networks.

Pure Python implementation compatible with e3nn's Irreps string format.
No framework dependency — this is purely a metadata/bookkeeping module.
"""

from __future__ import annotations

import re
from collections import namedtuple
from functools import cached_property
from typing import Iterator, Sequence, Union


class Irrep:
    """Single irreducible representation of O(3), characterized by (l, p).

    l: angular momentum quantum number (0, 1, 2, ...)
    p: parity (+1 for even 'e', -1 for odd 'o')
    dim: representation dimension = 2*l + 1
    """

    __slots__ = ("_l", "_p")

    def __init__(self, l: Union[int, str, "Irrep", tuple], p: int = None):
        if isinstance(l, Irrep):
            self._l, self._p = l.l, l.p
            return
        if isinstance(l, str):
            l, p = self._parse(l)
        if isinstance(l, (tuple, list)):
            l, p = l
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be a non-negative integer, got {l}")
        if p not in (1, -1):
            raise ValueError(f"p must be +1 or -1, got {p}")
        self._l = l
        self._p = p

    @staticmethod
    def _parse(s: str) -> tuple[int, int]:
        s = s.strip()
        match = re.fullmatch(r"(\d+)([eoy])", s)
        if not match:
            raise ValueError(f"Cannot parse irrep string: '{s}'")
        l = int(match.group(1))
        char = match.group(2)
        if char == "e":
            p = 1
        elif char == "o":
            p = -1
        else:  # 'y' — spherical harmonic convention
            p = (-1) ** l
        return l, p

    @property
    def l(self) -> int:
        return self._l

    @property
    def p(self) -> int:
        return self._p

    @property
    def dim(self) -> int:
        return 2 * self._l + 1

    def __repr__(self) -> str:
        p_char = "e" if self._p == 1 else "o"
        return f"{self._l}{p_char}"

    def __eq__(self, other) -> bool:
        if isinstance(other, Irrep):
            return self._l == other._l and self._p == other._p
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._l, self._p))

    def __lt__(self, other: "Irrep") -> bool:
        return (self._l, self._p) < (other._l, other._p)

    def __mul__(self, other: "Irrep") -> list["Irrep"]:
        """Tensor product selection rules: |l1-l2| <= l3 <= l1+l2, p3 = p1*p2."""
        other = Irrep(other)
        p_out = self._p * other._p
        return [
            Irrep(l, p_out)
            for l in range(abs(self._l - other._l), self._l + other._l + 1)
        ]


MulIr = namedtuple("MulIr", ["mul", "ir"])
MulIr.__doc__ = "Multiplicity-Irrep pair: (mul, ir) with dim = mul * ir.dim"
MulIr.dim = property(lambda self: self.mul * self.ir.dim)

SortResult = namedtuple("SortResult", ["irreps", "p", "inv"])


class Irreps:
    """Direct sum of irreducible representations.

    Parses strings like "32x0e + 16x1o + 8x2e" into a tuple of MulIr.
    Provides dim, slices, sort, simplify and other utilities for
    managing equivariant feature layouts.
    """

    def __init__(self, irreps: Union[str, "Irreps", Sequence, None] = None):
        if irreps is None or (isinstance(irreps, str) and irreps.strip() == ""):
            self._data: tuple[MulIr, ...] = ()
            return
        if isinstance(irreps, Irreps):
            self._data = irreps._data
            return
        if isinstance(irreps, str):
            self._data = self._parse_string(irreps)
            return
        # Sequence of (mul, ir) or Irrep or str
        parsed = []
        for item in irreps:
            if isinstance(item, MulIr):
                parsed.append(item)
            elif isinstance(item, Irrep):
                parsed.append(MulIr(1, item))
            elif isinstance(item, str):
                parsed.extend(self._parse_string(item))
            elif isinstance(item, (tuple, list)):
                mul, ir_spec = item
                if isinstance(ir_spec, Irrep):
                    parsed.append(MulIr(int(mul), ir_spec))
                elif isinstance(ir_spec, str):
                    parsed.append(MulIr(int(mul), Irrep(ir_spec)))
                elif isinstance(ir_spec, (tuple, list)):
                    parsed.append(MulIr(int(mul), Irrep(*ir_spec)))
                else:
                    raise ValueError(f"Cannot parse irrep spec: {ir_spec}")
            else:
                raise ValueError(f"Cannot parse irreps item: {item}")
        self._data = tuple(parsed)

    @staticmethod
    def _parse_string(s: str) -> tuple[MulIr, ...]:
        result = []
        s = s.strip()
        if not s:
            return ()
        for part in s.split("+"):
            part = part.strip()
            if not part:
                continue
            if "x" in part:
                mul_str, ir_str = part.split("x", 1)
                mul = int(mul_str.strip())
                ir = Irrep(ir_str.strip())
            else:
                mul = 1
                ir = Irrep(part)
            result.append(MulIr(mul, ir))
        return tuple(result)

    @cached_property
    def dim(self) -> int:
        """Total dimension of the representation."""
        return sum(mulir.mul * mulir.ir.dim for mulir in self._data)

    @cached_property
    def num_irreps(self) -> int:
        """Total number of irreps (sum of multiplicities)."""
        return sum(mulir.mul for mulir in self._data)

    @property
    def lmax(self) -> int:
        if not self._data:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(mulir.ir.l for mulir in self._data)

    @cached_property
    def ls(self) -> list[int]:
        """List of l values, each repeated by multiplicity."""
        result = []
        for mul, ir in self._data:
            result.extend([ir.l] * mul)
        return result

    @cached_property
    def slices(self) -> list[slice]:
        """Return slice objects for indexing into the flat feature tensor."""
        result = []
        offset = 0
        for mulir in self._data:
            d = mulir.mul * mulir.ir.dim
            result.append(slice(offset, offset + d))
            offset += d
        return result

    def sort(self) -> SortResult:
        """Sort by (l, p). Returns SortResult(irreps, p, inv)."""
        indexed = list(enumerate(self._data))
        indexed.sort(key=lambda x: (x[1].ir.l, x[1].ir.p))
        p = tuple(i for i, _ in indexed)
        inv = [0] * len(p)
        for new_idx, old_idx in enumerate(p):
            inv[old_idx] = new_idx
        sorted_data = tuple(mulir for _, mulir in indexed)
        return SortResult(Irreps(list(sorted_data)), p, tuple(inv))

    def simplify(self) -> "Irreps":
        """Merge consecutive identical irreps, remove zero multiplicities."""
        if not self._data:
            return Irreps()
        result = []
        current_mul, current_ir = self._data[0]
        for mulir in self._data[1:]:
            if mulir.ir == current_ir:
                current_mul += mulir.mul
            else:
                if current_mul > 0:
                    result.append(MulIr(current_mul, current_ir))
                current_mul, current_ir = mulir.mul, mulir.ir
        if current_mul > 0:
            result.append(MulIr(current_mul, current_ir))
        return Irreps(result)

    def regroup(self) -> "Irreps":
        """Sort and then simplify (group like irreps together)."""
        return self.sort().irreps.simplify()

    def count(self, ir: Union[Irrep, str]) -> int:
        """Total multiplicity of a given irrep across all entries."""
        ir = Irrep(ir)
        return sum(mulir.mul for mulir in self._data if mulir.ir == ir)

    def filter(
        self,
        keep: Union[str, "Irreps", None] = None,
        drop: Union[str, "Irreps", None] = None,
        lmax: int = None,
    ) -> "Irreps":
        """Filter irreps by keep/drop sets or lmax."""
        keep_set = None
        if keep is not None:
            keep_irreps = Irreps(keep)
            keep_set = {mulir.ir for mulir in keep_irreps._data}
        drop_set = set()
        if drop is not None:
            drop_irreps = Irreps(drop)
            drop_set = {mulir.ir for mulir in drop_irreps._data}

        result = []
        for mulir in self._data:
            if lmax is not None and mulir.ir.l > lmax:
                continue
            if keep_set is not None and mulir.ir not in keep_set:
                continue
            if mulir.ir in drop_set:
                continue
            result.append(mulir)
        return Irreps(result)

    def remove_zero_multiplicities(self) -> "Irreps":
        return Irreps([mulir for mulir in self._data if mulir.mul > 0])

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> "Irreps":
        """Irreps for spherical harmonics up to lmax.

        p=-1: standard physics convention (alternating e/o)
        p=+1: all same parity
        """
        result = []
        for l in range(lmax + 1):
            if p == -1:
                parity = (-1) ** l
            else:
                parity = 1
            result.append(MulIr(1, Irrep(l, parity)))
        return Irreps(result)

    def __add__(self, other: Union["Irreps", str]) -> "Irreps":
        other = Irreps(other)
        return Irreps(list(self._data) + list(other._data))

    def __radd__(self, other):
        if isinstance(other, (Irreps, str)):
            return Irreps(other) + self
        return NotImplemented

    def __mul__(self, scalar: int) -> "Irreps":
        return Irreps([MulIr(mulir.mul * scalar, mulir.ir) for mulir in self._data])

    def __rmul__(self, scalar: int) -> "Irreps":
        return self.__mul__(scalar)

    def __iter__(self) -> Iterator[MulIr]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._data[idx]
        return Irreps(list(self._data[idx]))

    def __contains__(self, ir: Union[Irrep, str]) -> bool:
        ir = Irrep(ir)
        return any(mulir.ir == ir for mulir in self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            other = Irreps(other)
        if isinstance(other, Irreps):
            return self._data == other._data
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._data)

    def __repr__(self) -> str:
        parts = []
        for mul, ir in self._data:
            if mul == 1:
                parts.append(repr(ir))
            else:
                parts.append(f"{mul}x{ir!r}")
        return " + ".join(parts) if parts else ""

    def __str__(self) -> str:
        return repr(self)

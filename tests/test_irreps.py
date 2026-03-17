"""Tests for Irreps data structure."""

from mace_mlx.irreps import Irrep, Irreps, MulIr


class TestIrrep:
    def test_creation(self):
        ir = Irrep(0, 1)
        assert ir.l == 0
        assert ir.p == 1
        assert ir.dim == 1

    def test_from_string(self):
        assert Irrep("0e") == Irrep(0, 1)
        assert Irrep("1o") == Irrep(1, -1)
        assert Irrep("2e") == Irrep(2, 1)
        assert Irrep("3o") == Irrep(3, -1)

    def test_dim(self):
        assert Irrep("0e").dim == 1
        assert Irrep("1e").dim == 3
        assert Irrep("2e").dim == 5
        assert Irrep("3o").dim == 7

    def test_y_parity(self):
        assert Irrep("0y") == Irrep(0, 1)  # (-1)^0 = 1
        assert Irrep("1y") == Irrep(1, -1)  # (-1)^1 = -1
        assert Irrep("2y") == Irrep(2, 1)  # (-1)^2 = 1

    def test_eq_hash(self):
        assert Irrep("0e") == Irrep("0e")
        assert Irrep("0e") != Irrep("0o")
        assert hash(Irrep("1e")) == hash(Irrep(1, 1))

    def test_mul_selection_rules(self):
        products = Irrep("1e") * Irrep("1e")
        assert len(products) == 3  # l=0, 1, 2
        assert Irrep(0, 1) in products
        assert Irrep(1, 1) in products
        assert Irrep(2, 1) in products

    def test_repr(self):
        assert repr(Irrep(0, 1)) == "0e"
        assert repr(Irrep(1, -1)) == "1o"


class TestIrreps:
    def test_from_string(self):
        irreps = Irreps("32x0e + 16x1o + 8x2e")
        assert len(irreps) == 3
        assert irreps[0] == MulIr(32, Irrep(0, 1))
        assert irreps[1] == MulIr(16, Irrep(1, -1))
        assert irreps[2] == MulIr(8, Irrep(2, 1))

    def test_single_irrep_no_mul(self):
        irreps = Irreps("0e + 1o")
        assert len(irreps) == 2
        assert irreps[0].mul == 1
        assert irreps[1].mul == 1

    def test_dim(self):
        irreps = Irreps("2x0e + 1x1e")
        assert irreps.dim == 2 * 1 + 1 * 3  # = 5

        irreps2 = Irreps("32x0e + 16x1o + 8x2e")
        assert irreps2.dim == 32 * 1 + 16 * 3 + 8 * 5  # = 120

    def test_num_irreps(self):
        irreps = Irreps("32x0e + 16x1o + 8x2e")
        assert irreps.num_irreps == 32 + 16 + 8  # = 56

    def test_lmax(self):
        assert Irreps("32x0e + 16x1o + 8x2e").lmax == 2
        assert Irreps("0e").lmax == 0

    def test_ls(self):
        irreps = Irreps("2x0e + 1x1o")
        assert irreps.ls == [0, 0, 1]

    def test_slices(self):
        irreps = Irreps("2x0e + 1x1e + 1x2e")
        slices = irreps.slices
        assert slices[0] == slice(0, 2)   # 2*1 = 2
        assert slices[1] == slice(2, 5)   # 1*3 = 3
        assert slices[2] == slice(5, 10)  # 1*5 = 5

    def test_simplify(self):
        irreps = Irreps("1x0e + 1x0e + 1x1e")
        simplified = irreps.simplify()
        assert simplified == Irreps("2x0e + 1x1e")

    def test_simplify_removes_zero(self):
        irreps = Irreps([(0, "0e"), (3, "1o")])
        simplified = irreps.simplify()
        assert simplified == Irreps("3x1o")

    def test_sort(self):
        irreps = Irreps("1x1e + 1x0e")
        result = irreps.sort()
        assert result.irreps[0].ir == Irrep("0e")
        assert result.irreps[1].ir == Irrep("1e")

    def test_count(self):
        irreps = Irreps("3x0e + 2x1e + 1x1e")
        assert irreps.count("0e") == 3
        assert irreps.count("1e") == 3
        assert irreps.count("2e") == 0

    def test_filter_keep(self):
        irreps = Irreps("1x0e + 2x1o + 1x2e")
        filtered = irreps.filter(keep="1o")
        assert filtered == Irreps("2x1o")

    def test_filter_lmax(self):
        irreps = Irreps("1x0e + 2x1o + 1x2e")
        filtered = irreps.filter(lmax=1)
        assert filtered == Irreps("1x0e + 2x1o")

    def test_add(self):
        result = Irreps("0e") + Irreps("1e")
        assert result == Irreps("0e + 1e")

    def test_mul_scalar(self):
        result = Irreps("1x0e + 1x1e") * 3
        assert result == Irreps("3x0e + 3x1e")

    def test_spherical_harmonics(self):
        sh = Irreps.spherical_harmonics(3)
        assert len(sh) == 4
        assert sh[0].ir == Irrep("0e")
        assert sh[1].ir == Irrep("1o")
        assert sh[2].ir == Irrep("2e")
        assert sh[3].ir == Irrep("3o")

    def test_empty(self):
        empty = Irreps("")
        assert empty.dim == 0
        assert len(empty) == 0

    def test_iter(self):
        irreps = Irreps("2x0e + 3x1o")
        items = list(irreps)
        assert len(items) == 2
        mul, ir = items[0]
        assert mul == 2 and ir == Irrep("0e")

    def test_contains(self):
        irreps = Irreps("2x0e + 3x1o")
        assert "0e" in irreps
        assert "1o" in irreps
        assert "2e" not in irreps

    def test_repr(self):
        assert repr(Irreps("2x0e + 1x1o")) == "2x0e + 1o"

    def test_from_sequence(self):
        irreps = Irreps([(32, (0, 1)), (16, (1, -1))])
        assert irreps == Irreps("32x0e + 16x1o")

    def test_regroup(self):
        irreps = Irreps("1x1e + 1x0e + 1x1e")
        regrouped = irreps.regroup()
        assert regrouped.count("0e") == 1
        assert regrouped.count("1e") == 2

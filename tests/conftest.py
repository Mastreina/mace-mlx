"""Shared test fixtures for MACE-MLX tests."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def random_vectors(rng):
    """Random 3D vectors for testing, shape (100, 3)."""
    return rng.standard_normal((100, 3))


@pytest.fixture
def random_unit_vectors(random_vectors):
    """Random unit vectors for spherical harmonics testing."""
    norms = np.linalg.norm(random_vectors, axis=-1, keepdims=True)
    return random_vectors / norms

"""Correctness checks for the from-scratch LSTM cell.

The reference test copies parameters into torch.nn.LSTMCell and asserts the
outputs match within atol=1e-5.
"""

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import LSTMCell

ATOL = 1e-5


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)


def test_output_shapes() -> None:
    batch, input_size, hidden_size = 4, 8, 16
    cell = LSTMCell(input_size, hidden_size)
    x = torch.randn(batch, input_size)

    h, c = cell(x)

    assert h.shape == (batch, hidden_size)
    assert c.shape == (batch, hidden_size)


def test_default_state_is_zeros() -> None:
    batch, input_size, hidden_size = 2, 5, 7
    cell = LSTMCell(input_size, hidden_size)
    x = torch.randn(batch, input_size)

    h_default, c_default = cell(x)
    zeros = torch.zeros(batch, hidden_size)
    h_explicit, c_explicit = cell(x, (zeros, zeros))

    assert torch.allclose(h_default, h_explicit, atol=ATOL)
    assert torch.allclose(c_default, c_explicit, atol=ATOL)


@pytest.mark.parametrize("bias", [True, False])
def test_matches_torch_lstmcell(bias: bool) -> None:
    batch, input_size, hidden_size = 3, 10, 20
    ours = LSTMCell(input_size, hidden_size, bias=bias)
    ref = nn.LSTMCell(input_size, hidden_size, bias=bias)

    # Share parameters so any numerical difference is from the math, not init.
    with torch.no_grad():
        ours.weight_ih.copy_(ref.weight_ih)
        ours.weight_hh.copy_(ref.weight_hh)
        if bias:
            ours.bias_ih.copy_(ref.bias_ih)
            ours.bias_hh.copy_(ref.bias_hh)

    x = torch.randn(batch, input_size)
    h0 = torch.randn(batch, hidden_size)
    c0 = torch.randn(batch, hidden_size)

    h_ours, c_ours = ours(x, (h0, c0))
    h_ref, c_ref = ref(x, (h0, c0))

    assert torch.allclose(h_ours, h_ref, atol=ATOL)
    assert torch.allclose(c_ours, c_ref, atol=ATOL)

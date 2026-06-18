"""Hand-written LSTM cell.

Reference: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory".
Gate ordering and parameter layout follow torch.nn.LSTMCell so the two can be
compared directly (see tests/): weights are stacked in the order
input (i), forget (f), cell/candidate (g), output (o).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class LSTMCell(nn.Module):
    """A single LSTM step, computed from scratch.

    Given an input ``x`` and the previous hidden/cell state ``(h, c)``, returns
    the next ``(h', c')``. Matches ``torch.nn.LSTMCell`` numerically when given
    the same parameters.

    Parameters are laid out as in PyTorch:
        weight_ih: (4 * hidden_size, input_size)
        weight_hh: (4 * hidden_size, hidden_size)
        bias_ih:   (4 * hidden_size,)
        bias_hh:   (4 * hidden_size,)
    with the four chunks ordered [i, f, g, o].
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters uniformly, as torch.nn.LSTMCell does."""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute one LSTM step.

        Args:
            x: input tensor of shape (batch, input_size).
            state: optional (h, c), each of shape (batch, hidden_size).
                Defaults to zeros.

        Returns:
            (h_next, c_next), each of shape (batch, hidden_size).
        """
        if state is None:
            zeros = x.new_zeros(x.size(0), self.hidden_size)
            h_prev, c_prev = zeros, zeros
        else:
            h_prev, c_prev = state

        # TODO(01-lstm-cell): implement the LSTM recurrence from scratch.
        #   1. gates = x @ weight_ih.T (+ bias_ih) + h_prev @ weight_hh.T (+ bias_hh)
        #   2. split gates into i, f, g, o along dim=1
        #   3. i, f, o = sigmoid(...); g = tanh(...)
        #   4. c_next = f * c_prev + i * g
        #   5. h_next = o * tanh(c_next)
        raise NotImplementedError("LSTMCell.forward not implemented yet")

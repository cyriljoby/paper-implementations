"""Seq2Seq model: encoder-decoder LSTM (Sutskever et al., 2014).

Encoder reads the source sequence into a fixed state; decoder generates the
target one token at a time from that state. Uses nn.LSTM (the LSTM cell itself
is hand-written in 01-lstm-cell). batch_first throughout, matching data.py.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
import torch

from constants import PAD_ID


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # stored so Seq2Seq can check the decoder's dims line up with ours
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # each row is a vocab word's embedding vector, initialized to random weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        # dropout is a regulaizer that randomly zeroes out some fraction of the embedding dimensions to avoid overfitting certain features
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embed_dim, # width of the input to hidden weight matrix
            hidden_dim, # width of the hidden to hidden weight matrix. height of hidden to hidden and input to hidden weight matrices
            num_layers=num_layers, # number of stacked LSTM layers. making it deep to learn more complex representations
            dropout=dropout if num_layers > 1 else 0.0, #dropout to the output of each layer except the last
            batch_first=True,  # states that tensor fed to LSTM will be in format (batch, seq_len, input_size)
        )
    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """
            Before (src, shape (batch, seq_len)) — each token is a single integer:
            [
            [t1.1, t1.2, t1.3],     ← sentence 1: 3 token ids
            [t2.1, t2.2, t2.3],     ← sentence 2: 3 token ids
            ]

            After (embedded, shape (batch, seq_len, embed_dim)) — each integer is replaced by a 256-d vector:
            [
            [ [t1.1...], [t1.2...], [t1.3...] ],   ← sentence 1: 3 vectors
            [ [t2.1...], [t2.2...], [t2.3...] ],   ← sentence 2: 3 vectors
            ]
        """
        embedded = self.dropout(self.embedding(src)) # (batch, seq_len) -> (batch, seq_len, embed_dim)
        o, (h, c) = self.rnn(embedded) # (batch, seq_len, hidden_dim), (num_layer, batch, hidden_dim), (num_layer, batch, hidden_dim)
        return h, c


class Decoder(nn.Module):
    def __init__(self, 
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,) -> None:
        super().__init__()
        # saving for assertion in Seq2Seq that encoder/decoder hidden dims match
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # maps the hidden state to one logit (raw score) per target-vocab token
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, input_token: Tensor, hidden: Tensor, cell: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # makes the input_token 3-d so it can be fed to the LSTM
        input_token = input_token.unsqueeze(1)  # (batch,) -> (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1) -> (batch, 1, embed_dim)
        o, (h, c) = self.rnn(embedded, (hidden, cell))  # (batch, 1, embed_dim) -> (batch, 1, hidden_dim), (num_layer, batch, hidden_dim), (num_layer, batch, hidden_dim)
        logits = self.fc_out(o.squeeze(1))  # (batch, hidden_dim) -> (batch, vocab_size)
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        assert encoder.hidden_dim == decoder.hidden_dim, "Encoder and decoder hidden dimensions must match."
        assert encoder.num_layers == decoder.num_layers, "Encoder and decoder layer counts must match."
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src: Tensor, tgt: Tensor, teacher_forcing_ratio: float = 1.0
    ) -> Tensor:
        # forward processes all rows (all sentences in the batch) simultaneously
        batch, seq_len = tgt.size()
        outputs = torch.zeros(batch, seq_len, self.decoder.vocab_size, device=src.device)
        hidden, cell = self.encoder(src)
        input_token = tgt[:, 0]  # first token is always <sos>. column 0 all rows
        for time_step in range(1, seq_len):
            logits, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, time_step] = logits
            # teacher forcing for now
            input_token = tgt[:, time_step]
        return outputs

# LSTM Cell — From Scratch

A from-scratch PyTorch implementation of the Long Short-Term Memory cell
(Hochreiter & Schmidhuber, 1997), built as a learning exercise before using
`nn.LSTM` in the implementations of other NNs that follow.

## Why

Understanding the gate mechanics, forget, input, output, and the additive
cell state update that solves the vanishing gradient problem, by implementing them.

#Understanding LSTM StagesLong Short-Term Memory (LSTM) networks use a gating mechanism to regulate the flow of information, allowing the network to effectively maintain both long-term and short-term dependencies.1. The Forget GateThe forget gate decides how much of the long-term memory to keep. It looks at the previous hidden state (short-term memory) and the current input, applies weights, and passes them through a sigmoid function to produce a value between 0 and 1 for each element in the cell state.0 = Completely forget the information.1 = Completely keep the information.$$\text{Output Factor} = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$Key Output: > $\text{Long-Term Memory} = \text{Long-Term Memory} \times \% \text{ Long-Term to Remember}$2. The Input GateThe input gate decides what new information to store in the long-term memory. It consists of two parallel blocks:Right Block (Potential Long-Term Memory): Combines the short-term memory and current input, then passes them through a $\tanh$ activation function to create candidate values ($\tilde{C}_t$).Left Block (% Potential Memory to Remember): Runs a sigmoid layer to determine what percentage of that new potential memory should actually be added to the long-term memory.Key Output: > $\text{New Long-Term Memory} = (\text{Potential Long-Term Memory} \times \% \text{ Potential Memory to Remember}) + \text{Long-Term Memory (Post-Forget Gate)}$3. The Output GateThe output gate determines the new short-term memory (hidden state) that will be passed to the next cell and used for the current step's prediction.Right Block (Potential Short-Term Memory): The newly updated long-term memory is passed through a $\tanh$ function to scale the values between -1 and 1.Left Block (% Potential Memory to Remember): A sigmoid layer determines which parts of the potential short-term memory should actually be outputted.Key Output: > $\text{New Short-Term Memory} = \text{Potential Short-Term Memory} \times \% \text{ Potential Memory to Remember}$


## How It Works

The cell takes the current input and previous (hidden state, cell state), and
produces updated versions of both through three gates:

- **Forget gate:** sigmoid over [h_prev, x] — determines how much of the
  existing cell state to keep
- **Input gate:** sigmoid over [h_prev, x] scaled against a tanh candidate —
  determines what new information to add to the cell state
- **Output gate:** sigmoid over [h_prev, x] scaled against tanh of the updated
  cell state — determines what to expose as the new hidden state

The cell state itself is updated additively (old * forget + new * input),
which gives gradients a direct path to flow backward through many time steps
without vanishing.

## Verification

The implementation is tested against PyTorch's `nn.LSTMCell` — given identical
weights and inputs, outputs match within `atol=1e-5`.

```bash
pytest tests/test_lstm_cell.py
```

## What's Next

This cell is the primitive used inside the encoder and decoder LSTMs in
[02-seq2seq/](../02-seq2seq/), where two separate LSTM networks are chained
through a shared hidden state to map one variable-length sequence to another.
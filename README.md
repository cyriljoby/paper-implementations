# Paper Implementations

From-scratch PyTorch implementations of foundational ML papers, ordered to show the
progression from recurrent models to attention-based architectures. This is a
portfolio/learning project — clarity and correctness matter more than performance.

## Reading order

Folders are numbered for reading order. Each is added only when its implementation
begins (no empty scaffolding):

1. `01-lstm-cell/` — hand-written LSTM cell
2. `02-seq2seq/` — encoder–decoder sequence-to-sequence
3. `03-attention/` — additive/dot-product attention
4. `04-transformer/` — full from-scratch transformer

## Layout of each implementation

```
NN-name/
├── README.md          ← paper summary, design choices, deviations, results
├── src/               ← implementation files
└── tests/             ← correctness checks (shape, overfit-one-batch, reference)
```

## Conventions

- Python 3.10+, PyTorch only — no HuggingFace, no `nn.TransformerEncoder`. Core
  primitives (attention, LSTM cell) are hand-written; `nn.Linear`, `nn.Embedding`,
  `nn.Dropout` are fair game.
- Type hints on all function signatures.
- Seed everything: `torch.manual_seed`, `random.seed`, `numpy.random.seed`.
- Every model must pass an overfit-one-batch test before full training.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

Run `pytest tests/` from inside any implementation folder.

See [CLAUDE.md](CLAUDE.md) for full conventions.

# GPT From Scratch

Building GPT-style language models from scratch, following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial series.

The focus is on understanding — every line of code is explained, every concept is built up from first principles.

## Notebooks

### [`nanogpt.ipynb`](notebooks/nanogpt.ipynb) — NanoGPT on Shakespeare

A complete GPT implementation trained on Tiny Shakespeare (~1MB of text).

Built progressively in two parts:

**Part 1: Bigram Language Model**
- Character-level tokenizer (encode/decode)
- Train/validation split and batch loading
- Bigram model — the simplest possible baseline
- Training loop with loss tracking

**Part 2: The Transformer**
- The mathematical intuition behind attention (bag of words → matrix trick → softmax → self-attention)
- Self-attention head with Query, Key, Value projections
- Multi-head attention
- Feed-forward network
- Residual connections + Layer Normalisation
- Full GPT model with positional embeddings
- Loss curve visualisation

**Results after 5000 training steps (~20 min on RTX 5060 Ti):**
```
Train loss: 1.09
Val loss:   1.47
```

**Sample output:**
```
OXFORD:
Strink me not: but know it banishment,
And hence instill with that you caft hard
Of kisters' mind or feath, to extrempt a treason.

KING HENRY VI:
My deserves hate that doubt thou art dull:
yet what we is he wink was the arguit;
Why there is depress'd: I'ld no scapace it but frail
```

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | 10.8M |
| Embedding dim (`n_embd`) | 384 |
| Attention heads (`n_head`) | 6 |
| Transformer layers (`n_layer`) | 6 |
| Context window (`block_size`) | 256 |
| Dropout | 0.2 |
| Training steps | 5000 |
| Batch size | 64 |

## Setup

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install jupyter matplotlib
```

Requires Python 3.10+ and an NVIDIA GPU with CUDA support. Tested on RTX 5060 Ti (16GB).

Open `nanogpt.ipynb` in Jupyter or VS Code and run cells top to bottom.

## What's Next

- [ ] Train on fantasy fiction (Royal Road / Project Gutenberg)
- [ ] Save and load model checkpoints
- [ ] Subword tokenisation (BPE)
- [ ] Scale up model size

## References

- [Andrej Karpathy — Let's build GPT (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

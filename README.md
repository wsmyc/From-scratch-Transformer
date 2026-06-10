# Transformer from Scratch: English-to-Italian Neural Machine Translation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-brightgreen)](https://arxiv.org/abs/1706.03762)

A clean, modular, and fully-documented PyTorch implementation of the **Transformer** architecture introduced in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). This project trains an English-to-Italian sequence-to-sequence model without relying on high-level abstractions like `torch.nn.Transformer`, making every mathematical operation explicit and inspectable.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Mathematical Foundations](#mathematical-foundations)
   - [Scaled Dot-Product Attention](#1-scaled-dot-product-attention)
   - [Multi-Head Attention](#2-multi-head-attention)
   - [Positional Encoding](#3-positional-encoding)
   - [Feed-Forward Network](#4-position-wise-feed-forward-network)
   - [Layer Normalization & Residuals](#5-layer-normalization--residual-connections)
   - [Masking Strategies](#6-masking-strategies)
3. [Architecture & Code Map](#architecture--code-map)
4. [Dataset & Tokenization](#dataset--tokenization)
5. [Training Configuration](#training-configuration)
6. [Usage](#usage)
7. [Extending the Project](#extending-the-project)
8. [References](#references)

---

## Project Overview

This repository implements the complete Transformer encoder-decoder stack from first principles. It is designed for:

- **Educational clarity**: Every sub-layer (attention, FFN, normalization) is written as a standalone `nn.Module`.
- **Research flexibility**: Easy to modify attention variants, positional encoding schemes, or depth.
- **Practical application**: A working end-to-end training pipeline for bilingual machine translation (EN → IT).

The architecture strictly follows the original paper's hyperparameter recommendations (d_model=512, 8 attention heads, 6 encoder/decoder layers) while using **Pre-LayerNorm** residual connections for training stability—a widely adopted modern refinement.

---

## Mathematical Foundations

### 1. Scaled Dot-Product Attention

The core operation of the Transformer is attention, defined as a weighted sum of values $V$, where the weight assigned to each value is computed by a compatibility function between the query $Q$ and the corresponding key $K$.

For an input sequence, we first project the hidden states into query, key, and value matrices:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ are learned linear projections.

The attention function is then:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why the scale factor $\frac{1}{\sqrt{d_k}}$?**  
For large values of $d_k$, the dot products $QK^T$ grow in magnitude, pushing the softmax into regions with extremely small gradients. The scaling factor counteracts this effect.

In code (`model.py`):
```python
attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
attention_scores = attention_scores.softmax(dim=-1)
output = attention_scores @ value
```

### 2. Multi-Head Attention

Rather than performing a single attention function with $d_{\text{model}}$-dimensional keys, values, and queries, it is beneficial to linearly project them $h$ times into smaller dimensions $d_k = d_v = d_{\text{model}} / h$.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Dimensions in this project:**
- $d_{\text{model}} = 512$
- $h = 8$ heads
- $d_k = d_v = 512 / 8 = 64$ per head

The implementation reshapes the projected tensors from `(batch, seq, 512)` → `(batch, 8, seq, 64)`, computes attention in parallel across heads, concatenates back to `(batch, seq, 512)`, and applies a final output projection $W^O$.

### 3. Positional Encoding

Since the Transformer contains no recurrence or convolution, it is permutation-invariant. To inject information about the relative or absolute position of tokens, we add sinusoidal positional encodings to the input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the token position and $i$ is the dimension index.

**Properties:**
- The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.
- It allows the model to attend to relative positions via linear transformations, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

In code, the encoding matrix is pre-computed and registered as a non-trainable buffer:

```python
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

### 4. Position-Wise Feed-Forward Network

In addition to attention sub-layers, each layer contains a fully connected feed-forward network applied identically to each position:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This is equivalent to two linear transformations with a ReLU activation in between. The inner dimension is $d_{ff} = 2048$, expanding the representation before projecting back to $d_{\text{model}} = 512$.

### 5. Layer Normalization & Residual Connections

Each sub-layer (attention and FFN) is wrapped in a residual connection, followed by layer normalization.

**Layer Normalization** stabilizes training by normalizing across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance computed across the last dimension (the $d_{\text{model}}$ features), and $\gamma, \beta$ are learned affine parameters.

**Residual Connection (Pre-LN variant):**  
This implementation uses the **Pre-LayerNorm** formulation (He et al., 2016; Xiong et al., 2020), which applies normalization *before* the sub-layer, improving gradient flow in deep stacks:

$$x_{\text{out}} = x + \text{Dropout}(\text{Sublayer}(\text{LayerNorm}(x)))$$

This differs from the original paper's Post-LN but is now standard in most production Transformers.

### 6. Masking Strategies

Two masks are essential for correct autoregressive behavior:

| Mask | Purpose | Implementation |
|------|---------|----------------|
| **Padding Mask** | Prevents the model from attending to `<PAD>` tokens used for batching. | `(encoder_input != pad_token).unsqueeze(0).unsqueeze(0)` |
| **Causal (Look-Ahead) Mask** | Prevents the decoder from attending to future tokens during training. | `torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0` |

The decoder mask is the **logical AND** of the padding mask and the causal mask.

---

## Architecture & Code Map

```
Transformer/
├── config.py           # Hyperparameters & paths
├── dataset.py          # BilingualDataset with tokenization & masking
├── model.py            # Full Transformer stack (from scratch)
├── train.py            # Training loop with checkpointing
├── tokenizer_en.json   # English WordLevel tokenizer
└── tokenizer_it.json   # Italian WordLevel tokenizer
```

### `model.py` — Component Breakdown

| Class | Role | Mathematical Operation |
|-------|------|----------------------|
| `InputEmbeddings` | Token to vector | $x \mapsto \text{Embedding}(x) \times \sqrt{d_{\text{model}}}$ |
| `PositionalEncoding` | Inject position info | $x \mapsto x + PE$ |
| `MultiHeadAttention` | Relate tokens globally | Scaled dot-product over $h$ heads |
| `FeedForwardBlock` | Point-wise transformation | $\text{Linear}(\text{ReLU}(\text{Linear}(x)))$ |
| `LayerNorm` | Stabilize activations | Normalize over feature dim; learnable $\gamma, \beta$ |
| `ResidualConnection` | Enable deep gradients | Pre-LN residual: $x + \text{Dropout}(\text{Sublayer}(\text{Norm}(x)))$ |
| `EncoderBlock` | Encode source context | Self-Attn → FFN |
| `Encoder` | Stack of $N=6$ blocks | Repeat + final LayerNorm |
| `DecoderBlock` | Decode target autoregressively | Masked Self-Attn → Cross-Attn → FFN |
| `Decoder` | Stack of $N=6$ blocks | Repeat + final LayerNorm |
| `ProjectionLayer` | Map to vocabulary | $\text{Linear}(d_{\text{model}}, V_{\text{tgt}}) + \text{log\_softmax}$ |
| `Transformer` | End-to-end assembly | `encode()` → `decode()` → `project()` |

### `dataset.py` — Data Flow

1. **Tokenization**: Source (EN) and target (IT) sentences are encoded using WordLevel tokenizers.
2. **Special Tokens**: `[SOS]` (start), `[EOS]` (end), and `[PAD]` are prepended/appended.
3. **Padding**: Sequences are padded to `seq_len = 350`.
4. **Mask Construction**:
   - `encoder_mask`: Padding mask for the source.
   - `decoder_mask`: Padding mask AND causal mask for the target.
5. **Label Construction**: The target sequence shifted left by one position (teacher forcing).

---

## Dataset & Tokenization

This project is configured for **English → Italian** translation. The tokenizers are trained with Hugging Face `tokenizers` using a **WordLevel** model, which maps whole words (and common punctuation) to integer IDs.

**Special Tokens:**
| Token | ID Purpose |
|-------|------------|
| `[SOS]` | Start of sequence |
| `[EOS]` | End of sequence |
| `[PAD]` | Padding (ignored in loss/attention) |

*Note: For production-scale translation, you may replace the WordLevel tokenizers with BPE (Byte-Pair Encoding) or SentencePiece to handle out-of-vocabulary words and morphological richness.*

---

## Training Configuration

Defined in `config.py`:

```python
{
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "experiment_name": "runs/tmodel"
}
```

**Optimizer:** Adam with $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$ (standard for Transformers).  
**Loss:** Cross-Entropy with `ignore_index=pad_token_id`.  
**Scheduler:** (Recommended addition) Warmup + cosine decay.  
**Checkpointing:** Weights are saved per epoch as `tmodel_{epoch}.pt`.

---

## Usage

### 1. Installation

```bash
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
pip install torch tokenizers datasets tqdm tensorboard
```

### 2. Prepare Data

Ensure your bilingual dataset is in Hugging Face `datasets` format with a `translation` field:

```python
{
    "translation": {
        "en": "The cat sat on the mat.",
        "it": "Il gatto si sedette sul tappeto."
    }
}
```

### 3. Train

```bash
python train.py
```

Monitor training via TensorBoard:
```bash
tensorboard --logdir runs/tmodel
```

### 4. Inference / Translation

Load the trained model and run greedy decoding:

```python
from model import build_transformer
from config import get_config, get_weights_file_path

config = get_config()
model = build_transformer(
    src_vocab_size=10000, 
    tgt_vocab_size=10000,
    src_seq_len=config['seq_len'], 
    tgt_seq_len=config['seq_len']
)
# Load checkpoint
model.load_state_dict(torch.load(get_weights_file_path(config, "19")))

# Run encode → decode → project
encoder_output = model.encode(source_tokens, source_mask)
decoder_output = model.decode(encoder_output, source_mask, target_tokens, target_mask)
logits = model.project(decoder_output)
predicted_token = logits.argmax(dim=-1)
```

---

## Extending the Project

This codebase is intentionally modular. Here are direct ways to extend it:

| Goal | How To |
|------|--------|
| **New Language Pair** | Update `lang_src` / `lang_tgt` in `config.py` and train new tokenizers. |
| **BERT-style Encoder** | Use only the `Encoder` stack; remove the decoder and projection layer. Add a `[CLS]` token pooling head. |
| **GPT-style Decoder** | Remove the encoder and cross-attention. Use only the `Decoder` with causal masking. |
| **Vision Transformer** | Replace `InputEmbeddings` with a patch embedding layer (e.g., $16 \times 16$ image patches). Remove the decoder entirely for classification. |
| **Different Attention** | Swap `MultiHeadAttention` with Linear Attention, Flash Attention, or Sliding Window Attention. |
| **Beam Search** | Replace greedy `argmax` in inference with a beam search algorithm over `log_softmax` scores. |

---

## References

1. **Vaswani, A., et al.** (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS.
2. **He, K., et al.** (2016). Deep Residual Learning for Image Recognition. CVPR.
3. **Ba, J. L., et al.** (2016). Layer Normalization. arXiv:1607.06450.
4. **Xiong, R., et al.** (2020). On Layer Normalization in the Transformer Architecture. ICML.

---

**License:** MIT  
**Contributions:** Issues and PRs are welcome. If you use this for research or education, please cite the original *Attention Is All You Need* paper.

---

*Built with PyTorch. No black-box `nn.Transformer` modules were harmed in the making of this repository.*

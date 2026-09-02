# Character-Level Transformer Language Model

> A decoder-only Transformer built from scratch in PyTorch that learns to generate *A Song of Ice and Fire*-style text, one character at a time.

This project is a compact, educational implementation of a GPT-style language model. It trains directly on the raw text of *A Game of Thrones* and *A Clash of Kings*—no pretrained weights, tokenizers, or high-level Transformer libraries involved.

Given the preceding characters in a passage, the model predicts a distribution for the next one, then samples from it to keep writing. The generated text is not semantically dependable, but it picks up useful signals such as dialogue formatting, paragraph breaks, names, punctuation, and fantasy-style phrasing.

## What’s Inside

- **Decoder-only Transformer** with causal (masked) self-attention
- **Character-level vocabulary** learned from the supplied text files
- **6 Transformer blocks** with pre-layer normalization and residual connections
- **~11M trainable parameters**
- **Autoregressive generation** with multinomial sampling
- **Apple Silicon support** through PyTorch MPS, with CPU fallback

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Train and generate text

```bash
python transformer.py
```

The script combines the two novels, trains for 5,000 iterations, prints periodic training and validation losses, and finally generates 2,000 new characters. It uses an MPS device when available; otherwise it runs on CPU.

## How It Works

### Data Pipeline

The training corpus is the concatenation of:

- `A Clash of Kings.txt`
- `a game of thrones.txt`

Each unique character—including whitespace, punctuation, quotation marks, and newlines—becomes a token. The encoded text is split sequentially into 90% training data and 10% validation data.

During training, the model receives random 512-character windows. Its target is the same window shifted one character to the right, making this a standard next-character prediction task.

### Model Architecture

```text
Input characters
      │
      ├── Token embeddings
      ├── Learned positional embeddings
      │
      ▼
6 × Transformer blocks
      ├── Pre-LayerNorm
      ├── 6-head causal self-attention
      ├── Residual connection + dropout
      ├── Pre-LayerNorm
      ├── Feed-forward network
      └── Residual connection + dropout
      │
      ▼
Final LayerNorm → vocabulary projection → next-character logits
```

| Component | Configuration |
| --- | --- |
| Architecture | Decoder-only causal Transformer |
| Transformer blocks | 6 |
| Attention heads | 6 per block |
| Embedding dimension | 384 |
| Attention head dimension | 64 |
| Context window | 512 characters |
| Feed-forward width | 1,536 |
| Dropout | 0.2 |
| Normalization | Pre-LayerNorm |
| Objective | Cross-entropy next-character prediction |

The causal mask prevents each position from attending to future characters, so the model can only use context that would be available during generation.

### Training Setup

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Batch size | 64 sequences |
| Sequence length | 512 characters |
| Training iterations | 5,000 |
| Evaluation interval | 500 iterations |
| Evaluation batches | 200 per split |
| Random seed | 1337 |

## Results

This run was trained on an Apple M4 MPS 10-core GPU. Training and validation loss decrease steadily and remain close near the end of the run.

![Training and validation loss](https://github.com/user-attachments/assets/4d07650d-0af8-449d-b6f5-9af6034bfffc)

| Metric | Result |
| --- | ---: |
| Final training loss | **1.1451** |
| Final validation loss | **1.1959** |
| Validation perplexity | **~3.31** |
| Generated sample length | **2,000 characters** |

Perplexity is calculated as `exp(validation_loss)`. For a character-level model trained from scratch, this score indicates that the model has narrowed most next-character predictions to a small set of plausible options.

<details>
<summary>Full loss history</summary>

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.7964 | 4.7888 |
| 500 | 2.3628 | 2.3700 |
| 1,000 | 1.8123 | 1.8286 |
| 1,500 | 1.5618 | 1.5861 |
| 2,000 | 1.4209 | 1.4451 |
| 2,500 | 1.3327 | 1.3599 |
| 3,000 | 1.2698 | 1.3001 |
| 3,500 | 1.2285 | 1.2653 |
| 4,000 | 1.1962 | 1.2381 |
| 4,500 | 1.1647 | 1.2128 |
| 4,999 | **1.1451** | **1.1959** |

</details>

## Example Output

```text
Catelyn sns. “Nor is the road, my lord rate. Courage, I know my commanding the
room crows. I have you only these doorsapoiley.” When they could dung not forest for Joffrey’s
bastard stared beside Marsh free and fingerbridge.

Yes. The sound had no end.
“Take thy would above them leave to promise to any oil?” He told himself
a look at Casterly, “He got a warch for one is quite of him and he knew he would save me
here.”
```

The model does not understand plot or factual relationships, but it reproduces the visual rhythm of the source material: quoted dialogue, character-name fragments, line breaks, and Westerosi-sounding word patterns.

## Project Layout

```text
.
├── transformer.py          # Transformer, data pipeline, training, generation
├── transformers.ipynb      # Step-by-step language-model learning notebook
├── bigram.py               # Earlier bigram language-model experiment
├── requirements.txt        # Python dependencies
├── a game of thrones.txt   # Training corpus source
└── A Clash of Kings.txt    # Training corpus source
```

## Learning Path

The main implementation lives in `transformer.py`. The companion `transformers.ipynb` builds up the underlying ideas progressively:

1. Character tokenization and train/validation splits
2. Context windows and mini-batch construction
3. A bigram language-model baseline
4. Single-head causal self-attention and masking

## Notes

- This repository favors clarity over production abstractions: the model, data loading, training loop, and generation logic are intentionally visible.
- Generated passages are statistical continuations of patterns in the training text. They are not quotes, summaries, or canonical continuations of the source novels.
- The included novels remain subject to their respective copyright terms. Use the repository for personal learning and experimentation.

# Character-Level Transformer Language Model

An approximately **11M-parameter decoder-only Transformer**, implemented from scratch in PyTorch and trained to generate *A Song of Ice and Fire*-style text one character at a time.

Rather than using a pretrained tokenizer or model, this project learns directly from the raw text of *A Game of Thrones* and *A Clash of Kings*. Given up to 512 preceding characters, it predicts a probability distribution over the next character, then samples from that distribution to continue the passage.

The result is a compact model that learns the books' surface-level writing patterns: punctuation, spacing, dialogue, character names, sentence rhythm, and recognizably Westerosi vocabulary.

## Highlights

- Built from scratch with PyTorch — no Hugging Face model or pretrained weights
- Character-level next-token prediction
- 6-layer causal Transformer with multi-head self-attention
- ~11M trainable parameters
- Trained for 5,000 iterations on an Apple M4 MPS 10-core GPU
- Final validation loss of **1.1959** and validation perplexity of **~3.31**
- Generates 2,000-character samples with recognizably GoT-ish structure and phrasing

## Dataset

The training corpus is created by concatenating the included plain-text novels:

- `a game of thrones.txt`
- `A Clash of Kings.txt`

Every unique character in the combined corpus becomes a vocabulary item. The model therefore treats letters, whitespace, punctuation, quotation marks, and newlines as individual tokens.

The encoded corpus is split sequentially into:

| Split | Portion | Purpose |
| --- | ---: | --- |
| Training | 90% | Parameter updates |
| Validation | 10% | Held-out loss estimation |

## Architecture

This is a GPT-style autoregressive architecture: every position can attend only to itself and earlier positions. A lower-triangular causal mask prevents a character from seeing future characters during training.

```text
Input characters
      │
      ├── Token embeddings (384 dimensions)
      ├── Learned positional embeddings (up to 512 positions)
      │
      ▼
6 × Transformer blocks
      ├── Pre-LayerNorm
      ├── 6-head masked self-attention
      ├── Residual connection + dropout
      ├── Pre-LayerNorm
      ├── Feed-forward network (384 → 1536 → 384)
      └── Residual connection + dropout
      │
      ▼
Final LayerNorm → Linear vocabulary head → next-character logits
```

| Component | Configuration |
| --- | --- |
| Model type | Decoder-only causal Transformer |
| Parameters | ~11M |
| Transformer blocks | 6 |
| Attention heads | 6 per block |
| Embedding dimension | 384 |
| Head dimension | 64 (`384 / 6`) |
| Context length | 512 characters |
| Feed-forward width | 1,536 (4× embedding dimension) |
| Dropout | 0.2 |
| Normalization | Pre-LayerNorm |
| Objective | Cross-entropy next-character prediction |

## Training Configuration

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Batch size | 64 sequences |
| Sequence length | 512 characters |
| Training iterations | 5,000 |
| Evaluation interval | Every 500 iterations |
| Evaluation batches | 200 per split |
| Random seed | 1337 |
| Accelerator | Apple M4 MPS 10-core GPU |

At each step, the script samples 64 random contiguous 512-character windows from the training split. The input is each window; the target is the same window shifted by one character. AdamW then updates the model using cross-entropy loss.

## Training Results

![Training results](https://github.com/user-attachments/assets/4d07650d-0af8-449d-b6f5-9af6034bfffc)

Loss was evaluated at the beginning of training and every 500 iterations thereafter. Both training and validation loss steadily decreased, with a small final gap between them.

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

| Final metric | Value |
| --- | ---: |
| Training loss | **1.1451** |
| Validation loss | **1.1959** |
| Validation perplexity | **~3.31** |
| Generated sample length | **2,000 characters** |

Perplexity is calculated as `exp(validation_loss)`. A validation perplexity of about 3.31 means the model's next-character distribution is typically concentrated over a small number of plausible continuations — a strong outcome for a character-level model trained from scratch.

## Sample Generation

After training, generation begins from a zero-valued context token. The model repeatedly predicts and multinomial-samples the next character, keeping only the latest 512 characters as context once the sequence grows beyond the context window.

An excerpt from the 2,000-character output:

```text
Catelyn sns. “Nor is the road, my lord rate. Courage, I know my commanding the
room crows. I have you only these doorsapoiley.” When they could dung not forest for Joffrey’s
bastard stared beside Marsh free and fingerbridge.

Yes. The sound had no end.
“Take thy would above them leave to promise to any oil?” He told himself
a look at Casterly, “He got a warch for one is quite of him and he knew he would save me
here.”
```

The text is not semantically reliable, but it captures useful stylistic signals: dialogue punctuation, paragraph breaks, proper-name fragments, medieval/fantasy word choices, and the cadence of the source material.

## Project Structure

```text
.
├── transformer.py          # Model, data pipeline, training loop, and generation
├── requirements.txt        # Python dependency list
├── a game of thrones.txt   # Training corpus source
├── A Clash of Kings.txt    # Training corpus source
└── bigram.py               # Earlier bigram baseline
```

## Run Locally

### Requirements

- Python 3.9 or later
- PyTorch (listed in `requirements.txt`)
- A device supported by PyTorch; the script uses Apple Metal Performance Shaders (`mps`) when available and otherwise falls back to CPU

Create and activate a virtual environment, then install the project dependencies:

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

Then train and generate text:

```bash
python transformer.py
```

The script will:

1. Load and combine the two novels.
2. Build a character vocabulary and train/validation split.
3. Train the Transformer for 5,000 iterations.
4. Print train and validation loss every 500 steps.
5. Generate and print 2,000 new characters.

## Notes

- This project is educational and intentionally keeps the model, optimizer, data loading, and generation loop visible in one file.
- The model name in the code remains `BigramLanguageModel` from an earlier iteration, but the implemented network is a multi-layer Transformer language model.
- Generated text is a statistical continuation of patterns learned from the training corpus; it is not a faithful quote, summary, or continuation of either novel.

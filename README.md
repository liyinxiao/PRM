## Personalized Re-ranking for Recommendation

Pytorch implementation of Personalized Re-ranking for Recommendation (https://arxiv.org/abs/1904.06813).

### Model Files

- **`prm_model.py`** — PRM model using PyTorch built-ins (`nn.TransformerEncoder`, LayerNorm). Clean and concise.
- **`prm_model_custom.py`** — PRM model with custom multi-head attention and position-wise FFN, using BatchNorm (closer to the original TF implementation in LibRerank).

Both implement the same PRM architecture (embedding + position encoding + Transformer encoder + output head), but differ in normalization (LayerNorm vs BatchNorm) and input projection approach.

### Architecture

```
                         Input (per list of n items)
                         ========================

  itm_spar (B, n, 5)                          itm_dens (B, n, 1)
  [sparse feature IDs]                        [dense features]
        |                                           |
        v                                           |
  +--------------+                                  |
  | Embedding    |  (feature_size+1, eb_dim=16)     |
  | (padding_idx |   padding row = 0                |
  |  = 0)        |                                  |
  +--------------+                                  |
        |                                           |
        v                                           |
  (B, n, 5*16=80)                                   |
        |                                           |
        +-------------------+-----------------------+
                            |
                            v
                    Concat (B, n, 81)       <-- ft_num = spar_num*eb_dim + dens_num
                            |
                            +
                            |  + pos_mtx (n, 81)    <-- learnable position embedding
                            v
                    (B, n, 81)
                            |
                            v
                +-----------------------+
                |  Linear (81 -> 64)    |   <-- project to d_model
                +-----------------------+
                            |
                            v
              +-------------------------------+
              |   Transformer Encoder x4      |
              |  +-------------------------+  |
              |  | Multi-Head Attention     |  |
              |  | (d_model=64, n_head=1)   |  |
              |  +-------------------------+  |
              |             |                 |
              |        Add & Norm             |
              |             |                 |
              |  +-------------------------+  |
              |  | FFN (64 -> 128 -> 64)   |  |
              |  +-------------------------+  |
              |             |                 |
              |        Add & Norm             |
              +-------------------------------+
                            |
                            v
                      (B, n, 64)
                            |
                            v
              +-------------------------------+
              |        Output Head            |
              |  BatchNorm1d(64)              |
              |  Linear(64 -> 64) + ReLU      |
              |  Dropout                      |
              |  Linear(64 -> 1)              |
              +-------------------------------+
                            |
                            v
                      (B, n)  logits
                            |
                            v
                  Softmax over n positions
                            |
                            v
                  Mask padding positions
                            |
                            v
                  y_pred (B, n)  scores   --> re-rank by descending score
```

### Data

The `Data/` folder is from [LibRerank](https://github.com/LibRerank-Community/LibRerank).

### Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch numpy
```

### Training

```
python train.py --setting_path config/prm_setting.json
```

To use the custom (BatchNorm) variant, change the import in `train.py`:
```python
from prm_model_custom import PRMModel
```


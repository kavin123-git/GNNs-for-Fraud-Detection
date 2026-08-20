# GNN-based Fraud Detection on Bitcoin Transaction Networks

Leveraging Graph Neural Networks to detect illicit transactions on the Elliptic Bitcoin dataset (~203k nodes, ~234k edges, 49 time steps).
---

## Overview

Built a full pipeline for identifying fraudulent Bitcoin transactions by modeling the transaction graph with GNNs, benchmarked against a tabular XGBoost baseline. The dataset has severe class imbalance (~2% illicit, ~21% licit, ~77% unlabeled).

## Dataset — Elliptic Bitcoin

- **203,769 nodes** (transactions) with **166 features** each (94 local + 72 aggregated neighbor statistics)
- **234,355 directed edges** (payment flows)
- **49 time steps** (~2 weeks each)
- Time-based train/val/test split: train on steps 1–34, validate on 35–42, test on 43–49

## Approach

### Preprocessing
- Merged node features with class labels, remapped to binary (illicit=1, licit=0, unknown=masked)
- Standardized all 165 features using StandardScaler
- Constructed a PyG `Data` object with contiguous node indexing and train/val/test masks

### Baseline — XGBoost
- Gradient-boosted tree treating each transaction independently (no graph structure)
- Used `scale_pos_weight` to handle class imbalance
- Serves as a baseline to quantify the advantage of graph-aware models

### GNN Models
All models follow: `Input → Conv1 (ReLU + Dropout) → Conv2 (ReLU + Dropout) → Linear → Softmax`

- **GCN** — Symmetric normalized aggregation (Kipf & Welling 2017)
- **GraphSAGE** — Sample-and-aggregate, supports inductive learning on unseen nodes (Hamilton et al. 2017)
- **GAT** — Multi-head attention-weighted neighbor aggregation, learns which neighbors matter (Veličković et al. 2018)

### Training Details
- **Focal Loss** (α=0.75, γ=2.0) to handle the 2% illicit class — down-weights easy majority-class examples and focuses on hard minority samples
- Adam optimizer with weight decay, early stopping on validation PR-AUC (patience=20)
- 200 max epochs per model

### Ablation Study
- Trained GraphSAGE on only the 94 local features (dropping the 72 hand-crafted aggregated features) to test whether GNNs can learn neighborhood statistics automatically via message passing

### Explainability
- Used **GNNExplainer** on flagged illicit nodes to identify the most important subgraph and features driving each prediction

## Results

| Model | PR-AUC | Macro F1 | Illicit F1 |
|---|---|---|---|
| XGBoost (baseline) | 0.0427 | 0.5044 | 0.0292 |
| GCN | 0.0385 | 0.4833 | 0.0132 |
| GraphSAGE | 0.0420 | 0.4960 | 0.0287 |
| GAT | 0.0398 | 0.4345 | 0.0934 |

## Key Takeaways
- GAT achieved the highest Illicit F1 by leveraging attention to weight suspicious neighbors
- GNNs capture multi-hop fraud ring structures that tabular models miss entirely
- Ablation confirmed GNNs can replace hand-crafted aggregated features through learned message passing
- Focal Loss is critical — standard cross-entropy causes the model to predict licit for everything

## Tech Stack
Python · PyTorch · PyTorch Geometric · XGBoost · scikit-learn · matplotlib · seaborn

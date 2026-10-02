[README.md](https://github.com/user-attachments/files/32969050/README.md)
# grey-wolf-optimizer-feature-selection# Grey Wolf Optimizer for Feature Selection

A Python implementation of the Grey Wolf Optimizer (GWO) applied to feature selection in classification systems. Built as part of a university heuristic optimization course project.

## What it does

Datasets often have many features (columns), but not all of them are useful. This project uses GWO — an algorithm inspired by how grey wolves hunt in packs — to automatically find the smallest subset of features that still gives high classification accuracy.

Tested on the **Breast Cancer Wisconsin** dataset (569 samples, 30 features, 2 classes).

## Results

| Method | Accuracy | Features Used | Reduction |
|--------|----------|---------------|-----------|
| All Features | 92.79% | 30 | 0% |
| Random Selection | 84.01% | 2 | 93.3% |
| **GWO (ours)** | **95.08%** | **2** | **93.3%** |

GWO selected just **2 features** out of 30:
- `mean texture`
- `worst perimeter`

...and achieved **better accuracy** than using all 30 features.

## How it works

GWO simulates the social hierarchy of a grey wolf pack:
- **Alpha (α)** — best solution found so far
- **Beta (β)** — second best
- **Delta (δ)** — third best
- **Omega (Ω)** — remaining wolves, follow the top 3

Each wolf represents a possible feature subset. Over 50 iterations, wolves move toward the best solutions, gradually converging on an optimal feature subset.

## Installation

```bash
pip install scikit-learn numpy
```

## Usage

```bash
python gwo_feature_selection.py
```

The script will:
1. Load the Breast Cancer Wisconsin dataset (built into scikit-learn, no download needed)
2. Run GWO for 50 iterations with 10 wolves
3. Print convergence progress every 10 iterations
4. Print a final results table comparing GWO vs baselines

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population size | 10 | Number of wolves |
| Max iterations | 50 | Number of rounds |
| Classifier | KNN (k=5) | Used to evaluate feature subsets |
| Cross-validation | 5-fold | For accuracy estimation |
| Fitness weight (α) | 0.99 | Accuracy vs feature count tradeoff |
| Binarisation threshold | 0.5 | Converts continuous positions to 0/1 |
| Random seed | 42 | For reproducibility |

## References

- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. *Advances in Engineering Software*, 69, 46–61.
- Emary, E., Zawbaa, H. M., & Hassanien, A. E. (2016). Binary grey wolf optimization approaches for feature selection. *Neurocomputing*, 172, 371–381.

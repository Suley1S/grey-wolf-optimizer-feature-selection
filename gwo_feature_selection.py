import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import random
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────
NUM_WOLVES   = 10      # population size
MAX_ITER     = 50      # iterations
K_FOLDS      = 5       # cross-validation folds
ALPHA_WEIGHT = 0.99    # weight for accuracy vs feature count
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
data     = load_breast_cancer()
X, y     = data.data, data.target
n_features = X.shape[1]
feature_names = data.feature_names

print(f"Dataset: Breast Cancer Wisconsin")
print(f"Samples: {X.shape[0]}  |  Features: {n_features}  |  Classes: 2\n")

# ─────────────────────────────────────────
# FITNESS FUNCTION
# ─────────────────────────────────────────
def fitness(position):
    selected = np.where(position > 0.5)[0]
    if len(selected) == 0:
        return 1.0   # worst possible (no features selected)
    
    clf = KNeighborsClassifier(n_neighbors=5)
    acc = cross_val_score(clf, X[:, selected], y, cv=K_FOLDS).mean()
    
    # balance accuracy vs fewer features
    error_rate   = 1 - acc
    feature_ratio = len(selected) / n_features
    return ALPHA_WEIGHT * error_rate + (1 - ALPHA_WEIGHT) * feature_ratio

# ─────────────────────────────────────────
# GREY WOLF OPTIMIZER
# ─────────────────────────────────────────
def gwo():
    # initialise wolves with random positions in [0,1]
    wolves = np.random.rand(NUM_WOLVES, n_features)
    
    alpha_pos = np.zeros(n_features); alpha_score = float('inf')
    beta_pos  = np.zeros(n_features); beta_score  = float('inf')
    delta_pos = np.zeros(n_features); delta_score = float('inf')

    history = []

    for iteration in range(MAX_ITER):
        for i in range(NUM_WOLVES):
            wolves[i] = np.clip(wolves[i], 0, 1)
            score = fitness(wolves[i])

            if score < alpha_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos,  beta_score  = alpha_pos.copy(), alpha_score
                alpha_pos, alpha_score = wolves[i].copy(), score
            elif score < beta_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos,  beta_score  = wolves[i].copy(), score
            elif score < delta_score:
                delta_pos, delta_score = wolves[i].copy(), score

        a = 2 - iteration * (2 / MAX_ITER)   # linearly decreases 2 → 0

        for i in range(NUM_WOLVES):
            for j in range(n_features):
                # Alpha contribution
                r1, r2 = random.random(), random.random()
                A1 = 2*a*r1 - a;  C1 = 2*r2
                X1 = alpha_pos[j] - A1 * abs(C1*alpha_pos[j] - wolves[i][j])

                # Beta contribution
                r1, r2 = random.random(), random.random()
                A2 = 2*a*r1 - a;  C2 = 2*r2
                X2 = beta_pos[j]  - A2 * abs(C2*beta_pos[j]  - wolves[i][j])

                # Delta contribution
                r1, r2 = random.random(), random.random()
                A3 = 2*a*r1 - a;  C3 = 2*r2
                X3 = delta_pos[j] - A3 * abs(C3*delta_pos[j] - wolves[i][j])

                wolves[i][j] = (X1 + X2 + X3) / 3

        history.append(alpha_score)
        if (iteration+1) % 10 == 0:
            selected_count = np.sum(alpha_pos > 0.5)
            print(f"  Iter {iteration+1:3d}/{MAX_ITER}  |  Best fitness: {alpha_score:.5f}  |  Features selected: {selected_count}")

    return alpha_pos, alpha_score, history

# ─────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────
def all_features_accuracy():
    clf = KNeighborsClassifier(n_neighbors=5)
    return cross_val_score(clf, X, y, cv=K_FOLDS).mean()

def random_selection_accuracy(n_selected):
    idx = np.random.choice(n_features, n_selected, replace=False)
    clf = KNeighborsClassifier(n_neighbors=5)
    return cross_val_score(clf, X[:, idx], y, cv=K_FOLDS).mean(), idx

# ─────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────
print("=" * 55)
print("Running Grey Wolf Optimizer...")
print("=" * 55)
best_pos, best_fitness, history = gwo()

selected_features = np.where(best_pos > 0.5)[0]
clf = KNeighborsClassifier(n_neighbors=5)
gwo_acc = cross_val_score(clf, X[:, selected_features], y, cv=K_FOLDS).mean()

all_acc = all_features_accuracy()
rand_acc, rand_idx = random_selection_accuracy(len(selected_features))

# ─────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────
print("\n")
print("=" * 55)
print("           RESULTS SUMMARY")
print("=" * 55)
print(f"{'Method':<25} {'Accuracy':>10} {'# Features':>12}")
print("-" * 55)
print(f"{'All Features':<25} {all_acc*100:>9.2f}% {n_features:>12}")
print(f"{'Random Selection':<25} {rand_acc*100:>9.2f}% {len(rand_idx):>12}")
print(f"{'GWO (ours)':<25} {gwo_acc*100:>9.2f}% {len(selected_features):>12}")
print("=" * 55)

print(f"\nGWO Selected {len(selected_features)} out of {n_features} features:")
for i, idx in enumerate(selected_features):
    print(f"  [{i+1}] {feature_names[idx]}")

print(f"\nFeature Reduction: {n_features} → {len(selected_features)} "
      f"({100*(1 - len(selected_features)/n_features):.1f}% reduction)")
print(f"Accuracy change vs all features: {(gwo_acc - all_acc)*100:+.2f}%")

print("\n[COPY EVERYTHING ABOVE AND PASTE IT BACK TO CLAUDE]")

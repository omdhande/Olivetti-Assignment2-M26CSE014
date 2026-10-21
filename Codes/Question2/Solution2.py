#!/usr/bin/env python3
"""
Solution2.py
Q2(a)-(d): SAMME multiclass boosting from scratch on Olivetti Faces.

Expected directory structure:
Olivetti-Assignment/
└── Codes/
    ├── Question1/
    │   ├── q1_split.npz
    │   ├── best_tree.joblib
    │   └── final_bagged_models.joblib
    └── Question2/
        └── Solution2.py

Run from Question2 with:
    MPLBACKEND=Agg python Solution2.py

The script uses the saved Q1 split and frozen Q1(a) tree. It does not use
the test set to select hyperparameters.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots without needing a Qt/Wayland window
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# COMMON SETUP: reuse the exact Q1 split and Q1(a) model
# ============================================================

HERE = Path(__file__).resolve().parent
Q1_DIR = HERE.parent / "Question1"
SEED = 26014  # numeric characters from M26CSE014
DEPTHS = [1, 2, 3, 4]
ROUND_OPTIONS = [10, 25, 50, 100]
MAX_ROUNDS = max(ROUND_OPTIONS)

data = fetch_olivetti_faces()
X = data.data
y = data.target
classes = np.unique(y)
n_classes = len(classes)

split_path = Q1_DIR / "q1_split.npz"
tree_path = Q1_DIR / "best_tree.joblib"
bag_path = Q1_DIR / "final_bagged_models.joblib"

if not split_path.exists():
    raise FileNotFoundError(
        f"Cannot find {split_path}. Run Solution1.py from the Question1 "
        "folder first so it saves q1_split.npz."
    )
if not tree_path.exists():
    raise FileNotFoundError(
        f"Cannot find {tree_path}. Run Question1/Solution1.py first."
    )

split = np.load(split_path)

# Support the key names commonly used by Q1 split-saving scripts.
def get_split_indices(npz_file, possible_names, split_name):
    for key in possible_names:
        if key in npz_file.files:
            return np.asarray(npz_file[key], dtype=int)
    raise KeyError(
        f"Could not find {split_name} indices in {split_path}. "
        f"Available keys: {npz_file.files}. Expected one of: {possible_names}"
    )

train_idx = get_split_indices(split, ("train_idx", "train_indices", "train"), "training")
val_idx = get_split_indices(split, ("val_idx", "valid_idx", "validation_idx", "val_indices", "val"), "validation")
test_idx = get_split_indices(split, ("test_idx", "test_indices", "test"), "test")

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

best_tree = joblib.load(tree_path)

print("Dataset:", X.shape)
print("Train/validation/test:", X_train.shape, X_val.shape, X_test.shape)
print("Number of classes:", n_classes)
print("Loaded Q1(a) tree:", tree_path)
print("Q1(a) tree depth:", best_tree.get_depth())
print("Q1(a) tree min_samples_leaf:", best_tree.min_samples_leaf)
print("Using saved Q1 split:", split_path)


# ============================================================
# Q2(a): ONE COMPLETE SAMME ROUND, <= 10 samples, 3 classes
# ============================================================

def question_2a():
    print("\n========== Q2(a): ONE SAMME ROUND ==========")

    # Nine samples, three classes, one feature. This is only a small
    # demonstration; it is not the Olivetti data.
    X_demo = np.array([
        [0.00], [0.10], [0.20], [0.30],
        [1.00], [1.10],
        [2.00], [2.10], [2.20]
    ])
    y_demo = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])

    N = len(y_demo)
    K = len(np.unique(y_demo))
    w = np.full(N, 1.0 / N)

    print("Initial weights:", np.round(w, 6))

    weak = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    weak.fit(X_demo, y_demo, sample_weight=w)
    pred = weak.predict(X_demo)

    incorrect = (pred != y_demo)
    error = float(np.sum(w * incorrect))

    print("True labels:       ", y_demo)
    print("Weak learner output:", pred)
    print("Incorrect flags:   ", incorrect.astype(int))
    print("Weighted error:", error)
    print("Random multiclass threshold:", 1.0 - 1.0 / K)

    # A weak learner must be strictly better than random guessing.
    if error >= 1.0 - 1.0 / K:
        print("This stump is not better than random; no update is performed.")
        print("Try a different demo sample arrangement or stump depth.")
        return

    # SAMME alpha = log((1-error)/error) + log(K-1)
    # Clipping only avoids a numerical log(0) if error is exactly zero.
    error_for_log = max(error, 1e-12)
    alpha = np.log((1.0 - error_for_log) / error_for_log) + np.log(K - 1)

    print("Learner weight alpha:", alpha)

    # Same update rule used in Q2(b): increase weights of mistakes.
    w = w * np.exp(alpha * incorrect)
    w = w / np.sum(w)

    print("Updated normalized weights:", np.round(w, 6))
    print("Sum of updated weights:", np.sum(w))


# ============================================================
# SAMME implementation from scratch for Q2(b)-(d)
# ============================================================

class SAMME:
    def __init__(self, max_depth, n_estimators, seed=SEED):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.seed = seed
        self.classes_ = None
        self.models = []
        self.alphas = []
        self.weighted_errors = []
        self.max_sample_weights = []
        self.top10_weight_sums = []
        self.cumulative_training_seconds = []

    def fit(self, X_fit, y_fit):
        start = time.perf_counter()
        n = len(y_fit)
        self.classes_ = np.unique(y_fit)
        K = len(self.classes_)
        weights = np.full(n, 1.0 / n)

        for round_index in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.seed + round_index
            )
            tree.fit(X_fit, y_fit, sample_weight=weights)
            predictions = tree.predict(X_fit)
            incorrect = (predictions != y_fit)
            error = float(np.sum(weights * incorrect))

            # SAMME cannot use a learner that is no better than random.
            if error >= 1.0 - 1.0 / K:
                print(
                    f"  depth={self.max_depth}: stopping at round "
                    f"{round_index + 1}; weighted error {error:.6f} "
                    "is not better than random."
                )
                break

            error_for_log = max(error, 1e-12)
            alpha = (
                np.log((1.0 - error_for_log) / error_for_log)
                + np.log(K - 1)
            )

            self.models.append(tree)
            self.alphas.append(float(alpha))
            self.weighted_errors.append(error)

            # SAMME sample-weight update; normalize after updating.
            weights *= np.exp(alpha * incorrect)
            weights /= np.sum(weights)

            self.max_sample_weights.append(float(np.max(weights)))
            self.top10_weight_sums.append(
                float(np.sum(np.sort(weights)[-min(10, n):]))
            )
            self.cumulative_training_seconds.append(time.perf_counter() - start)

            # Perfect fit: extra rounds are unnecessary.
            if error <= 1e-12:
                break

        return self

    def predict_prefix(self, X_eval, rounds=None):
        if not self.models:
            raise ValueError("No valid SAMME weak learners were trained.")

        count = len(self.models) if rounds is None else min(rounds, len(self.models))
        if count == 0:
            raise ValueError("Requested zero boosting rounds.")

        scores = np.zeros((len(X_eval), len(self.classes_)), dtype=float)
        for tree, alpha in zip(self.models[:count], self.alphas[:count]):
            pred = tree.predict(X_eval)
            for class_index, class_label in enumerate(self.classes_):
                scores[:, class_index] += alpha * (pred == class_label)

        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba_vote(self, X_eval, rounds=None):
        """Normalized weighted votes q_k(x) from the assignment."""
        count = len(self.models) if rounds is None else min(rounds, len(self.models))
        if count == 0:
            raise ValueError("No learners available.")

        scores = np.zeros((len(X_eval), len(self.classes_)), dtype=float)
        for tree, alpha in zip(self.models[:count], self.alphas[:count]):
            pred = tree.predict(X_eval)
            for class_index, class_label in enumerate(self.classes_):
                scores[:, class_index] += alpha * (pred == class_label)

        total_alpha = np.sum(self.alphas[:count])
        if total_alpha <= 0:
            return np.full_like(scores, 1.0 / len(self.classes_))
        return scores / total_alpha


# ============================================================
# Q2(b): tune depth and rounds using clean validation accuracy only
# ============================================================

def question_2b():
    print("\n========== Q2(b): SAMME ON OLIVETTI ==========")
    all_depth_models = {}
    candidate_rows = []
    best = None

    for depth in DEPTHS:
        print(f"\nTraining SAMME with max_depth={depth}, up to {MAX_ROUNDS} rounds")
        model = SAMME(depth, MAX_ROUNDS, seed=SEED).fit(X_train, y_train)
        all_depth_models[depth] = model

        for requested_rounds in ROUND_OPTIONS:
            actual_rounds = min(requested_rounds, len(model.models))
            if actual_rounds == 0:
                continue

            train_pred = model.predict_prefix(X_train, actual_rounds)
            val_pred = model.predict_prefix(X_val, actual_rounds)
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            elapsed = model.cumulative_training_seconds[actual_rounds - 1]

            row = {
                "depth": depth,
                "requested_rounds": requested_rounds,
                "actual_rounds": actual_rounds,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "training_seconds": elapsed
            }
            candidate_rows.append(row)
            print(
                f"depth={depth:>2}, requested={requested_rounds:>3}, "
                f"actual={actual_rounds:>3}, train={train_acc:.4f}, "
                f"validation={val_acc:.4f}, time={elapsed:.2f}s"
            )

            # Primary selection criterion is validation accuracy.
            # Ties prefer fewer learners, then a shallower tree.
            key = (val_acc, -actual_rounds, -depth)
            if best is None or key > best[0]:
                best = (key, depth, actual_rounds, model)

    if best is None:
        raise RuntimeError("No valid SAMME model was trained.")

    _, selected_depth, selected_rounds, _selected_full_model = best
    selected_model = SAMME(
        selected_depth, selected_rounds, seed=SEED
    ).fit(X_train, y_train)

    val_accuracy = accuracy_score(
        y_val, selected_model.predict_prefix(X_val)
    )
    print("\nSelected model (validation accuracy only)")
    print("Selected depth:", selected_depth)
    print("Selected requested/actual rounds:", selected_rounds)
    print("Validation accuracy:", val_accuracy)

    # Save the selected boosting model and tuning table.
    joblib.dump(selected_model, HERE / "q2_selected_samme.joblib")
    joblib.dump(candidate_rows, HERE / "q2b_tuning_results.joblib")

    # Required plots for the selected model.
    r = np.arange(1, len(selected_model.models) + 1)
    plots = [
        ("weighted_training_error", selected_model.weighted_errors,
         "Weighted training error", "q2b_weighted_training_error.png"),
        ("learner_weight_alpha", selected_model.alphas,
         "Learner weight (alpha)", "q2b_learner_weight.png"),
        ("maximum_sample_weight", selected_model.max_sample_weights,
         "Maximum normalized sample weight", "q2b_max_sample_weight.png"),
        ("top10_weight_sum", selected_model.top10_weight_sums,
         "Sum of 10 largest sample weights", "q2b_top10_sample_weights.png")
    ]

    for _, values, ylabel, filename in plots:
        plt.figure(figsize=(7, 4))
        plt.plot(r, values, marker="o", markersize=3)
        plt.xlabel("Boosting round")
        plt.ylabel(ylabel)
        plt.title(f"Q2(b): {ylabel}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(HERE / filename, dpi=300, bbox_inches="tight")
        plt.close()

    return selected_model, candidate_rows


# ============================================================
# Q2(c): compare boosting and bagging with 25, 50, 100 learners
# ============================================================

def majority_vote(predictions, class_labels):
    # predictions shape: (number_of_learners, number_of_examples)
    n_examples = predictions.shape[1]
    votes = np.zeros((n_examples, len(class_labels)), dtype=int)
    class_to_col = {label: i for i, label in enumerate(class_labels)}

    for learner_predictions in predictions:
        for row, label in enumerate(learner_predictions):
            votes[row, class_to_col[label]] += 1

    chosen_columns = np.argmax(votes, axis=1)
    final_predictions = np.asarray(class_labels)[chosen_columns]
    confidence = np.max(votes, axis=1) / predictions.shape[0]
    return final_predictions, confidence, votes


def train_bagging_for_q2():
    """Train 100 bootstrap trees using the frozen Q1(a) tree settings.

    This measures bagging training time for Q2(c). The Q1(a) selected
    max_depth/min_samples_leaf and the saved Q1 split are reused.
    """
    depth = best_tree.max_depth
    leaf = best_tree.min_samples_leaf
    rng = np.random.default_rng(SEED + 1)
    n = len(X_train)
    models = []
    cumulative_times = []
    start = time.perf_counter()

    print(
        "\nTraining 100 comparison bagging trees using Q1(a) settings "
               f"(max_depth={depth}, min_samples_leaf={leaf})..."
    )
    for i in range(100):
        indices = rng.choice(n, size=n, replace=True)
        tree = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=SEED + i
        )
        tree.fit(X_train[indices], y_train[indices])
        models.append(tree)
        cumulative_times.append(time.perf_counter() - start)

    joblib.dump(models, HERE / "q2c_bagging_models.joblib")
    return models, cumulative_times


def question_2c(selected_model):
    print("\n========== Q2(c): BOOSTING VS BAGGING ==========")
    checkpoints = [25, 50, 100]

    # Train a comparable bagging ensemble with the frozen Q1(a) settings,
    # and measure its training time for the Q2(c) table.
    bag_models, bag_times = train_bagging_for_q2()

    # For boosting timing, train the selected depth up to 100 rounds.
    # The selected model may have fewer rounds; use its actual count if it stopped.
    boost_depth = selected_model.max_depth
    boost_full = SAMME(boost_depth, 100, seed=SEED).fit(X_train, y_train)

    rows = []
    print(
        f"{'Model':<10} {'Learners':>8} {'Actual':>8} "
        f"{'Train':>9} {'Validation':>12} {'Clean test':>11} {'Train time':>12}"
    )

    for count in checkpoints:
        # Bagging predictions from the first count trees.
        actual_bag = min(count, len(bag_models))
        if actual_bag:
            bag_train_stack = np.array([m.predict(X_train) for m in bag_models[:actual_bag]])
            bag_val_stack = np.array([m.predict(X_val) for m in bag_models[:actual_bag]])
            bag_test_stack = np.array([m.predict(X_test) for m in bag_models[:actual_bag]])

            bag_train_pred, _, _ = majority_vote(bag_train_stack, classes)
            bag_val_pred, _, _ = majority_vote(bag_val_stack, classes)
            bag_test_pred, _, _ = majority_vote(bag_test_stack, classes)

            bag_train_acc = accuracy_score(y_train, bag_train_pred)
            bag_val_acc = accuracy_score(y_val, bag_val_pred)
            bag_test_acc = accuracy_score(y_test, bag_test_pred)

            bag_time_text = f"{bag_times[actual_bag - 1]:.2f}s"

            row = {
                "model": "Bagging", "requested": count, "actual": actual_bag,
                "train_accuracy": bag_train_acc, "validation_accuracy": bag_val_acc,
                "clean_test_accuracy": bag_test_acc, "training_time": bag_time_text
            }
            rows.append(row)
            print(
                f"{'Bagging':<10} {count:8d} {actual_bag:8d} "
                f"{bag_train_acc:9.4f} {bag_val_acc:12.4f} "
                f"{bag_test_acc:11.4f} {bag_time_text:>12}"
            )

        # Boosting uses weighted voting and selected depth.
        actual_boost = min(count, len(boost_full.models))
        if actual_boost:
            tr = boost_full.predict_prefix(X_train, actual_boost)
            va = boost_full.predict_prefix(X_val, actual_boost)
            te = boost_full.predict_prefix(X_test, actual_boost)
            bt = accuracy_score(y_train, tr)
            bv = accuracy_score(y_val, va)
            bte = accuracy_score(y_test, te)
            btime = boost_full.cumulative_training_seconds[actual_boost - 1]

            row = {
                "model": "Boosting", "requested": count, "actual": actual_boost,
                "train_accuracy": bt, "validation_accuracy": bv,
                "clean_test_accuracy": bte, "training_time": btime
            }
            rows.append(row)
            print(
                f"{'Boosting':<10} {count:8d} {actual_boost:8d} "
                f"{bt:9.4f} {bv:12.4f} {bte:11.4f} {btime:11.2f}s"
            )

    joblib.dump(rows, HERE / "q2c_comparison_results.joblib")
    return bag_models, rows


# ============================================================
# Q2(d): evaluate tree, final bagging and boosting on clean/masked test
# ============================================================

def question_2d(selected_model, bag_models):
    print("\n========== Q2(d): CLEAN VS MASKED TEST ==========")

    # Mask only the test images. Rows and columns 24 through 39 inclusive.
    X_test_images = X_test.reshape(-1, 64, 64).copy()
    X_test_images[:, 24:40, 24:40] = 0.0
    X_test_masked = X_test_images.reshape(len(X_test), -1)

    # "Final bagged model": use the Q1-selected B when the Q1 results exist;
    # otherwise use 100 trees from the bagging models available here.
    final_B = 100
    q1c_results_path = Q1_DIR / "q1c_results.joblib"
    if q1c_results_path.exists():
        q1c_results = joblib.load(q1c_results_path)
        if isinstance(q1c_results, dict):
            # Some scripts save a dictionary containing a results list.
            for container_key in ("results", "rows", "metrics", "q1c_results"):
                if container_key in q1c_results and isinstance(q1c_results[container_key], (list, tuple)):
                    q1c_results = q1c_results[container_key]
                    break
        if isinstance(q1c_results, (list, tuple)) and q1c_results:
            def get_result_value(row, names):
                if not isinstance(row, dict):
                    return None
                for name in names:
                    if name in row:
                        return row[name]
                return None

            valid_rows = []
            for row in q1c_results:
                b_value = get_result_value(row, ("B", "b", "n_estimators", "num_trees"))
                val_value = get_result_value(
                    row,
                    ("Validation accuracy", "validation_accuracy", "val_accuracy",
                     "Validation", "validation", "val_acc")
                )
                if b_value is not None and val_value is not None:
                    valid_rows.append((float(val_value), -int(b_value), int(b_value)))
            if valid_rows:
                # Highest validation accuracy; ties choose the smaller ensemble.
                final_B = max(valid_rows)[2]
            else:
                print(
                    "Warning: could not identify B and validation-accuracy columns "
                    "in q1c_results.joblib; using B=100 for Q2(d)."
                )
    # Prefer the actual Q1-selected final bagged model for Q2(d).
    # If it is unavailable/too small, fall back to the Q2(c) comparison ensemble.
    if bag_path.exists():
        q1_final_bag = joblib.load(bag_path)
        if len(q1_final_bag) >= final_B:
            final_bag = q1_final_bag[:final_B]
            print(f"Using Q1 saved final bagged model for Q2(d), B={final_B}.")
        else:
            final_B = min(final_B, len(bag_models))
            final_bag = bag_models[:final_B]
            print("Q1 saved bagged model was too small; using Q2(c) bagging models.")
    else:
        final_B = min(final_B, len(bag_models))
        final_bag = bag_models[:final_B]
        print("Q1 saved bagged model not found; using Q2(c) bagging models.")

    if not final_bag:
        raise RuntimeError(
            "No bagging trees are available for Q2(d). Check "
            "final_bagged_models.joblib or q2c_bagging_models.joblib."
        )

    # Baseline tree confidence = max predicted class probability.
    def tree_result(X_eval):
        pred = best_tree.predict(X_eval)
        confidence = np.max(best_tree.predict_proba(X_eval), axis=1)
        return pred, confidence

    def bag_result(X_eval):
        stack = np.array([model.predict(X_eval) for model in final_bag])
        return majority_vote(stack, classes)

    def boost_result(X_eval):
        pred = selected_model.predict_prefix(X_eval)
        probs = selected_model.predict_proba_vote(X_eval)
        confidence = np.max(probs, axis=1)
        return pred, confidence

    result_sets = {}
    for name, X_eval in [("Clean", X_test), ("Masked", X_test_masked)]:
        tp, tc = tree_result(X_eval)
        bp, bc, bvotes = bag_result(X_eval)
        sp, sc = boost_result(X_eval)

        result_sets[name] = {
            "tree_pred": tp, "tree_conf": tc,
            "bag_pred": bp, "bag_conf": bc, "bag_votes": bvotes,
            "boost_pred": sp, "boost_conf": sc
        }

        print(f"\n{name} test set")
        print(f"Single tree accuracy: {accuracy_score(y_test, tp):.4f}")
        print(f"Bagging accuracy (B={final_B}): {accuracy_score(y_test, bp):.4f}")
        print(
            f"Boosting accuracy (depth={selected_model.max_depth}, "
            f"rounds={len(selected_model.models)}): "
            f"{accuracy_score(y_test, sp):.4f}"
        )
        print(f"Mean tree confidence: {np.mean(tc):.4f}")
        print(f"Mean bagging winning-vote fraction: {np.mean(bc):.4f}")
        print(f"Mean boosting normalized weighted-vote confidence: {np.mean(sc):.4f}")

    # Find a face where bagging and boosting respond differently to masking.
    clean = result_sets["Clean"]
    masked = result_sets["Masked"]
    changed = []
    for i in range(len(y_test)):
        bag_changed = clean["bag_pred"][i] != masked["bag_pred"][i]
        boost_changed = clean["boost_pred"][i] != masked["boost_pred"][i]
        prediction_disagrees = masked["bag_pred"][i] != masked["boost_pred"][i]
        if bag_changed != boost_changed or prediction_disagrees:
            changed.append(i)

    print("\nFace showing a different bagging/boosting response:")
    if changed:
        i = changed[0]
        print("Test-set position:", i)
        print("Actual identity:", int(y_test[i]))
        for name in ("Clean", "Masked"):
            res = result_sets[name]
            print(
                f"{name}: tree={int(res['tree_pred'][i])} "
                f"(conf={res['tree_conf'][i]:.4f}); "
                f"bagging={int(res['bag_pred'][i])} "
                f"(conf={res['bag_conf'][i]:.4f}); "
                f"boosting={int(res['boost_pred'][i])} "
                f"(conf={res['boost_conf'][i]:.4f})"
            )
            print("Bagging vote counts by class:")
            for class_index in np.flatnonzero(res["bag_votes"][i]):
                print(
                    f"  identity {int(classes[class_index])}: "
                    f"{int(res['bag_votes'][i, class_index])} votes"
                )
        print(
            "Interpretation: bagging combines equal votes from bootstrap trees; "
            "boosting weights each tree by alpha and emphasizes examples that "
            "earlier learners misclassified. Therefore the two ensembles can "
            "react differently when the central image region is removed."
        )
    else:
        print(
            "No test face produced a different bagging/boosting response "
            "under this split and these fitted models."
        )

    joblib.dump(result_sets, HERE / "q2d_clean_masked_results.joblib")
    np.savez(
        HERE / "q2d_masked_test.npz",
        X_test_masked=X_test_masked,
        y_test=y_test
    )

# ============================================================
# MAIN: this block runs when you type `python solution2.py`
# ============================================================

def main():
    print("Starting Q2(a)-(d) SAMME assignment...", flush=True)
    print("Output directory:", HERE, flush=True)

    print("\n[1/4] Running Q2(a)...", flush=True)
    question_2a()

    print("\n[2/4] Running Q2(b)...", flush=True)
    selected_samme, _ = question_2b()

    print("\n[3/4] Running Q2(c)...", flush=True)
    bag_models, _ = question_2c(selected_samme)

    print("\n[4/4] Running Q2(d)...", flush=True)
    question_2d(selected_samme, bag_models)

    print("\nAll Q2 sections completed successfully.", flush=True)
    print("Results and plots saved in:", HERE, flush=True)


if __name__ == "__main__":
    main()

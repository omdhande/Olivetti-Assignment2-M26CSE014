
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# COMMON SETUP
# ============================================================

data = fetch_olivetti_faces()

X = data.data
y = data.target

roll_no = "M26CSE014"
seed = int("".join(c for c in roll_no if c.isdigit()))

rng_split = np.random.default_rng(seed)

train_idx = []
val_idx = []
test_idx = []

for person in np.unique(y):
    indices = np.where(y == person)[0]
    indices = rng_split.permutation(indices)

    train_idx.extend(indices[:6])
    val_idx.extend(indices[6:8])
    test_idx.extend(indices[8:10])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

np.savez(
    "q1_split.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

n = len(X_train)
n_classes = len(np.unique(y_train))

B_values = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_values)

print("Dataset:", X.shape)
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Classes:", n_classes)
print("Seed:", seed)


# ============================================================
# Q1(a): DECISION TREE BASELINE
# ============================================================

print("\n========== Q1(a): DECISION TREE ==========")

tree_results = []
best_model = None
best_val_acc = -1.0

for depth in [2, 4, 8, 16, None]:
    for leaf in [1, 2, 4]:

        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=seed
        )

        model.fit(X_train, y_train)

        train_acc = accuracy_score(
            y_train, model.predict(X_train)
        )

        val_acc = accuracy_score(
            y_val, model.predict(X_val)
        )

        test_acc = accuracy_score(
            y_test, model.predict(X_test)
        )

        leaf_ids = model.apply(X_train)
        _, counts = np.unique(
            leaf_ids, return_counts=True
        )

        num_leaves = model.get_n_leaves()
        mean_samples = np.mean(counts)
        fraction_single = np.mean(counts == 1)

        tree_results.append({
            "depth": depth,
            "min_samples_leaf": leaf,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "test_accuracy": test_acc,
            "leaves": num_leaves,
            "mean_samples_per_leaf": mean_samples,
            "fraction_single_example": fraction_single
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

print(
    f"{'Depth':>8} {'Leaf':>6} {'Train':>9} "
    f"{'Validation':>12} {'Test':>9} {'Leaves':>8} "
    f"{'Mean/leaf':>11} {'Single frac':>12}"
)

for r in tree_results:
    print(
        f"{str(r['depth']):>8} "
        f"{r['min_samples_leaf']:6d} "
        f"{r['train_accuracy']:9.4f} "
        f"{r['validation_accuracy']:12.4f} "
        f"{r['test_accuracy']:9.4f} "
        f"{r['leaves']:8d} "
        f"{r['mean_samples_per_leaf']:11.3f} "
        f"{r['fraction_single_example']:12.4f}"
    )

joblib.dump(best_model, "best_tree.joblib")

best_depth = best_model.max_depth
best_leaf = best_model.min_samples_leaf

print("\nSelected tree")
print("Depth:", best_depth)
print("Minimum samples per leaf:", best_leaf)
print("Validation accuracy:", best_val_acc)
print(
    "Test accuracy:",
    accuracy_score(y_test, best_model.predict(X_test))
)


# ============================================================
# TRAIN 200 BOOTSTRAP TREES ONCE
# ============================================================

print("\nTraining bootstrap trees...")

rng_bag = np.random.default_rng(seed + 1)

models = []
inbag_masks = []
distinct_per_tree = []

for i in range(max_B):

    indices = rng_bag.choice(
        n, size=n, replace=True
    )

    inbag = np.zeros(n, dtype=bool)
    inbag[indices] = True

    X_boot = X_train[indices]
    y_boot = y_train[indices]

    tree = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
        random_state=seed + i
    )

    tree.fit(X_boot, y_boot)

    models.append(tree)
    inbag_masks.append(inbag)
    distinct_per_tree.append(len(np.unique(indices)))


# ============================================================
# COMMON MAJORITY VOTING FUNCTION
# ============================================================

def majority_vote(predictions):
    vote_counts = np.zeros(
        (predictions.shape[1], n_classes),
        dtype=int
    )

    for j in range(predictions.shape[1]):
        labels, counts = np.unique(
            predictions[:, j],
            return_counts=True
        )

        vote_counts[j, labels.astype(int)] = counts

    final_predictions = np.argmax(
        vote_counts, axis=1
    )

    confidence = (
        np.max(vote_counts, axis=1)
        / predictions.shape[0]
    )

    return final_predictions, confidence, vote_counts


# Cache predictions for training and validation sets
train_predictions = np.array([
    tree.predict(X_train) for tree in models
])

val_predictions = np.array([
    tree.predict(X_val) for tree in models
])


# ============================================================
# Q1(b): BAGGING FROM SCRATCH
# ============================================================

print("\n========== Q1(b): BAGGING ==========")

results_b = []

for B in B_values:

    train_pred, _, _ = majority_vote(
        train_predictions[:B]
    )

    val_pred, _, _ = majority_vote(
        val_predictions[:B]
    )

    test_tree_predictions = np.array([
        tree.predict(X_test)
        for tree in models[:B]
    ])

    test_pred, _, _ = majority_vote(
        test_tree_predictions
    )

    mean_distinct = np.mean(
        distinct_per_tree[:B]
    )

    result = {
        "B": B,
        "Train accuracy": accuracy_score(
            y_train, train_pred
        ),
        "Validation accuracy": accuracy_score(
            y_val, val_pred
        ),
        "Test accuracy": accuracy_score(
            y_test, test_pred
        ),
        "Average distinct examples": mean_distinct
    }

    results_b.append(result)

print(
    f"{'B':>5} {'Train':>10} {'Validation':>12} "
    f"{'Test':>10} {'Distinct':>12}"
)

for r in results_b:
    print(
        f"{r['B']:5d} "
        f"{r['Train accuracy']:10.4f} "
        f"{r['Validation accuracy']:12.4f} "
        f"{r['Test accuracy']:10.4f} "
        f"{r['Average distinct examples']:12.3f}"
    )

expected_distinct = n * (
    1 - (1 - 1 / n) ** n
)

print("\nTheoretical distinct examples:", expected_distinct)
print("Theoretical fraction:", expected_distinct / n)


# ============================================================
# Q1(c): OOB ACCURACY, DISAGREEMENT, VOTE MARGIN
# ============================================================

print("\n========== Q1(c): ENSEMBLE METRICS ==========")

results_c = []

for B in B_values:

    pred_val = val_predictions[:B]
    pred_train = train_predictions[:B]

    val_bag_pred, _, val_votes = majority_vote(
        pred_val
    )

    val_acc = accuracy_score(
        y_val, val_bag_pred
    )

    # Out-of-bag voting
    oob_votes = np.zeros(
        (n, n_classes), dtype=int
    )

    for i in range(B):

        oob_indices = np.where(
            ~inbag_masks[i]
        )[0]

        if len(oob_indices) == 0:
            continue

        oob_pred = models[i].predict(
            X_train[oob_indices]
        )

        for idx, pred in zip(oob_indices, oob_pred):
            oob_votes[idx, int(pred)] += 1

    has_oob_vote = oob_votes.sum(axis=1) > 0

    if np.any(has_oob_vote):
        oob_pred = np.argmax(
            oob_votes[has_oob_vote], axis=1
        )

        oob_acc = accuracy_score(
            y_train[has_oob_vote], oob_pred
        )
    else:
        oob_acc = np.nan

    # Average pairwise disagreement
    disagreements = []

    for i in range(B):
        for j in range(i + 1, B):
            disagreements.append(
                np.mean(pred_val[i] != pred_val[j])
            )

    mean_disagreement = (
        np.mean(disagreements)
        if disagreements else 0.0
    )

    # Mean vote margin
    sorted_votes = np.sort(
        val_votes, axis=1
    )[:, ::-1]

    winning_fraction = sorted_votes[:, 0] / B
    runner_up_fraction = sorted_votes[:, 1] / B

    mean_margin = np.mean(
        winning_fraction - runner_up_fraction
    )

    train_bag_pred, _, _ = majority_vote(
        pred_train
    )

    train_acc = accuracy_score(
        y_train, train_bag_pred
    )

    results_c.append({
        "B": B,
        "Train accuracy": train_acc,
        "Validation accuracy": val_acc,
        "OOB accuracy": oob_acc,
        "Pairwise disagreement": mean_disagreement,
        "Mean vote margin": mean_margin
    })

print(
    f"{'B':>5} {'Train':>10} {'Validation':>12} "
    f"{'OOB':>10} {'Disagreement':>14} {'Margin':>10}"
)

for r in results_c:
    print(
        f"{r['B']:5d} "
        f"{r['Train accuracy']:10.4f} "
        f"{r['Validation accuracy']:12.4f} "
        f"{r['OOB accuracy']:10.4f} "
        f"{r['Pairwise disagreement']:14.4f} "
        f"{r['Mean vote margin']:10.4f}"
    )

# Select ensemble size using validation accuracy only.
# If tied, select the smaller ensemble.
best_result = max(
    results_c,
    key=lambda r: (
        r["Validation accuracy"], -r["B"]
    )
)

final_B = best_result["B"]
final_models = models[:final_B]

print("\nSelected final B:", final_B)
print(
    "Validation accuracy:",
    best_result["Validation accuracy"]
)

joblib.dump(final_models, "final_bagged_models.joblib")
joblib.dump(results_b, "q1b_results.joblib")
joblib.dump(results_c, "q1c_results.joblib")

# Plot metrics
Bs = [r["B"] for r in results_c]

fig, axes = plt.subplots(
    1, 3, figsize=(16, 4.5)
)

axes[0].plot(
    Bs, [r["OOB accuracy"] for r in results_c],
    marker="o"
)
axes[0].set_title("Out-of-bag accuracy")
axes[0].set_xlabel("Number of trees (B)")
axes[0].set_ylabel("OOB accuracy")
axes[0].set_xticks(Bs)
axes[0].grid(True)

axes[1].plot(
    Bs,
    [r["Pairwise disagreement"] for r in results_c],
    marker="o"
)
axes[1].set_title("Average pairwise disagreement")
axes[1].set_xlabel("Number of trees (B)")
axes[1].set_ylabel("Disagreement")
axes[1].set_xticks(Bs)
axes[1].grid(True)

axes[2].plot(
    Bs, [r["Mean vote margin"] for r in results_c],
    marker="o"
)
axes[2].set_title("Mean vote margin")
axes[2].set_xlabel("Number of trees (B)")
axes[2].set_ylabel("Mean vote margin")
axes[2].set_xticks(Bs)
axes[2].grid(True)

plt.tight_layout()
plt.savefig(
    "q1c_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# Q1(d): CLEAN AND MASKED TEST SETS
# ============================================================

print("\n========== Q1(d): CLEAN AND MASKED TEST ==========")

# Create masked test images only.
# Rows 24-39 and columns 24-39 are set to zero.
X_test_images = X_test.reshape(-1, 64, 64)

X_test_masked_images = X_test_images.copy()
X_test_masked_images[:, 24:40, 24:40] = 0

X_test_masked = X_test_masked_images.reshape(
    len(X_test), -1
)


def evaluate_models(X_eval, y_eval, name):

    # Single tree prediction and confidence
    tree_pred = best_model.predict(X_eval)

    tree_probabilities = best_model.predict_proba(
        X_eval
    )

    tree_confidence = np.max(
        tree_probabilities, axis=1
    )

    # Bagging predictions and vote counts
    ensemble_predictions = np.array([
        tree.predict(X_eval)
        for tree in final_models
    ])

    bag_pred, bag_confidence, vote_counts = majority_vote(
        ensemble_predictions
    )

    print("\n", name)
    print(
        "Single tree accuracy:",
        accuracy_score(y_eval, tree_pred)
    )
    print(
        "Bagging accuracy:",
        accuracy_score(y_eval, bag_pred)
    )
    print(
        "Mean single tree confidence:",
        np.mean(tree_confidence)
    )
    print(
        "Mean bagging vote confidence:",
        np.mean(bag_confidence)
    )

    return {
        "tree_predictions": tree_pred,
        "tree_confidence": tree_confidence,
        "bag_predictions": bag_pred,
        "bag_confidence": bag_confidence,
        "vote_counts": vote_counts
    }


clean_results = evaluate_models(
    X_test, y_test, "CLEAN TEST SET"
)

masked_results = evaluate_models(
    X_test_masked, y_test, "MASKED TEST SET"
)


# ============================================================
# FIND TWO CASES BAGGING CORRECTS AND TWO BOTH GET WRONG
# ============================================================

corrected_cases = []
both_wrong_cases = []

for dataset_name, result in [
    ("Clean", clean_results),
    ("Masked", masked_results)
]:

    for i in range(len(y_test)):

        actual = y_test[i]
        tree_pred = result["tree_predictions"][i]
        bag_pred = result["bag_predictions"][i]

        tree_correct = tree_pred == actual
        bag_correct = bag_pred == actual

        if not tree_correct and bag_correct:
            corrected_cases.append(
                (dataset_name, i)
            )

        if not tree_correct and not bag_correct:
            both_wrong_cases.append(
                (dataset_name, i)
            )


def select_distinct_cases(cases, count=2):

    selected = []
    used_cases = set()

    for dataset_name, i in cases:

        key = (dataset_name, i)

        if key not in used_cases:
            selected.append(key)
            used_cases.add(key)

        if len(selected) == count:
            break

    return selected


corrected_cases = select_distinct_cases(
    corrected_cases, 2
)

both_wrong_cases = select_distinct_cases(
    both_wrong_cases, 2
)


def show_case(dataset_name, i):

    result = (
        clean_results
        if dataset_name == "Clean"
        else masked_results
    )

    actual = int(y_test[i])
    tree_pred = int(
        result["tree_predictions"][i]
    )
    bag_pred = int(
        result["bag_predictions"][i]
    )

    print("\n------------------------------")
    print("Dataset:", dataset_name)
    print("Test image index:", i)
    print("Actual identity:", actual)

    print("Single tree prediction:", tree_pred)
    print(
        "Single tree confidence:",
        round(
            float(result["tree_confidence"][i]), 4
        )
    )

    print("Bagging prediction:", bag_pred)
    print(
        "Bagging vote confidence:",
        round(
            float(result["bag_confidence"][i]), 4
        )
    )

    print("Bagging vote counts:")

    votes = result["vote_counts"][i]

    for class_id in np.flatnonzero(votes):
        print(
            f"Identity {class_id}: "
            f"{votes[class_id]} votes"
        )


print("\nCASES WHERE BAGGING CORRECTS THE TREE")

if len(corrected_cases) < 2:
    print(
        "Only", len(corrected_cases),
        "corrected case(s) found."
    )

for dataset_name, i in corrected_cases:
    show_case(dataset_name, i)


print("\nCASES WHERE BOTH MODELS ARE WRONG")

if len(both_wrong_cases) < 2:
    print(
        "Only", len(both_wrong_cases),
        "both-wrong case(s) found."
    )

for dataset_name, i in both_wrong_cases:
    show_case(dataset_name, i)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

joblib.dump(
    {
        "clean_results": clean_results,
        "masked_results": masked_results,
        "corrected_cases": corrected_cases,
        "both_wrong_cases": both_wrong_cases,
        "final_B": final_B
    },
    "q1d_results.joblib"
)

print("\nAll Q1 sections completed.")
print("Saved files:")
print("q1_split.npz")
print("best_tree.joblib")
print("final_bagged_models.joblib")
print("q1b_results.joblib")
print("q1c_results.joblib")
print("q1c_metrics.png")
print("q1d_results.joblib")


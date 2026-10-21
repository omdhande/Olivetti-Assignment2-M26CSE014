# Olivetti Faces Classification: Decision Trees, Bagging, and AdaBoost

**Student:** Om Dinesh Dhande\
**Roll Number:** M26CSE014\
**Dataset:** Olivetti Faces (`sklearn.datasets.fetch_olivetti_faces`)

## Overview

This project studies multi-class face classification on the Olivetti
Faces dataset. It compares a single decision tree with ensemble methods
implemented for the assignment, evaluates performance on clean and
corrupted images, and measures how predictions vary across resampled
training sets.

The dataset contains 400 grayscale face images from 40 identities. Each
image is 64 × 64 pixels and is represented by 4,096 pixel features.

## Methods

-   **Single decision tree:** Select hyperparameters using validation
    accuracy, then freeze the selected configuration.
-   **Bagging:** Train trees on bootstrap samples of the original
    training set and combine their predictions.
-   **AdaBoost (SAMME):** Implement the multi-class SAMME boosting
    procedure and combine weak learners using their learned weights.
-   **Robustness evaluation:** Evaluate the trained models on the
    specified corrupted test images, in which the central 16 × 16 pixel
    block is set to zero.
-   **Bias and variance experiment:** Keep the clean test set fixed,
    create 30 within-identity resampled training sets, refit the
    selected models without tuning again, and compare their
    class-probability predictions.

All image features are raw flattened pixel values. No PCA or other
dimensionality reduction is applied.

## Dataset and split

The code uses `fetch_olivetti_faces` from scikit-learn. The assignment
requires a fixed, identity-wise split with:

-   6 training images per identity
-   2 validation images per identity
-   2 clean test images per identity

The saved split should be reused across experiments so that model
comparisons are fair. If the assignment requires `olivetti_split.npz`,
keep that file in the repository and load it rather than generating a
new split each run.

The random seed is derived from the numeric digits in the roll number
`M26CSE014`, giving a seed of `26014`.

## Requirements

-   Python 3
-   NumPy
-   scikit-learn
-   Matplotlib (if generating plots)

Install the dependencies in a virtual environment:

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scikit-learn matplotlib
```

## Repository contents

A suggested repository layout is:

``` text
Olivetti-Assignment2-M26CSE014/
├── README.md
├── solution.py
├── olivetti_split.npz
├── results/
│   ├── results.json
│   └── logs/
├── figures/
├── figure_scripts/
└── requirements.txt
```

Keep only files that are actually present in your repository. Add any
additional scripts, configuration files, logs, or plots required by your
assignment.

## Run the code

From the repository root, activate the virtual environment and run:

``` bash
python solution.py
```

If your implementation uses separate scripts for individual questions,
run the exact commands used to produce each reported result and document
them here. For reproducibility, record the Python and scikit-learn
versions, configuration values, random seed, and final Git commit hash.

## Evaluation and reported results

Record the required metrics in the result files and final report,
including the metrics requested for each question. Depending on the
experiment, these may include:

-   Training, validation, and clean test accuracy
-   Number of leaves and mean samples per leaf for the selected tree
-   Bagging accuracy as the number of estimators changes
-   Mean number of distinct examples in bootstrap samples, compared with
    its theoretical expectation
-   Robustness performance on the corrupted test set
-   Bias/variance-related measurements from the 30 resampled training
    sets
-   Plots and logs used to support the reported results

Do not put unverified or example scores in this README. Generate all
reported values from the final code run.

## Reproducibility checklist

-   [ ] Use the fixed saved train/validation/test split.
-   [ ] Derive and use the required random seed.
-   [ ] Select hyperparameters using validation data only.
-   [ ] Freeze selected hyperparameters before final test evaluation.
-   [ ] Do not tune again during the bias/variance resampling
    experiment.
-   [ ] Save configurations, logs, result files, and figures.
-   [ ] Include the exact commands required to reproduce the results.
-   [ ] Verify that the repository contains every file required for
    submission.

## Notes

This README describes the intended project workflow. Before submission,
update the repository layout and run commands to match the files that
are actually present in your GitHub repository.

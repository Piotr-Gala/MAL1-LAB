# Assignment 2: Candidate Test 2022 Analysis Part 1 - Exam Checklist Map

This file maps the official exam checklist to the Assignment 2 notebook.

Sources:

- `Materials/exam/Exam_information_and_assignments_overview.pdf`
- `Materials/exam/Exam_theory_topics.pdf`
- Notebook: `MAL1-LAB/Answers/Assignment2/Assignment2.ipynb`

---

## Assignment-Specific Checklist From Exam Overview

Official topic for Assignment 2: **Candidates I**

- Visualization (box plot)
- Descriptive analysis
- Classification algorithms
- Decision trees
- Random forests
- Gradient-boosted trees

---

## 1. Visualization: Box Plot

**Status:** covered

**Where in notebook**

- Cells 19-22: box plots for age and confidence score by party

**What happens**

- Plot age distribution by party.
- Create `confidence_score` as the proportion of strong answers, meaning answers equal to `-2` or `2`.
- Plot confidence-score distribution by party.
- Use party colors for readability.

**Why it matters**

The exam overview explicitly lists visualization with box plots for this assignment.
Box plots show median, quartiles, spread and possible outliers.

**What to say**

```text
I use box plots to compare distributions across parties.
The age plot describes candidate demographics, while the confidence-score plot describes how often candidates give strong answers.
```

**Possible question**

```text
What does a box plot show?
```

**Answer**

```text
A box plot shows the median, quartiles, spread and possible outliers of a numerical variable.
```

---

## 2. Descriptive Analysis

**Status:** covered

**Where in notebook**

- Cells 5-10: loading data, `info()`, `describe()`
- Cells 11-14: missing values, duplicate check, response range
- Cells 15-18: party counts and age summary
- Cells 19-22: descriptive box plots

**What happens**

- Load candidate datasets.
- Check shape and first rows.
- Inspect data types and summary statistics.
- Replace invalid age value `0` with missing.
- Define metadata columns and response features.
- Check duplicates.
- Confirm response range from `-2` to `2`.
- Inspect party imbalance.

**Why it matters**

Descriptive analysis explains what the data looks like before modelling.
It also reveals class imbalance, which affects metric choice.

**What to say**

```text
Before modelling, I inspect the dataset structure, missing values, response scale and party distribution.
The party counts show that the classification problem is imbalanced, so I later use macro F1 together with accuracy.
```

**Possible question**

```text
Why is descriptive analysis needed before modelling?
```

**Answer**

```text
It helps identify data issues, understand feature types and detect imbalance before choosing preprocessing, splitting and evaluation methods.
```

---

## 3. Classification Algorithms

**Status:** covered

**Where in notebook**

- Cells 23-26: classification setup
- Cells 27-29: baseline model
- Cells 30-31: shared cross-validation setup
- Cells 32-42: three classification algorithms
- Cells 43-49: comparison and final test evaluation

**What happens**

- Define target `y` as party affiliation.
- Define features `X` as the 49 candidate-test answer columns.
- Remove `Løsgænger` because it has only three observations.
- Encode party labels.
- Use stratified train-test split.
- Use cross-validation on the training data for model comparison.
- Evaluate final selected model once on the test set.

**Why it matters**

This is the supervised learning part of the assignment.
The model learns to predict party labels from answer patterns.

**What to say**

```text
This is a multiclass classification problem.
The input features are candidate-test answers and the target is party affiliation.
I use cross-validation on the training data for model comparison and keep the test set untouched until the end.
```

**Possible question**

```text
Why do you have no separate validation set?
```

**Answer**

```text
Because I use cross-validation on the training data for model comparison.
The folds play the role of validation, while the test set is still untouched and used only once at the end.
```

---

## 4. Decision Trees

**Status:** covered

**Where in notebook**

- Cells 32-35: Decision Tree model, CV scores, feature importance and tree preview

**What happens**

- Train a `DecisionTreeClassifier`.
- Limit tree complexity using `max_depth=6` and `min_samples_leaf=5`.
- Evaluate with cross-validation.
- Inspect feature importance.
- Plot only the first two tree levels.

**Why it matters**

Decision trees are explicitly listed in the exam overview.
They are interpretable, but can easily overfit if not constrained.

**What to say**

```text
A decision tree predicts by following a sequence of feature-based splits.
I limit the tree depth and minimum leaf size to reduce overfitting.
I only plot the first levels because the full multiclass tree is too large to read.
```

**Possible question**

```text
Why can decision trees overfit?
```

**Answer**

```text
Because a deep tree can learn very specific patterns in the training data, including noise.
Limiting depth and leaf size makes the tree simpler and helps generalization.
```

---

## 5. Random Forests

**Status:** covered

**Where in notebook**

- Cells 36-39: Random Forest model, CV scores, feature importance
- Cells 46-49: final model evaluation, where Random Forest is selected

**What happens**

- Train a `RandomForestClassifier`.
- Use 100 trees.
- Evaluate with stratified cross-validation.
- Inspect feature importance.
- Select Random Forest as the best model based on macro F1.
- Evaluate it on the test set.

**Why it matters**

Random forests are explicitly listed in the exam overview.
They are ensembles of decision trees and usually generalize better than a single tree.

**What to say**

```text
Random Forest combines many decision trees trained on different bootstrap samples and feature subsets.
This reduces variance compared with one tree.
In this notebook, Random Forest performs best in cross-validation and on the final test evaluation.
```

**Possible question**

```text
Why does Random Forest often perform better than one Decision Tree?
```

**Answer**

```text
One tree has high variance and can overfit.
Random Forest averages many different trees, which reduces variance and improves generalization.
```

---

## 6. Gradient-Boosted Trees

**Status:** covered

**Where in notebook**

- Cells 40-42: Gradient-Boosted Trees model and CV scores
- Cells 43-45: comparison with Decision Tree and Random Forest

**What happens**

- Train a `GradientBoostingClassifier`.
- Use `n_estimators=80`, `learning_rate=0.05` and `max_depth=3`.
- Evaluate with stratified cross-validation.
- Compare against Decision Tree and Random Forest.

**Why it matters**

Gradient-boosted trees are explicitly listed in the exam overview.
They are another tree ensemble method, but unlike Random Forest, boosting builds trees sequentially.

**What to say**

```text
Gradient Boosting builds trees one after another.
Each new tree tries to improve the errors made by the current ensemble.
It can be powerful, but it can also overfit if the model is too complex.
```

**Possible question**

```text
What is the difference between Random Forest and Gradient Boosting?
```

**Answer**

```text
Random Forest builds many trees mostly independently and averages them.
Gradient Boosting builds trees sequentially, where each tree tries to correct previous errors.
```

---

## Extra Exam-Safe Points

### Train, Validation And Test

**Status:** covered through train-test plus CV

**Where in notebook**

- Cells 23-26: train-test split
- Cells 30-45: cross-validation on training data
- Cells 46-49: final test evaluation

**What to say**

```text
I do not use a separate validation set because cross-validation on the training data is used for model comparison.
This is especially useful here because the dataset has many imbalanced party classes.
The test set is still untouched and used only once at the end.
```

### Metrics

**Status:** covered

**Where in notebook**

- Cells 27-29: baseline accuracy and macro F1
- Cells 32-45: CV accuracy and macro F1
- Cells 46-49: test accuracy, macro F1, classification report, confusion matrix

**What to say**

```text
Accuracy measures overall correctness.
Macro F1 is important because every party gets equal weight, which matters with imbalanced classes.
The confusion matrix is larger than in binary assignments because this is a multiclass problem.
```

### Limitations

**Status:** covered

**Where in notebook**

- Cells 53-54: conclusion and limitations

**What to say**

```text
The main limitations are that answers are self-reported, parties are not perfect ideology labels, classes are imbalanced and tree-based feature importance is model-specific rather than causal.
```


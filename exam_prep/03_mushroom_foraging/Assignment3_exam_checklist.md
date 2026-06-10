# Assignment 3: Mushroom Foraging - Exam Checklist Map

This file maps the official exam checklist to the Assignment 3 notebook.

Sources:

- `Materials/exam/Exam_information_and_assignments_overview.pdf`
- `Materials/exam/Exam_theory_topics.pdf`
- Notebook: `MAL1-LAB/Answers/Assignment3/Assignment3.ipynb`

---

## Assignment-Specific Checklist From Exam Overview

Official topic for Assignment 3: **Mushroom foraging**

- Data preparation
- Logistic regression
- Validation methods
- Confusion matrix
- Performance metrics for classification

---

## 1. Data Preparation

**Status:** covered

**Where in notebook**

- Cells 5-8: loading data, shape, `info()`, `describe()`
- Cells 9-12: data types, missing values, target distribution
- Cells 13-21: duplicates, `X`/`y`, split, stratification, missing-value based column removal
- Cells 23-25: preprocessing pipeline

**What happens**

- Load `secondary_data.csv`.
- Check dataset shape: `61069` rows, `21` columns.
- Check missing values.
- Check class distribution.
- Check duplicates but do not remove rows before split.
- Split into train, validation and test before fitting preprocessing.
- Drop columns with more than 80% missing values, using only training data to decide.
- Impute numerical and categorical missing values.
- Scale numerical features.
- One-hot encode categorical features.

**Why it matters**

The exam theory PDF lists data exploration, data cleaning, missing values, scaling, handling string data, one-hot encoding and preprocessing/splitting as important topics.

This notebook shows these steps in a leakage-safe order.

**What to say**

```text
I first inspected the data, checked missing values and target distribution, and then split the data before fitting preprocessing steps.
This is important because preprocessing decisions should not use validation or test data.
```

**Possible question**

```text
Why did you split before preprocessing?
```

**Answer**

```text
To avoid data leakage. If I fit an imputer, scaler, encoder or feature-selection decision on the full dataset, information from validation or test data would influence training.
```

---

## 2. Logistic Regression

**Status:** covered

**Where in notebook**

- Cells 23-25: base Logistic Regression pipeline
- Cells 26-33: Logistic Regression with different `C` values using single validation split
- Cells 34-38: `GridSearchCV` tuning of `C`
- Cells 42-49: final Logistic Regression model and test evaluation

**What happens**

- Logistic Regression is used as the classification model.
- Hyperparameter `C` is tuned.
- Best `C` is `0.1`.
- The model is evaluated on validation and final test data.

**Why it matters**

The assignment overview explicitly lists Logistic Regression for Mushroom Foraging.
The theory PDF also lists logistic regression as an exam topic: classification algorithm, probability output, sigmoid/logit, regularization and hyperparameters.

**What to say**

```text
Logistic Regression is used because this is a binary classification task and the assignment requires it.
The model estimates class probabilities and then classifies mushrooms as edible or poisonous.
I tune C because it controls regularization strength.
```

**Possible question**

```text
What does C mean in Logistic Regression?
```

**Answer**

```text
C controls regularization strength. A smaller C means stronger regularization, while a larger C means weaker regularization.
```

---

## 3. Validation Methods

**Status:** covered

The theory PDF lists:

- Purpose of validation in ML workflow
- Train-test
- Train-val-test
- Cross-validation
- Leave-one-out cross-validation
- Use cases, advantages and disadvantages
- Test set only for final evaluation
- Validation set for evaluation and hyperparameter tuning
- Preprocessing and splitting data
- Stratification
- Generalization and overfitting

---

### 3.1 Purpose Of Validation

**Where in notebook**

- Cells 16-17: train/validation/test split and distribution check
- Cells 26-33: single validation split
- Cells 34-38: Stratified K-Fold CV
- Cells 39-41: nested CV
- Cells 42-49: final test evaluation

**What to say**

```text
Validation is used to estimate how well the model generalizes and to choose hyperparameters without touching the final test set.
```

**Why it matters**

Without validation, we might choose a model that works well only on the training data.

---

### 3.2 Train-Test

**Status:** concept covered

**Where in notebook**

- Cell 16 creates `X_train_val` and `X_test`.
- Cell 43 fits the final model on `X_train_val`.
- Cells 45-49 evaluate on `X_test`.

**What to say**

```text
The test set is held out from the beginning and used only once for final evaluation.
This gives a more realistic estimate of performance on unseen data.
```

**Notebook detail**

```text
test_size=0.2
```

---

### 3.3 Train-Val-Test

**Status:** covered

**Where in notebook**

- Cell 16

**What happens**

```text
X_train_val, X_test = first split
X_train, X_val = second split
```

**What to say**

```text
I use train data to fit the model, validation data to tune hyperparameters, and test data only for final evaluation.
```

**Why it matters**

This supports hyperparameter tuning without contaminating the test set.

---

### 3.4 Cross-Validation

**Status:** covered

**Where in notebook**

- Cells 34-38

**What happens**

- `StratifiedKFold(n_splits=5)`
- `GridSearchCV`
- scoring: F1-score for poisonous class
- best `C`: `0.1`
- mean F1: about `0.830`

**What to say**

```text
Cross-validation trains and validates the model several times on different folds.
This gives a more stable estimate than one validation split.
I use stratified folds to preserve the edible and poisonous class proportions.
```

**Advantage**

```text
More stable than one split.
```

**Disadvantage**

```text
More computationally expensive because the model is trained multiple times.
```

---

### 3.5 Leave-One-Out Cross-Validation

**Status:** theory only, not used in notebook

**Where in notebook**

- Not implemented intentionally.

**Why not used**

The notebook already uses three validation methods:

- single validation split
- Stratified K-Fold cross-validation
- nested cross-validation

Leave-One-Out would train one model per observation.
With over `61000` rows, that would be computationally impractical and not useful for this exam notebook.

**What to say if asked**

```text
Leave-One-Out cross-validation uses one observation as validation data and all remaining observations as training data.
It can use almost all data for training, but it is very expensive because it needs one model fit per observation.
For this dataset with over 61000 rows, I did not use it because Stratified K-Fold gives a much better practical trade-off.
```

**Advantage**

```text
Uses almost all data for training in each iteration.
```

**Disadvantage**

```text
Very expensive and sensitive to individual observations.
```

---

### 3.6 Use Of Test Set Only For Final Evaluation

**Status:** covered

**Where in notebook**

- Cell 16: test split created
- Cells 42-49: test set used at the end

**What to say**

```text
The test set is not used for tuning or preprocessing decisions.
It is used only once at the end to estimate final generalization performance.
```

**Why it matters**

If the test set is used during model selection, it stops being a fair final evaluation.

---

### 3.7 Use Of Validation Set For Hyperparameter Tuning

**Status:** covered

**Where in notebook**

- Cells 26-33: single validation split tunes `C`
- Cells 34-38: GridSearchCV tunes `C`

**What to say**

```text
The validation set is used to compare different C values and choose the best model before final testing.
```

---

### 3.8 Preprocessing And Splitting Data

**Status:** covered strongly

**Where in notebook**

- Cell 16: split first
- Cells 18-20: missing-value based column removal using training data
- Cells 23-25: preprocessing inside pipeline

**What to say**

```text
I split the data before fitting preprocessing steps.
The pipeline fits imputation, scaling and encoding only on training data during validation, then applies the learned transformations to validation or test data.
```

**Important exam phrase**

```text
Fit on train, transform validation and test.
```

---

### 3.9 Stratification

**Status:** covered

**Where in notebook**

- Cell 16: `stratify=y` and `stratify=y_train_val`
- Cell 17: class distribution check
- Cells 34-41: `StratifiedKFold`

**What to say**

```text
Stratification keeps the class proportions similar in train, validation and test sets.
This is important because the dataset is slightly imbalanced.
```

**Notebook detail**

```text
Train, validation and test all have about 55.5% poisonous and 44.5% edible mushrooms.
```

---

### 3.10 Generalization And Overfitting

**Status:** covered

**Where in notebook**

- Cells 26-33: validation performance
- Cells 34-41: CV and nested CV estimates
- Cells 42-49: test performance
- Cells 50-51: interpretation and limitations

**What to say**

```text
Generalization means performance on unseen data.
I use validation and cross-validation to reduce the risk of choosing a model that only performs well on one split.
The final test set gives the most important estimate of generalization.
```

---

## 4. Confusion Matrix

**Status:** covered

**Where in notebook**

- Cell 30: validation confusion matrix
- Cell 45: test confusion matrix

**Classes**

```text
e = edible
p = poisonous
```

**For poisonous as positive class**

```text
                 Predicted edible     Predicted poisonous
Actual edible    TN                  FP
Actual poisonous FN                  TP
```

**Most important error**

```text
False negative: actual poisonous, predicted edible.
```

**What to say**

```text
The confusion matrix shows the types of errors, not only the total score.
In this problem, false negatives are the most dangerous because they mean poisonous mushrooms predicted as edible.
```

---

## 5. Performance Metrics For Classification

**Status:** covered

**Where in notebook**

- Cells 29 and 32: validation classification report and recall
- Cells 47-49: ROC curve, precision-recall curve, accuracy, precision, recall, F1, ROC AUC
- Cell 50: final interpretation

**Final test results**

```text
Accuracy: 0.8167
Precision for poisonous: 0.8422
Recall for poisonous: 0.8241
F1 for poisonous: 0.8330
ROC AUC: 0.8816
```

**Accuracy**

```text
accuracy = (TP + TN) / all predictions
```

Measures overall correctness.

**Precision**

```text
precision = TP / (TP + FP)
```

For poisonous class: when the model predicts poisonous, how often is it correct?

**Recall**

```text
recall = TP / (TP + FN)
```

For poisonous class: how many actually poisonous mushrooms did the model detect?

**F1-score**

```text
F1 = 2 * precision * recall / (precision + recall)
```

Balances precision and recall.

**ROC curve**

```text
Shows the trade-off between true positive rate and false positive rate across thresholds.
```

**Precision-recall curve**

```text
Shows the trade-off between precision and recall across thresholds.
```

**Most relevant metric**

```text
Recall for the poisonous class.
```

**Why**

Because false negatives are dangerous: a poisonous mushroom predicted as edible is the worst type of mistake.

**What to say**

```text
Accuracy is useful as an overall score, but it is not enough here because all mistakes are not equally dangerous.
Recall for the poisonous class is the key metric because it measures how many dangerous mushrooms are detected.
F1-score is also useful because it balances recall and precision.
```

---

## Quick Defense Summary

```text
The official checklist for this assignment is covered.
Data preparation is handled before modelling, with split-first preprocessing to avoid leakage.
Logistic Regression is used and its C hyperparameter is tuned.
Three validation approaches are shown: single validation split, Stratified K-Fold and nested cross-validation.
Confusion matrices and classification metrics are used for evaluation.
The key metric is recall for the poisonous class, because false negatives are dangerous.
```

## If Asked What Is Not In The Notebook

```text
Leave-One-Out cross-validation is not implemented because it would require one model fit per observation.
For more than 61000 rows it is computationally impractical.
I can explain it theoretically, but I chose more practical validation methods in the notebook.
```

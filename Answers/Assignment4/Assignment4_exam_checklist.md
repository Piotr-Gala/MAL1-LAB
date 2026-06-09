# Assignment 4: Detecting Exoplanets - Exam Checklist Map

This file maps the exam-relevant topics to the Assignment 4 notebook.

Sources:

- Notebook: `MAL1-LAB/Answers/Assignment4/Assignment4.ipynb`
- Notes: `MAL1-LAB/Answers/Assignment4/Assignment4_notebook_explanation.md`
- Speech notes: `MAL1-LAB/Answers/Assignment4/Assignment4_speech_notes.md`

---

## Assignment-Specific Checklist

Official topic for Assignment 4: **Detecting Exoplanets**

- Data preparation
- Missing values and outliers
- Train / validation / test split
- Leakage-safe preprocessing
- Removing target-proxy features
- Correlation analysis and multicollinearity
- Logistic Regression
- Support Vector Machine
- Confusion matrix
- Performance metrics for classification
- Model comparison and limitations

---

## 1. Data Preparation

**Status:** covered

**Where in notebook**

- Cells 4-5: load dataset and inspect first rows
- Cells 6-8: initial overview and column renaming
- Cells 9-13: missing values and target encoding
- Cells 19-20: prepare `X` and `y`

**What happens**

- Load `exoplanet_dataset.csv`.
- Check dataset shape: `9564` rows and `49` columns.
- Inspect data types, missing values and summary statistics.
- Rename technical Kepler column names to more descriptive names.
- Remove columns with 100% missing values.
- Remove identifier/name columns.
- Encode disposition labels numerically.
- Select `KeplerDispositionStatus` as the target.

**What to say**

```text
I started by loading and inspecting the Kepler dataset, then removed unusable columns and encoded the disposition labels.
The final target is KeplerDispositionStatus, which creates a binary classification problem: false positive versus candidate.
```

---

## 2. Missing Values And Outliers

**Status:** covered

**Where in notebook**

- Cells 9-11: missing value percentages and column removal
- Cells 14-18: IQR outlier analysis and missing values before split
- Cell 28: median imputation after split

**What happens**

- Missing percentages are calculated for all columns.
- Two columns with 100% missing values are removed.
- `dropna()` is checked but not used because it would remove `1761` rows.
- IQR is used to count potential outliers.
- Outliers are kept.
- Missing feature values are later imputed with the median.

**What to say**

```text
I analyzed missing values and outliers, but I did not remove all incomplete rows or extreme values.
Dropping missing rows would remove 1761 observations, and extreme astronomical measurements may represent real physical cases.
```

---

## 3. Target-Proxy Feature Removal

**Status:** covered

**Where in notebook**

- Cell 20

**What happens**

The feature matrix removes target-related columns:

```text
DispositionScore
KeplerDispositionStatus
ArchiveDispositionStatus
```

It also removes the explicit false-positive flag columns:

```text
NotTransit-LikeFalsePositiveFlag
koi_fpflag_ss
CentroidOffsetFalsePositiveFlag
EphemerisMatchIndicatesContaminationFalsePositiveFlag
```

**Why it matters**

The false-positive flags are very close to the way the target label is assigned. Keeping them gives the model a strong shortcut. Removing them lowers the score, but makes the result easier to defend.

**What to say**

```text
I removed the explicit false-positive flag columns before modelling.
This makes the experiment more conservative, because the model has to rely more on physical and measurement-based features.
```

---

## 4. Train / Validation / Test Split

**Status:** covered

**Where in notebook**

- Cells 21-22

**Notebook detail**

```text
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

**What to say**

```text
I use a train-validation-test split with stratification.
The training set is used for fitting, the validation set for model selection, and the test set only for final evaluation.
```

---

## 5. Leakage-Safe Preprocessing

**Status:** covered

**Where in notebook**

- Cells 21-28
- Cell 41 for final retraining on train + validation

**What happens**

- Data is split before correlation filtering, imputation and scaling.
- Correlation analysis is fitted only on `X_train`.
- The same selected columns are removed from validation and test.
- Imputer is fitted only on training data.
- Scaler is fitted only on training data.
- Test set is transformed using objects fitted on training data.

**Important exam phrase**

```text
Fit on train, transform validation and test.
```

---

## 6. Correlation Analysis And Multicollinearity

**Status:** covered

**Where in notebook**

- Cells 23-26

**What happens**

- Correlation matrix is calculated only on `X_train`.
- Two highly correlated columns are removed:

```text
PlanetaryRadiusLowerUnc, Earthradii
InsolationFluxLowerUnc, Earthflux
```

**What to say**

```text
I calculated correlations only on the training data to avoid leakage.
Then I removed two highly correlated features from all splits to reduce redundancy and multicollinearity.
```

---

## 7. Logistic Regression

**Status:** covered

**Where in notebook**

- Cells 29-34
- Cell 41 for final model
- Cells 42-43 for test evaluation

**What happens**

- Logistic Regression is trained with several `C` values.
- Best `C` is selected using validation F1-score.
- Best `C`: `10`.
- Validation F1-score: about `0.803`.
- Final test F1-score: about `0.820`.

**What to say**

```text
I tuned Logistic Regression by testing several C values and selecting the one with the best validation F1-score.
The best C was 10. The final test F1-score was about 0.820.
```

---

## 8. Support Vector Machine

**Status:** covered

**Where in notebook**

- Cells 35-39
- Cell 41 for final model
- Cells 42-43 for test evaluation

**What happens**

- SVM is trained with several `C` values.
- Two kernels are tested: `linear` and `rbf`.
- Best model uses:

```text
C = 10
kernel = rbf
```

- Validation F1-score: about `0.826`.
- Final test F1-score: about `0.855`.

**What to say**

```text
I also trained an SVM model with different C values and linear or RBF kernels.
The best SVM used an RBF kernel with C equal to 10, and it performed best on the final test set.
```

---

## 9. Confusion Matrix

**Status:** covered

**Where in notebook**

- Cell 33: Logistic Regression validation confusion matrix
- Cell 39: SVM validation confusion matrix
- Cell 43: final test confusion matrices

**Classes**

```text
0 = FALSE POSITIVE
1 = CANDIDATE
```

**For candidate as positive class**

```text
                 Predicted false positive   Predicted candidate
Actual false positive    TN                 FP
Actual candidate         FN                 TP
```

---

## 10. Performance Metrics For Classification

**Status:** covered

**Where in notebook**

- Cells 30, 32, 36, 38, 42

**Final test results**

```text
Logistic Regression:
Accuracy: about 0.821
F1-score: about 0.820

SVM:
Accuracy: about 0.855
F1-score: about 0.855
```

**What to say**

```text
Accuracy gives the overall correctness, but precision and recall show different types of errors.
F1-score is useful because it balances precision and recall.
```

---

## 11. Model Comparison And Limitations

**Status:** covered

**Where in notebook**

- Cells 40-45

**What happens**

- After tuning, both models are retrained on train + validation data.
- Final evaluation is done on the untouched test set.
- SVM performs better than Logistic Regression.
- Removing false-positive flags makes the result more conservative and easier to defend.

**What to say**

```text
After removing the explicit false-positive flag columns, the final scores are lower than in the original version.
This is expected, because the model no longer gets a strong shortcut from target-proxy features.
SVM achieved the best final test performance, with about 85.5% accuracy and F1-score.
```

---

## Quick Defense Summary

```text
The notebook covers the main exam-relevant topics for this assignment.
Data preparation includes missing values, outliers, target encoding and feature preparation.
The explicit false-positive flag columns are removed before modelling to reduce target-proxy risk.
The split is done before correlation filtering, imputation and scaling to avoid leakage.
Logistic Regression and SVM are trained and tuned on validation data.
The final test set is used only at the end.
SVM is selected because it performs better on the final test set.
```

## If Asked What Was Fixed

```text
The original version allowed the model to use explicit false-positive flag columns.
Those columns were removed from X before modelling.
As a result, the performance dropped from the old flag-based scores to more realistic final test scores, and SVM became the better final model.
```

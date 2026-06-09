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

**Why it matters**

The model needs clean numerical input features and a clearly defined target variable. Removing identifiers also reduces the risk of memorization.

**What to say**

```text
I started by loading and inspecting the Kepler dataset, then removed unusable columns and encoded the disposition labels.
The final target is KeplerDispositionStatus, which creates a binary classification problem: false positive versus candidate.
```

**Possible question**

```text
Why did you remove identifier columns?
```

**Answer**

```text
Because identifiers do not describe physical properties and may cause the model to memorize objects instead of learning general patterns.
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

**Why it matters**

Astronomical data can contain real extreme values, so automatically removing outliers could remove useful observations. Median imputation keeps more data and is robust to outliers.

**What to say**

```text
I analyzed missing values and outliers, but I did not remove all incomplete rows or extreme values.
Dropping missing rows would remove 1761 observations, and extreme astronomical measurements may represent real physical cases.
```

**Possible question**

```text
Why did you not remove outliers?
```

**Answer**

```text
Because extreme values in astronomical data may be valid physical observations, not errors. Removing them could remove useful information.
```

---

## 3. Train / Validation / Test Split

**Status:** covered

**Where in notebook**

- Cells 19-22

**What happens**

```text
X_train_val, X_test = first split
X_train, X_val = second split
```

**Notebook detail**

```text
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

**Why it matters**

- Training data fits the model.
- Validation data tunes hyperparameters.
- Test data is kept untouched until final evaluation.
- Stratification keeps class proportions similar in all subsets.

**What to say**

```text
I use a train-validation-test split with stratification.
The training set is used for fitting, the validation set for model selection, and the test set only for final evaluation.
```

**Possible question**

```text
Why use stratification?
```

**Answer**

```text
Stratification keeps the class proportions similar in train, validation and test sets.
```

---

## 4. Leakage-Safe Preprocessing

**Status:** covered

**Where in notebook**

- Cells 19-28
- Cell 41 for final retraining on train + validation

**What happens**

- `X` and `y` are prepared first.
- Data is split before correlation filtering, imputation and scaling.
- Correlation analysis is fitted only on `X_train`.
- The same selected columns are removed from validation and test.
- Imputer is fitted only on training data.
- Scaler is fitted only on training data.
- Test set is transformed using objects fitted on training data.

**Why it matters**

This avoids data leakage. Validation and test information should not influence preprocessing or model fitting.

**Important exam phrase**

```text
Fit on train, transform validation and test.
```

**Possible question**

```text
What is data leakage?
```

**Answer**

```text
Data leakage happens when information from validation or test data is used during training or preprocessing.
```

---

## 5. Correlation Analysis And Multicollinearity

**Status:** covered

**Where in notebook**

- Cells 23-26

**What happens**

- Correlation matrix is calculated only on `X_train`.
- Strong absolute correlations above `0.95` are extracted.
- Two highly correlated columns are removed:

```text
PlanetaryRadiusLowerUnc, Earthradii
InsolationFluxLowerUnc, Earthflux
```

**Why it matters**

Highly correlated variables may carry redundant information. This can make linear models harder to interpret.

**What to say**

```text
I calculated correlations only on the training data to avoid leakage.
Then I removed two highly correlated features from all splits to reduce redundancy and multicollinearity.
```

**Possible question**

```text
What is multicollinearity?
```

**Answer**

```text
Multicollinearity occurs when two or more features are strongly correlated and carry similar information.
```

---

## 6. Logistic Regression

**Status:** covered

**Where in notebook**

- Cells 29-34
- Cell 41 for final model
- Cells 42-43 for test evaluation

**What happens**

- Logistic Regression is trained with several `C` values.
- Best `C` is selected using validation F1-score.
- Best `C`: `0.1`.
- Validation F1-score is about `0.994`.
- Final test F1-score is about `0.902`.

**Why it matters**

Logistic Regression is a simple, interpretable classification model. It is useful as a strong baseline and is easier to explain than more complex models.

**What to say**

```text
I tuned Logistic Regression by testing several C values and selecting the one with the best validation F1-score.
The best C was 0.1. Logistic Regression was selected as the final model because it performed slightly better on the test set and is easier to explain.
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

## 7. Support Vector Machine

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
C = 0.01
kernel = linear
```

- Validation F1-score is about `0.993`.
- Final test F1-score is about `0.898`.

**Why it matters**

SVM is another classification model that finds a decision boundary with a maximum margin. Comparing it with Logistic Regression shows whether a different classifier generalizes better.

**What to say**

```text
I also trained an SVM model with different C values and linear or RBF kernels.
The best SVM used a linear kernel with C equal to 0.01, but Logistic Regression was slightly better on the final test set.
```

**Possible question**

```text
What is SVM?
```

**Answer**

```text
SVM is a classification algorithm that tries to find a decision boundary with the maximum margin between classes.
```

---

## 8. Confusion Matrix

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

**What to say**

```text
The confusion matrix shows not only how many predictions were correct, but also which types of errors the model made.
In this task, a false positive means predicting candidate for an object that is actually a false positive, while a false negative means missing an actual candidate.
```

**Possible question**

```text
What is a false negative here?
```

**Answer**

```text
A false negative means that an actual candidate was predicted as a false positive, so the model missed a potential planet candidate.
```

---

## 9. Performance Metrics For Classification

**Status:** covered

**Where in notebook**

- Cells 30, 32, 36, 38, 42

**Metrics used**

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

**Final test results**

```text
Logistic Regression:
Accuracy: about 0.902
F1-score: about 0.902

SVM:
Accuracy: about 0.899
F1-score: about 0.898
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

When the model predicts candidate, how often is it correct?

**Recall**

```text
recall = TP / (TP + FN)
```

How many actual candidates did the model detect?

**F1-score**

```text
F1 = 2 * precision * recall / (precision + recall)
```

Balances precision and recall.

**What to say**

```text
Accuracy gives the overall correctness, but precision and recall show different types of errors.
F1-score is useful because it balances precision and recall.
```

**Possible question**

```text
Why use F1-score?
```

**Answer**

```text
F1-score balances precision and recall, so it is useful when we care about both false positives and false negatives.
```

---

## 10. Model Comparison And Limitations

**Status:** covered

**Where in notebook**

- Cells 40-45

**What happens**

- After tuning, both models are retrained on train + validation data.
- Final evaluation is done on the untouched test set.
- Logistic Regression performs slightly better than SVM.
- The main limitation is proxy information from false-positive flags.

**Why it matters**

The final test set gives the most realistic estimate of generalization. Very high validation results should not be overinterpreted.

**What to say**

```text
Both models performed very well on validation data, but the test performance was lower, around 90%.
I selected Logistic Regression because it was slightly better and simpler.
The main limitation is that some false-positive flag features may act as proxy variables for the target label.
```

**Possible question**

```text
Why did validation performance differ from test performance?
```

**Answer**

```text
The validation result was probably optimistic. The test set gives a more realistic estimate of generalization.
```

---

## Quick Defense Summary

```text
The notebook covers the main exam-relevant topics for this assignment.
Data preparation includes missing values, outliers, target encoding and feature preparation.
The split is done before correlation filtering, imputation and scaling to avoid leakage.
Logistic Regression and SVM are trained and tuned on validation data.
The final test set is used only at the end.
Logistic Regression is selected because it performs slightly better and is simpler to explain.
The main limitation is that false-positive flags may act as proxy features for the target.
```

## If Asked What Was Fixed

```text
The correlation analysis was changed to avoid data leakage.
Originally, correlation filtering could be done before the split, but now the correlation matrix is calculated only on the training data.
The selected columns are then removed from train, validation and test sets.
```

# Assignment 1: EV Car Prices - Exam Checklist Map

This file maps the official exam checklist to the Assignment 1 notebook.

Sources:

- `Materials/exam/Exam_information_and_assignments_overview.pdf`
- `Materials/exam/Exam_theory_topics.pdf`
- Notebook: `MAL1-LAB/Answers/Assignment1/Assignment1.ipynb`
- Notes: `MAL1-LAB/Answers/Assignment1/Assignment1_notebook_explanation.md`
- Speech notes: `MAL1-LAB/Answers/Assignment1/Assignment1_speech_notes.md`

---

## Assignment-Specific Checklist From Exam Overview

Official topic for Assignment 1: **Car prices**

- Regression algorithms
- Linear regression with linear algebra
- Linear regression with library functions
- Performance metrics for regression
- Regularization with Lasso, Ridge and Elastic Net
- kNN classification

---

## 1. Data Preparation And Split

**Status:** covered

**Where in notebook**

- Cells 0-2: load data, inspect dataset, create `X` and `y`, train/test split

**What happens**

- Load `car_prices.xlsx`.
- Check dataset information and missing values.
- Define target variable: `Price (DKK)`.
- Define feature matrix: all columns except `Price (DKK)`.
- Split data into training and test sets.

**Notebook details**

```text
Dataset shape: 6226 rows, 16 columns
Target: Price (DKK)
Split: 80% train, 20% test
random_state = 42
```

**Why it matters**

The exam theory topics include validation methods, train-test split, preprocessing order and generalization.

This notebook uses the test set only for evaluation, not for fitting the regression models.

**What to say**

```text
I separate X and y first, then split the data into train and test sets.
The model is fitted on the training data and evaluated on the test data to estimate generalization.
```

**Possible question**

```text
Why do you need a test set?
```

**Answer**

```text
The test set estimates how well the model performs on unseen data.
If I evaluate only on training data, I may overestimate performance.
```

---

## 2. Linear Regression With Linear Algebra

**Status:** covered

**Where in notebook**

- Cells 3-9: NumPy implementation of linear regression

**What happens**

- Convert `X_train` and `y_train` to NumPy arrays.
- Add a column of ones for the intercept.
- Calculate model coefficients using the pseudoinverse.
- Predict on `X_test`.
- Calculate MSE and `R2` manually.

**Why it matters**

The exam overview explicitly lists **linear regression with linear algebra** for Assignment 1.

The theory topics also include:

- simple linear regression,
- regression algorithms,
- regression performance metrics,
- matrix/correlation concepts.

**What to say**

```text
I build the design matrix manually by adding an intercept column.
Then I calculate the least-squares coefficient vector and use it to predict test-set prices.
```

**Formula to know**

```text
y_hat = X_b @ theta_hat
```

The classic normal-equation form is:

```text
theta_hat = (X.T X)^-1 X.T y
```

In the notebook I use:

```text
theta_hat = pinv(X) y
```

because it is more numerically stable.

**Possible question**

```text
Why do you add a column of ones?
```

**Answer**

```text
It represents the intercept term, so the model can learn a baseline price.
```

**Possible question**

```text
Why use np.linalg.pinv?
```

**Answer**

```text
It gives a stable least-squares solution and avoids problems if X.T X is hard to invert or if features are correlated.
```

---

## 3. Linear Regression With Library Functions

**Status:** covered

**Where in notebook**

- Cells 10-15: correlation matrix and `LinearRegression`

**What happens**

- Calculate a correlation matrix and show it as a heatmap.
- Fit `LinearRegression` on `X_train` and `y_train`.
- Predict on `X_test`.
- Calculate MSE, RMSE and `R2`.

**Notebook details**

```text
MSE: 2774486707.59
RMSE: 52673.40 DKK
R2: 0.8644
```

**Why it matters**

The exam overview explicitly lists **linear regression with library functions**.

The theory topics include regression performance metrics:

- MSE,
- MAE,
- RMSE,
- R-squared.

This notebook covers MSE, RMSE and `R2`. MAE is not used in the notebook, but you should know what it means.

**What to say**

```text
The Scikit-learn model confirms the manual result.
RMSE is about 52,673 DKK, so predictions are typically off by around 52,000 DKK in the original unit.
R2 is about 0.864, meaning the model explains around 86% of the variance in test prices.
```

**Possible question**

```text
What is the difference between MSE and RMSE?
```

**Answer**

```text
MSE is the average squared error.
RMSE is the square root of MSE, so it is in the same unit as the target variable.
Here RMSE is easier to interpret because it is measured in DKK.
```

**Possible question**

```text
What does R2 mean?
```

**Answer**

```text
R2 tells how much of the variance in the target is explained by the model.
An R2 of about 0.864 means the model explains about 86% of the test-set price variation.
```

**Possible question**

```text
Could the correlation heatmap cause data leakage?
```

**Answer**

```text
It would be risky if I used it to select features using the full dataset.
Here it is only used for exploration and interpretation, not for feature selection.
```

---

## 4. Regression Metrics

**Status:** mostly covered

**Where in notebook**

- Cells 7-8: manual MSE and `R2`
- Cells 14-15: Scikit-learn MSE, RMSE and `R2`
- Cells 20-23: regularized model metrics

**Covered metrics**

- MSE
- RMSE
- `R2`

**Not directly used, but exam-relevant**

- MAE

**What to say**

```text
MSE penalizes large errors strongly because errors are squared.
RMSE is easier to interpret because it is measured in the target unit, DKK.
R2 describes explained variance.
MAE would be the average absolute error and is less sensitive to large outliers than MSE or RMSE.
```

**Possible question**

```text
Why might RMSE be preferred over MSE for explaining results?
```

**Answer**

```text
RMSE is in the same unit as the target variable.
So instead of explaining squared Danish Kroner, I can say the model is off by around 52,000 DKK.
```

---

## 5. Regularization With Ridge, Lasso And Elastic Net

**Status:** covered

**Where in notebook**

- Cells 16-18: task description and scaling note
- Cells 19-20: scaling and training Ridge, Lasso and Elastic Net
- Cells 21-23: selecting best models and comparing with scaled OLS
- Cells 24-27: coefficient comparison and interpretation

**What happens**

- Standardize `X_train` and `X_test`.
- Standardize `y_train` and `y_test`.
- Fit Ridge, Lasso and Elastic Net using several alpha values.
- Compare scaled MSE and `R2`.
- Compare most important features by absolute standardized coefficient size.

**Notebook details**

```text
Best Ridge alpha: 0.001
Best Lasso alpha: 0.01
Best Elastic Net alpha: 0.01
Best regularized model: Lasso with alpha = 0.01
```

**Why it matters**

The exam overview explicitly lists **Regularization with Lasso, Ridge and Elastic Net**.

The theory topics list:

- Ridge regression,
- Lasso regression,
- Elastic Net regression,
- scaling for algorithms sensitive to feature scale.

**What to say**

```text
Regularization penalizes large coefficients.
Ridge uses an L2 penalty and shrinks coefficients.
Lasso uses an L1 penalty and can set some coefficients exactly to zero.
Elastic Net combines both penalties.
```

**Important exam phrase**

```text
Fit the scaler on train, transform test.
```

**Possible question**

```text
Why do Ridge and Lasso need scaling?
```

**Answer**

```text
Because the penalty depends on coefficient size.
If features are on different scales, the penalty affects them unfairly.
```

**Possible question**

```text
What does alpha control?
```

**Answer**

```text
Alpha controls regularization strength.
A larger alpha means stronger regularization and usually smaller coefficients.
```

**Possible question**

```text
Did regularization improve the model a lot?
```

**Answer**

```text
No. Lasso is slightly best among the tested regularized models, but the improvement over OLS is small.
So OLS is already a strong baseline for this dataset.
```

---

## 6. Scaling And Leakage

**Status:** covered

**Where in notebook**

- Cell 18: scaling note
- Cell 19: `StandardScaler` for regression
- Cell 31: `StandardScaler` for kNN classification

**What happens**

```text
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

The scaler learns means and standard deviations from training data only.

**Why it matters**

The exam theory topics include:

- scaling,
- preprocessing and splitting data,
- train-test workflow,
- test set only for final evaluation.

**What to say**

```text
Scaling is fitted only on training data.
The test set is transformed with the already fitted scaler, so its statistics do not influence training.
```

**Possible question**

```text
What is data leakage here?
```

**Answer**

```text
Data leakage would happen if information from the test set influenced preprocessing or model fitting.
For example, fitting a scaler on the full dataset before the split would leak test-set statistics into training.
```

---

## 7. kNN Classification

**Status:** covered

**Where in notebook**

- Cells 28-29: create binary target using median price
- Cell 30: stratified train/test split
- Cell 31: feature scaling
- Cells 32-34: train and evaluate kNN with different `k` values and distance metrics

**What happens**

- Median price is calculated.
- Cars above median are labelled `Expensive`.
- Cars at or below median are labelled `Cheap`.
- kNN is tested with several `k` values and distance metrics.
- Best result is selected by accuracy.

**Notebook details**

```text
Median price: 304900 DKK
Best kNN: k = 5, metric = manhattan
Accuracy: about 0.94
```

**Why it matters**

The exam overview explicitly lists **kNN classification** for Assignment 1.

The theory topics list:

- kNN,
- importance of scaling,
- general classification concepts,
- classification metrics.

**What to say**

```text
kNN classifies a new observation based on the labels of its nearest training examples.
Because it uses distances, scaling is required.
In this notebook, k = 5 with Manhattan distance gave the best accuracy.
```

**Possible question**

```text
Why use stratification in the classification split?
```

**Answer**

```text
Because the target is now binary.
Stratification keeps the Cheap and Expensive proportions similar in train and test sets.
```

**Possible question**

```text
What is the trade-off in choosing k?
```

**Answer**

```text
A very small k can overfit because predictions depend too much on individual points.
A very large k can underfit because it smooths the decision boundary too much.
```

**Possible question**

```text
Why is kNN sensitive to scaling?
```

**Answer**

```text
kNN uses distances.
If features have very different scales, large-scale features dominate the distance calculation.
```

---

## 8. Classification Metrics

**Status:** partially covered

**Where in notebook**

- Cells 32-34: accuracy, confusion matrix and classification report

**Covered**

- accuracy
- confusion matrix
- precision/recall/F1 through `classification_report`

**What to say**

```text
Accuracy gives the overall proportion of correct predictions.
The confusion matrix shows which class is misclassified.
The classification report gives precision, recall and F1-score for each class.
```

**Possible question**

```text
Why is accuracy acceptable here?
```

**Answer**

```text
The target was created using the median price, so the Cheap and Expensive classes are almost balanced.
Accuracy is therefore a reasonable first metric, but I still check the confusion matrix and classification report.
```

---

## 9. Limitations

**Status:** covered

**Where in notebook**

- Cell 35: limitations

**Main limitations**

- Linear models may miss non-linear relationships.
- RMSE is still around 52,000 DKK, which is large in real money terms.
- Regularization only slightly improves performance.
- kNN depends strongly on scaling, distance metric and `k`.
- Cheap/Expensive is an artificial target created from the median price.

**What to say**

```text
The models are useful and explain much of the price variation, but they are not perfect.
Individual price predictions can still be far from the real price, and the classification task is simplified because Cheap and Expensive are defined by the median.
```

---

## Quick Exam Checklist

- Explain what `X` and `y` are.
- Explain why train/test split is needed.
- Explain why the test set should be used only for final evaluation.
- Explain the intercept column in linear regression.
- Explain the normal equation idea.
- Defend using `np.linalg.pinv`.
- Explain MSE.
- Explain RMSE.
- Explain `R2`.
- Know what MAE means, even though it is not used in the notebook.
- Explain why Ridge, Lasso and Elastic Net need scaling.
- Explain Ridge vs Lasso vs Elastic Net.
- Explain what `alpha` controls.
- Explain `fit_transform` on train and `transform` on test.
- Explain data leakage in preprocessing.
- Explain why kNN needs scaling.
- Explain the small `k` vs large `k` trade-off.
- Explain why stratification is used in the classification part.
- Explain accuracy and confusion matrix.
- Mention limitations clearly.

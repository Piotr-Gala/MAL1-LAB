# Assignment 1: EV Car Prices - Notebook Explanation

## Opening Speech

```text
Good morning. In this assignment I worked with a dataset of electric vehicle prices.
The main target variable is Price (DKK), so the first part of the notebook is a regression problem.

I started by loading the dataset, checking basic information and missing values, and then splitting the data into training and test sets.
The split is done before fitting models or scalers, so the test set stays unseen until evaluation.

Then I implemented multiple linear regression using linear algebra with NumPy.
After that, I used Scikit-learn to build an ordinary least squares model and evaluate it with MSE, RMSE and R2.

In the next part, I scaled the data and compared Ridge, Lasso and Elastic Net using several alpha values.
Finally, I converted the price problem into a binary classification problem by splitting cars into Cheap and Expensive using the median price, and I trained a kNN classifier.
```

---

## Cells 0-2: Load Data And Train / Test Split

**What to say**

```text
At the beginning I load the Excel file into a pandas DataFrame.
Then I inspect the dataset with info(), check missing values, separate the response variable from the features, and create a train/test split.
```

**Notebook details**

```text
6226 rows and 16 columns
Target variable: Price (DKK)
Random seed: 42
Train/test split: 80% training, 20% test
```

**Plain English**

`X` contains the input features.  
`y` contains the target value that the model should predict.

The split is made before model fitting. This matters because the test set should simulate unseen future data.

**Most important**

- `pd.read_excel()` loads the dataset.
- `df.info()` checks column types and missing values.
- `X = df.drop("Price (DKK)", axis=1)` removes the target from the feature matrix.
- `y = df["Price (DKK)"]` stores the target.
- `train_test_split(..., random_state=42)` creates reproducible train and test sets.

**Question**

```text
What is the target variable?
```

**Answer**

```text
The target variable is Price (DKK), which is the current listed price of the electric vehicle.
```

**Question**

```text
Why do you split the data before modelling?
```

**Answer**

```text
Because I need an unseen test set to estimate how well the model generalizes to new data.
```

**Question**

```text
Is stratification needed here?
```

**Answer**

```text
For the regression part, no. The target is continuous, so normal train_test_split is acceptable.
For the later classification part, I use stratification because the target becomes binary.
```

---

## Cells 3-9: Linear Regression With Linear Algebra

**What to say**

```text
In this section I implement linear regression using linear algebra.
I convert the training data to NumPy arrays, add a column of ones for the intercept, calculate the coefficient vector, and then predict prices for the test set.
Finally, I calculate MSE and R2 manually.
```

**Plain English**

Linear regression tries to find coefficients that make the predicted prices close to the real prices.

The column of ones represents the intercept term. Without it, the model would be forced through zero.

**Key formula**

```text
y_hat = X_b @ theta_hat
```

where:

- `X_b` is the feature matrix with an intercept column,
- `theta_hat` is the coefficient vector,
- `y_hat` is the predicted price.

**Important detail**

The notebook uses:

```python
np.linalg.pinv(X_train_b) @ y_train_np
```

This is the Moore-Penrose pseudoinverse. It gives a stable least-squares solution and avoids directly inverting `X.T @ X`.

**Result**

```text
R2 is about 0.864
```

So the model explains about 86% of the variance in the test prices.

**Question**

```text
Why do you add a column of ones?
```

**Answer**

```text
The column of ones allows the model to learn an intercept, meaning a baseline price when all feature values are zero.
```

**Question**

```text
Why use np.linalg.pinv instead of directly calculating the inverse?
```

**Answer**

```text
It is more numerically stable and still gives the least-squares solution.
It also works better if the feature matrix has correlated columns or is not full rank.
```

**Question**

```text
What does R2 mean here?
```

**Answer**

```text
R2 tells how much of the variance in car prices is explained by the model.
Here, about 86% of the variance is explained on the test set.
```

---

## Cells 10-15: Correlation Matrix And OLS With Scikit-Learn

**What to say**

```text
Here I use library functions to build the same type of regression model.
First I calculate a correlation matrix and show it as a heatmap.
Then I train an ordinary least squares model using LinearRegression from Scikit-learn and evaluate it with MSE, RMSE and R2.
```

**Notebook details**

```text
MSE: 2774486707.59
RMSE: 52673.40 DKK
R2: 0.8644
```

**Plain English**

The manual linear algebra model and the Scikit-learn OLS model give the same type of result.

RMSE is easier to explain than MSE because RMSE is in the original unit: Danish Kroner.

**Important**

The correlation matrix is used for exploration and interpretation only. It is not used for feature selection, so it does not introduce model-selection leakage.

**Question**

```text
Why is RMSE useful?
```

**Answer**

```text
RMSE is in the same unit as the target variable, so I can say that predictions are typically off by about 52,000 DKK.
```

**Question**

```text
Why is MSE so large?
```

**Answer**

```text
Because prices are measured in Danish Kroner and the errors are squared.
Large target values naturally create large squared errors.
```

**Question**

```text
Could the correlation matrix cause data leakage?
```

**Answer**

```text
It could be risky if I used it to select features based on the whole dataset.
In this notebook I only use it for exploration and interpretation, not for selecting the model inputs.
```

---

## Cells 16-27: Ridge, Lasso And Elastic Net

**What to say**

```text
In this section I compare regularized regression models.
Before using Ridge, Lasso and Elastic Net, I standardize both X and y.
This is important because regularization penalizes coefficient sizes, so all features should be on a comparable scale.
```

**Plain English**

Regularization adds a penalty for large coefficients.

Ridge shrinks coefficients but usually keeps all features.  
Lasso can shrink some coefficients all the way to zero.  
Elastic Net combines Ridge and Lasso.

**Leakage-safe scaling**

```text
scaler_X.fit_transform(X_train)
scaler_X.transform(X_test)
```

The scaler learns the mean and standard deviation only from the training data.
Then the same transformation is applied to the test data.

**Notebook details**

```text
Best Ridge alpha: 0.001
Best Lasso alpha: 0.01
Best Elastic Net alpha: 0.01
Best regularized model in this run: Lasso with alpha = 0.01
```

The differences are small, so regularization does not dramatically improve performance over OLS.

**Top features**

OLS and Ridge mainly agree on:

- `Original Price (DKK)`
- `Model Year`
- `Mileage (km)`
- `0-100 km/h (s)`
- `Electric Range (km)`

Lasso mainly agrees, but replaces one of the weaker features with:

- `Annual Road Tax (DKK)`

**Question**

```text
Why do you scale the data before Ridge and Lasso?
```

**Answer**

```text
Because Ridge and Lasso penalize coefficient size.
If features are on different scales, the penalty would affect them unfairly.
```

**Question**

```text
What is the difference between Ridge and Lasso?
```

**Answer**

```text
Ridge uses an L2 penalty and shrinks coefficients toward zero.
Lasso uses an L1 penalty and can set some coefficients exactly to zero, which makes it useful for feature selection.
```

**Question**

```text
What does a standardized coefficient mean?
```

**Answer**

```text
It means the expected change in the target, measured in standard deviations, for a one standard deviation increase in that feature, holding other features constant.
```

**Question**

```text
Did regularization clearly improve the model?
```

**Answer**

```text
No, only slightly. The OLS model was already a strong baseline for this dataset.
```

---

## Cells 28-34: kNN Classification

**What to say**

```text
In the final part I turn the regression problem into a classification problem.
I calculate the median price and classify cars as Expensive if their price is above the median, otherwise Cheap.
Then I split the data again using stratification, scale the features, and test kNN with several values of k and several distance metrics.
```

**Notebook details**

```text
Median price: 304900 DKK
Best kNN setting: k = 5, metric = manhattan
Accuracy: about 0.94
```

**Plain English**

kNN predicts a class by looking at the nearest training examples.
Because kNN uses distances, scaling is very important.

**Why stratify here**

The new target is binary: Cheap or Expensive.  
Stratification keeps the class proportions similar in train and test sets.

**Question**

```text
Why do you use the median to create the target?
```

**Answer**

```text
The median splits the dataset into two almost balanced groups: cheap and expensive cars.
This makes the classification problem simple and avoids a strongly imbalanced target.
```

**Question**

```text
Why does kNN need scaling?
```

**Answer**

```text
kNN is distance-based.
Without scaling, features with large numerical ranges, such as price-related or mileage values, could dominate the distance calculation.
```

**Question**

```text
What is the trade-off between small and large k?
```

**Answer**

```text
A very small k can overfit because it reacts too strongly to individual observations.
A very large k can underfit because it smooths the decision boundary too much.
```

**Question**

```text
Why check the confusion matrix?
```

**Answer**

```text
Accuracy alone does not show which class is misclassified.
The confusion matrix shows how many Cheap and Expensive cars were classified correctly or incorrectly.
```

---

## Cells 35: Limitations

**What to say**

```text
The notebook gives good baseline models, but there are limitations.
The regression models are mostly linear, so they may miss non-linear relationships.
The RMSE is still around 52,000 DKK, which means individual predictions can be quite far away from the real price.
The kNN classifier performs well, but it depends strongly on scaling and on the chosen value of k.
```

**Most important limitations**

- Linear models assume mostly linear relationships.
- RMSE is still large in real money terms.
- Regularization only slightly improves performance.
- kNN is sensitive to scaling and distance metric.
- The binary classification target is artificial because Cheap/Expensive is created from the median price.

**Question**

```text
What is the main limitation of the regression model?
```

**Answer**

```text
It is mostly linear, so it may not capture more complex non-linear relationships between car features and price.
```

**Question**

```text
Is 94% kNN accuracy enough to say the model is perfect?
```

**Answer**

```text
No. The target was artificially created from the median price, and kNN depends strongly on scaling and parameter choices.
The accuracy is good for this task, but it does not mean the model is perfect.
```

---

## Quick Exam Checklist

- Explain `X` vs `y`.
- Explain why the split happens before scaling.
- Explain `fit_transform` on train and `transform` on test.
- Explain MSE, RMSE and R2.
- Explain why RMSE is easier to interpret than MSE.
- Explain why `np.linalg.pinv` is used.
- Explain why Ridge/Lasso need scaling.
- Explain Ridge vs Lasso vs Elastic Net.
- Explain standardized coefficients.
- Explain why kNN needs scaling.
- Explain the small-k vs large-k trade-off.
- Mention that the correlation matrix is not used for feature selection.
- Mention limitations clearly.

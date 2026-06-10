# Assignment 1: EV Car Prices - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with a dataset of electric vehicle prices.
The main target variable is Price in Danish Kroner, so the first part of the notebook is a regression problem.

I started by loading the dataset, checking basic information and missing values, and splitting the data into training and test sets.
The split is done before fitting models or scalers, so the test set remains unseen until evaluation.

Then I implemented multiple linear regression using NumPy and linear algebra.
After that, I used Scikit-learn to build an ordinary least squares model and evaluate it with MSE, RMSE and R2.

Next, I standardized the data and compared Ridge, Lasso and Elastic Net regression using several alpha values.
Finally, I transformed the problem into binary classification by using the median price to label cars as Cheap or Expensive, and I trained a kNN classifier.
```

## Cells 0-2: Load Data And Train / Test Split

```text
In the first section I load the car prices dataset from an Excel file.
Then I inspect the data using info() and missing-value counts.
The dataset contains 6226 rows and 16 columns.

I separate the target variable, Price (DKK), from the input features.
Then I split the data into training and test sets using random_state 42, so the results are reproducible.
This split happens before fitting any model or scaler, which helps avoid data leakage.
```

## Cells 3-9: Linear Regression With Linear Algebra

```text
In this part I implement linear regression manually using NumPy.
I convert the training data to arrays and add a column of ones to include the intercept term.

Then I calculate the coefficient vector using the Moore-Penrose pseudoinverse.
I use np.linalg.pinv because it is more numerically stable than directly inverting X transpose X, especially when features are correlated.

After calculating the coefficients, I predict prices on the test set and calculate MSE and R2 manually.
The R2 is about 0.864, meaning the model explains about 86% of the variance in test-set car prices.
```

## Cells 10-15: Correlation Matrix And OLS

```text
Here I use library functions for correlation analysis and ordinary least squares regression.
The correlation heatmap helps me understand relationships between variables.
For example, Original Price has a strong positive relationship with current price, while Mileage is negatively related to price.

The correlation matrix is only used for exploration and interpretation.
I do not use it to select features, so it does not leak test information into the model-building process.

Then I train LinearRegression from Scikit-learn on the training data and evaluate it on the test data.
The RMSE is about 52,673 DKK, which means predictions are typically off by around 52,000 DKK.
The R2 is about 0.864, which matches the manual linear algebra result.
```

## Cells 16-27: Ridge, Lasso And Elastic Net

```text
In this section I compare regularized regression models.
Before training Ridge, Lasso and Elastic Net, I standardize both the features and the target.
This is important because regularization penalizes coefficient size, so all variables should be on comparable scales.

The scaler is fitted only on the training data using fit_transform.
The test data is transformed using transform only.
This avoids data leakage because the test-set mean and standard deviation are not used during training.

I test several alpha values for Ridge, Lasso and Elastic Net.
Lasso with alpha equal to 0.01 gives the best regularized result in this run, but the difference compared with OLS is small.
This means regularization does not dramatically improve the model on this dataset.

I also compare the largest standardized coefficients.
OLS and Ridge mainly identify Original Price, Model Year and Mileage as the most important predictors.
Lasso gives a very similar picture, but because of its L1 penalty it can shrink weaker coefficients more strongly.
```

## Cells 28-34: kNN Classification

```text
In the final part I turn the regression task into a classification task.
I calculate the median car price, which is 304900 DKK.
Cars above the median are labelled as Expensive, and cars at or below the median are labelled as Cheap.

Because this is now a classification problem, I use stratified train-test splitting.
This keeps the Cheap and Expensive class proportions similar in the training and test sets.

Then I scale the features, because kNN is distance-based.
Without scaling, features with larger numerical ranges could dominate the distance calculation.

I test several values of k and three distance metrics: euclidean, manhattan and minkowski.
The best result is k equal to 5 with Manhattan distance, with accuracy around 94%.
The confusion matrix and classification report show how the model performs separately for Cheap and Expensive cars.
```

## Cells 35: Limitations

```text
The regression models are useful baselines, but they are mostly linear.
That means they may miss more complex non-linear relationships between car features and price.

The RMSE is still around 52,000 DKK, so individual predictions can be quite far from the true listed price.
Regularization only slightly improves performance, which suggests that the basic OLS model is already a strong baseline here.

For kNN, the main limitation is that it depends strongly on scaling, distance metric and the chosen value of k.
Also, the Cheap versus Expensive target is artificial because I created it from the median price.
```

## Metrics Speech

```text
For regression, MSE is the average squared prediction error.
RMSE is the square root of MSE, so it is easier to interpret because it is measured in Danish Kroner.
R2 tells how much of the variation in car prices is explained by the model.

For classification, accuracy tells how many cars were classified correctly overall.
The confusion matrix shows the actual mistake types, so I can see whether Cheap or Expensive cars are misclassified more often.
```

## Leakage Speech

```text
The main leakage rule in this notebook is that the split comes before fitting preprocessing.
For scaling, I fit the scaler only on the training data and then transform the test data.
This means the test set does not influence the preprocessing parameters.

The correlation matrix is used only for exploration, not for feature selection.
If I had selected features using correlations calculated on the whole dataset, that would be a leakage risk.
```

## Final Sentence

```text
Overall, the notebook shows that linear regression already explains a large part of EV price variation, regularization only gives a small improvement, and kNN can classify cheap versus expensive cars well after proper scaling.
```

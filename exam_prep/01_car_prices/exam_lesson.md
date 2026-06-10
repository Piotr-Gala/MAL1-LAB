# Assignment 1: Car Prices - ADHD-Friendly Oral Exam Lesson

## How To Use This File

To jest wersja do nauki ustnej: najpierw masz normalny kawałek lekcji, a dopiero pod spodem wyjaśnienia słów z tej sekcji.

Schemat:

```text
section
lesson text
term explanations
exam sentences
```

Main exam pattern:

```text
What does it mean?
Where is it in your code?
Why did you do it?
```

---

# 0. Big Picture

Assignment 1 is about predicting electric vehicle prices.

The main target variable is `Price (DKK)`, so the first part of the notebook is a supervised regression problem.

Later, the notebook changes the task into binary classification by creating two classes: `Cheap` and `Expensive`.

So the assignment has two main parts:

```text
Regression:
input car features -> predict exact price

Classification:
input car features -> predict Cheap or Expensive
```

## Terms From This Section

`target`

```text
The thing the model should predict.
In A1, the target is Price (DKK).
```

`supervised`

```text
The model learns from examples where the correct answer is already known.
In A1, every car has a known real price.
```

`regression`

```text
Predicting a continuous number.
In A1, the number is the car price.
```

`classification`

```text
Predicting a class/category.
In A1, the classes are Cheap and Expensive.
```

`binary classification`

```text
Classification with two possible classes.
Here: Cheap or Expensive.
```

## Exam Sentences

```text
This assignment is mainly about predicting electric vehicle prices.
The first part is supervised regression because the target is a continuous price value.
The final part turns the same data into binary classification by creating Cheap and Expensive classes.
```

---

# 1. Features And Target

In this assignment, `X` contains the input features and `y` contains the target.

The target is:

```text
Price (DKK)
```

The features are all the other car columns, for example:

```text
Original Price
Model Year
Mileage
Electric Range
Acceleration
Battery Capacity
```

The model uses those features to learn patterns and predict the car price.

## Terms From This Section

`feature`

```text
An input variable used by the model.
Examples: Mileage, Model Year, Electric Range.
```

`X`

```text
The feature matrix.
In A1, X is all columns except Price (DKK).
```

`y`

```text
The target vector.
In A1, y is Price (DKK).
```

`input`

```text
The information given to the model.
In A1, inputs are car attributes.
```

`labelled example`

```text
One training example with both input features and the correct answer.
In A1: one car with its features and known price.
```

## Exam Sentences

```text
X contains the input features, and y contains the true price values.
The target is Price in DKK, because this is the value I want to predict.
Features are the input variables used to predict the car price.
```

## Where In Code

```text
Where I drop Price (DKK) from X and keep Price (DKK) as y.
```

---

# 2. Train-Test Split

Before training the model, the data is split into training and test data.

In Assignment 1, the split is approximately:

```text
80% training
20% test
random_state = 42
```

The training set is used to fit the model.

The test set is used only for final evaluation, so it simulates unseen future data.

This is important because evaluating only on training data would give an overly optimistic result.

## Terms From This Section

`training set`

```text
The data used to teach the model.
```

`test set`

```text
The data used to check final performance.
The model should not learn from it.
```

`fit`

```text
To train or learn from data.
When we call fit(), the model learns parameters.
```

`evaluation`

```text
Checking how good the model is.
```

`generalization`

```text
How well the model works on new unseen data.
```

`random_state`

```text
A fixed random seed.
It makes the split reproducible.
```

## Exam Sentences

```text
I split the data before fitting models or scalers because the test set should simulate unseen future data.
If I evaluated only on training data, the result would be too optimistic.
The split helps estimate generalization.
```

## Where In Code

```text
Where I use train_test_split with random_state 42.
```

---

# 3. Linear Regression

Linear regression predicts a continuous target as a weighted sum of input features plus an intercept.

The idea is:

```text
predicted price = intercept + w1*x1 + w2*x2 + ... + wn*xn
```

Each feature gets a coefficient.

For example, mileage may decrease predicted price, while newer model year or higher original price may increase predicted price.

In Assignment 1, linear regression is first implemented manually with NumPy, and later with Scikit-learn.

## Terms From This Section

`linear regression`

```text
A model that predicts a number using a linear combination of features.
```

`continuous target`

```text
A numerical value, not a category.
Price is continuous.
```

`weighted sum`

```text
Features multiplied by weights and added together.
```

`coefficient`

```text
The weight of a feature.
It tells how much the prediction changes when that feature changes, assuming other features stay fixed.
```

`intercept`

```text
The baseline value of the model when all features are zero.
```

## Exam Sentences

```text
Linear regression predicts a continuous target as a weighted sum of input features plus an intercept.
A coefficient tells how much the predicted price changes when one feature changes, assuming other features stay fixed.
The intercept is the baseline value of the model when all features are zero.
```

## Where In Code

```text
In the NumPy linear regression section and later in the Scikit-learn LinearRegression section.
```

---

# 4. Manual Linear Regression: Column Of Ones And Theta

In the manual NumPy implementation, a column of ones is added to the feature matrix.

This allows the model to learn an intercept.

The model parameters are stored in `theta`.

Simple map:

```text
theta = [intercept, coefficient_1, coefficient_2, ...]
```

Without the column of ones, the model would be forced through zero, meaning that if all features were zero, the prediction would also have to be zero.

## Terms From This Section

`column of ones`

```text
An artificial column filled with 1s.
It lets the model learn the intercept.
```

`theta`

```text
The vector of model parameters.
It contains the intercept and coefficients.
```

`parameter`

```text
A number learned by the model.
In linear regression, parameters are intercept and coefficients.
```

`forced through zero`

```text
The model has no intercept, so the line/hyperplane must pass through zero.
This is usually too restrictive.
```

## Exam Sentences

```text
I add a column of ones so the model can learn an intercept.
Without it, the model would be forced through zero.
Theta is the vector of model parameters, meaning the intercept and coefficients.
```

## Where In Code

```text
In the manual NumPy regression section, where I create the feature matrix with an added intercept column.
```

---

# 5. Pseudoinverse And Normal Equation

The normal equation is the theoretical formula for ordinary least squares:

```text
theta = (X.T X)^-1 X.T y
```

In the notebook, I use the Moore-Penrose pseudoinverse instead:

```python
np.linalg.pinv(X_train_b) @ y_train
```

This is more numerically stable than directly inverting `X.T X`, especially when features are correlated or the matrix is difficult to invert.

## Terms From This Section

`normal equation`

```text
The mathematical closed-form solution for linear regression.
```

`ordinary least squares`

```text
Linear regression method that minimizes squared errors.
```

`pseudoinverse`

```text
A safer matrix operation used to solve least-squares problems.
```

`numerically stable`

```text
Less likely to break or give bad results because of matrix/computation problems.
```

`correlated features`

```text
Features that contain similar information.
This can make matrix inversion harder.
```

## Exam Sentences

```text
I used the Moore-Penrose pseudoinverse because it gives a least-squares solution and is more numerically stable than directly inverting X transpose X.
```

## Where In Code

```text
In the manual linear regression section, where I calculate theta with NumPy instead of using Scikit-learn.
```

---

# 6. Scikit-Learn LinearRegression

After the manual implementation, I also use Scikit-learn `LinearRegression`.

The manual implementation shows the mathematical idea.

The Scikit-learn version is cleaner and follows the normal machine learning workflow:

```python
model.fit(X_train, y_train)
model.predict(X_test)
```

## Terms From This Section

`Scikit-learn`

```text
A Python machine learning library.
```

`fit`

```text
Train the model on data.
```

`predict`

```text
Use the trained model to produce predictions.
```

`workflow`

```text
The standard sequence of steps: split, fit, predict, evaluate.
```

## Exam Sentences

```text
The manual implementation shows the mathematical idea.
The Scikit-learn version is cleaner, more practical and fits the normal machine learning workflow with fit and predict.
```

## Where In Code

```text
In the section where I create LinearRegression(), fit it on X_train and y_train, and predict on X_test.
```

---

# 7. Regression Metrics

After making predictions, the model is evaluated with regression metrics.

In Assignment 1, the important metrics are:

```text
MSE
RMSE
R2
```

MSE is the average squared prediction error.

RMSE is the square root of MSE, so it is in the same unit as the target: DKK.

R2 tells how much of the variance in car prices is explained by the model.

In Assignment 1, RMSE is about `52,000 DKK` and R2 is about `0.864`.

## Terms From This Section

`prediction`

```text
The model output.
In A1, this is the predicted car price.
```

`error`

```text
The difference between true value and predicted value.
```

`MSE`

```text
Mean Squared Error.
Average squared prediction error.
Large mistakes are punished strongly.
```

`RMSE`

```text
Root Mean Squared Error.
It is easier to interpret because it is in the same unit as the target.
```

`R2`

```text
How much variance in the target is explained by the model.
R2 around 0.86 means about 86% of price variation is explained.
```

`variance`

```text
How much values differ from each other.
Here: how much car prices vary.
```

## Exam Sentences

```text
MSE is the average squared prediction error, so large mistakes are punished strongly.
RMSE is useful because it is in the same unit as the target, so I can interpret it as an average price error in DKK.
R2 around 0.86 means the model explains about 86% of the variation in prices.
```

Do not say:

```text
R2 means 86% predictions are correct.
```

Say:

```text
R2 means about 86% of the variation in prices is explained by the model.
```

## Where In Code

```text
After predictions, where I calculate MSE, RMSE and R2.
```

---

# 8. Regularization: Ridge, Lasso And Elastic Net

Regularization adds a penalty for large coefficients.

The goal is to reduce overfitting by making the model simpler.

In Assignment 1, I compare:

```text
Ridge
Lasso
Elastic Net
```

Ridge uses an L2 penalty and shrinks coefficients toward zero.

Lasso uses an L1 penalty and can set some coefficients exactly to zero.

Elastic Net combines L1 and L2 regularization.

The parameter `alpha` controls the strength of regularization.

## Terms From This Section

`regularization`

```text
A penalty for large coefficients.
It can reduce overfitting.
```

`overfitting`

```text
The model performs well on training data but poorly on unseen data.
```

`underfitting`

```text
The model is too simple to capture the real pattern.
```

`Ridge`

```text
L2 regularization.
Shrinks coefficients toward zero.
```

`Lasso`

```text
L1 regularization.
Can set coefficients exactly to zero, so it can act like feature selection.
```

`Elastic Net`

```text
Combination of L1 and L2 regularization.
```

`alpha`

```text
Regularization strength.
Higher alpha means stronger penalty.
```

## Exam Sentences

```text
Regularization adds a penalty for large coefficients. It can reduce overfitting by making the model simpler, but too much regularization can cause underfitting.
Ridge uses an L2 penalty and shrinks coefficients toward zero.
Lasso uses an L1 penalty and can set some coefficients exactly to zero.
Elastic Net combines L1 and L2 regularization.
Alpha controls the strength of regularization.
```

## Where In Code

```text
In the Ridge, Lasso and Elastic Net section, where I test several alpha values after scaling.
```

---

# 9. Scaling And Data Leakage

Before Ridge, Lasso, Elastic Net and kNN, the features are scaled.

Scaling makes numerical features comparable.

This matters because some features have very different ranges:

```text
Mileage can be like 120000
Acceleration can be like 7.5
Model Year can be like 2021
```

For scaling, the scaler must be fitted only on training data.

The test data should only be transformed.

This avoids data leakage.

Correct workflow:

```python
scaler.fit_transform(X_train)
scaler.transform(X_test)
```

## Terms From This Section

`scaling`

```text
Putting features on a comparable numeric scale.
```

`standardization`

```text
A common scaling method where values are transformed using mean and standard deviation.
```

`StandardScaler`

```text
Scikit-learn tool for standardizing features.
```

`fit_transform`

```text
Learn the transformation and apply it.
Used on training data.
```

`transform`

```text
Apply an already learned transformation.
Used on test data.
```

`data leakage`

```text
When information from validation/test data influences training.
This makes evaluation too optimistic.
```

## Exam Sentences

```text
Scaling makes features comparable, so features with large numeric ranges do not dominate the model.
I use fit_transform on training data because the scaler learns the training mean and standard deviation there.
I use transform on test data because the test set should not influence preprocessing.
Data leakage happens when information from the test set influences training, making evaluation too optimistic.
```

## Where In Code

```text
Where I use StandardScaler, fitted on train and applied to test.
```

---

# 10. kNN Classification

At the end of Assignment 1, the regression problem is changed into binary classification.

The notebook creates two classes using the median price:

```text
price > median -> Expensive
price <= median -> Cheap
```

Then kNN is trained to classify cars as Cheap or Expensive.

kNN classifies a new point by looking at the `k` nearest training examples and choosing the majority class.

Because kNN is distance-based, scaling is very important.

## Terms From This Section

`kNN`

```text
k nearest neighbors.
A classifier that looks at nearby training examples.
```

`k`

```text
The number of nearest neighbors used for prediction.
```

`nearest neighbors`

```text
The most similar training examples according to a distance metric.
```

`majority class`

```text
The class that appears most often among the neighbors.
```

`median split`

```text
Using the median price to divide cars into Cheap and Expensive.
```

`distance-based`

```text
The algorithm depends on distances between points.
```

## Exam Sentences

```text
I used the median price to create two almost balanced classes: Cheap and Expensive.
kNN classifies a new point by looking at the k nearest training examples and choosing the majority class.
kNN needs scaling because it is distance-based.
```

## Where In Code

```text
In the final classification section, after creating Cheap and Expensive labels from the median price.
```

---

# 11. Choosing k And Distance Metric

The value of `k` controls how many neighbors are used.

A small `k` is flexible but can overfit because it reacts strongly to individual observations.

A large `k` is smoother but can underfit because it may ignore local patterns.

The distance metric defines what "near" means.

In Assignment 1, different metrics are tested, such as:

```text
Euclidean
Manhattan
Minkowski
```

## Terms From This Section

`small k`

```text
Few neighbors.
Sensitive to noise, can overfit.
```

`large k`

```text
Many neighbors.
Smoother, but can underfit.
```

`distance metric`

```text
The rule for measuring similarity between cars.
```

`Euclidean distance`

```text
Straight-line distance.
```

`Manhattan distance`

```text
Distance measured as summed absolute differences.
```

## Exam Sentences

```text
A small k can overfit because it reacts strongly to individual observations.
A large k can underfit because it smooths the decision boundary too much.
The distance metric defines how similarity between cars is measured.
```

---

# 12. Classification Metrics

For the kNN classification part, the notebook uses classification metrics.

Accuracy tells how many cars were classified correctly overall.

The confusion matrix shows which types of mistakes were made.

Precision tells how many predicted positives were actually positive.

Recall tells how many actual positives were found.

F1-score balances precision and recall.

In Assignment 1, the kNN classifier reaches about 94% accuracy.

## Terms From This Section

`accuracy`

```text
Percentage of correct predictions overall.
```

`confusion matrix`

```text
A table showing correct and incorrect predictions per class.
```

`precision`

```text
Of the examples predicted as positive, how many were truly positive.
```

`recall`

```text
Of the actual positive examples, how many were found.
```

`F1-score`

```text
A balance between precision and recall.
```

## Exam Sentences

```text
Accuracy tells how many cars were classified correctly overall.
The confusion matrix shows the actual mistake types.
F1-score balances precision and recall.
```

---

# 13. Limitations

The regression models are useful baselines, but they are mostly linear.

That means they may miss nonlinear relationships between car features and price.

The RMSE is still around 52,000 DKK, so individual predictions can be far from the real price.

Regularization only slightly improves performance, so the ordinary linear regression model was already a strong baseline.

For kNN, the main limitation is that it depends strongly on scaling, distance metric and the value of `k`.

Also, the Cheap/Expensive target is artificial because it was created from the median price.

## Terms From This Section

`baseline`

```text
A simple reference model used for comparison.
```

`linear`

```text
Based mostly on straight-line relationships.
```

`nonlinear relationship`

```text
A more complex relationship that a simple linear model may not capture.
```

`artificial target`

```text
A target created manually for the assignment, not a natural label from the real world.
```

## Exam Sentences

```text
The main limitation is that the regression models are mostly linear, so they may miss nonlinear relationships in car prices.
Also, the RMSE is still around 52,000 DKK, which means individual predictions can be far from the real price.
The Cheap versus Expensive target is artificial because I created it from the median price.
```

---

# 14. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Data loading:
at the beginning, where I read car_prices.xlsx with pandas.

X and y:
where I drop Price (DKK) from X and keep Price (DKK) as y.

Train-test split:
where I use train_test_split with random_state 42.

Manual regression:
the NumPy section with the intercept column and pseudoinverse.

Library regression:
the Scikit-learn LinearRegression section.

Metrics:
after predictions, where I calculate MSE, RMSE and R2.

Regularization:
the Ridge, Lasso and Elastic Net section after scaling.

Scaling:
where I use StandardScaler, fitted on train and applied to test.

kNN:
the final classification section, after creating Cheap/Expensive labels.
```

---

# 15. A1 In 30 Seconds

```text
Assignment 1 is about predicting electric vehicle prices.
The main part is supervised regression, where the target is Price in DKK and the features are car attributes.

I first split the data into training and test sets.
Then I implemented linear regression manually with NumPy and also with Scikit-learn.
I evaluated it using MSE, RMSE and R2.

After that, I compared Ridge, Lasso and Elastic Net with scaled data.
Finally, I converted price into Cheap and Expensive classes using the median price and trained a kNN classifier.
```

---

# 16. Emergency Speaking Pattern

If you forget a formal definition, use this pattern:

```text
[Term] means [simple meaning].
In my assignment, I used it for [specific thing].
The reason is [why].
```

Example:

```text
Scaling means putting features on a comparable scale.
In my assignment, I used it before regularized regression and kNN.
The reason is that these methods are sensitive to feature scale.
```

---

# 17. Top Words To Memorize

```text
feature = input column
target = value to predict
X = features
y = target
coefficient = weight of a feature
intercept = baseline value
theta = model parameters
RMSE = typical error in target units
regularization = penalty for large coefficients
data leakage = test information enters training
kNN = nearest-neighbor classifier
scaling = putting features on comparable scale
```

Final survival sentence:

```text
The key thing in this notebook is that I understand what is fitted on training data, what is evaluated later, and why this avoids leakage or overfitting.
```

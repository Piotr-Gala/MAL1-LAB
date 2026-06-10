# Assignment 1: Car Prices - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment1.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This is mainly a supervised regression task. The model learns from labelled examples where the input features describe an electric vehicle and the target is a continuous value: price. Later, the task is also converted into a binary classification problem for kNN.

**How used in the assignment:** The notebook loads `car_prices.xlsx`, uses `Price (DKK)` as `y`, and uses the remaining columns as `X`. It first predicts price with linear regression and regularized regression, then creates cheap/expensive classes using the median price for kNN classification.

## 1. ML Workflow

**Question:** What is the ML workflow in this notebook?

**Theory answer:** A basic ML workflow is: understand the data, define features and target, split data, preprocess if needed, train models, tune hyperparameters, evaluate on unseen data, and discuss limitations.

**How used in the assignment:** The notebook inspects the dataset, creates `X` and `y`, uses an 80/20 train-test split with `random_state=42`, trains regression models, compares metrics, and ends with limitations.

## 2. Train-Test Split

**Question:** Why do you split into training and test data?

**Theory answer:** The training set is used to fit the model. The test set estimates generalization to unseen data. If we evaluate only on training data, the result can be too optimistic because the model has already seen those examples.

**How used in the assignment:** All regression models are fitted on the training set and evaluated on the test set. The test set is not used for fitting coefficients or scalers.

## 3. Linear Regression

**Question:** What does linear regression do?

**Theory answer:** Linear regression predicts a continuous target as a weighted sum of input features plus an intercept. It tries to find the line or hyperplane that minimizes squared prediction errors.

**How used in the assignment:** The notebook predicts `Price (DKK)` from car features. The model learns coefficients for each feature, meaning each coefficient describes how the predicted price changes when that feature changes, assuming other features stay fixed.

## 4. Linear Algebra Version

**Question:** What is the normal equation?

**Theory answer:** The normal equation is a closed-form solution for ordinary least squares: `theta = (X.T X)^-1 X.T y`. `X` is the design matrix, `theta` is the parameter vector, and `y` is the observation vector.

**How used in the assignment:** The notebook manually builds the design matrix by adding an intercept column and computes coefficients using NumPy. This shows the mathematical version of linear regression instead of only calling a library model.

## 5. Library Linear Regression

**Question:** Why also use Scikit-learn if you already implemented linear regression manually?

**Theory answer:** Manual implementation shows the theory. A library implementation is more robust, easier to reuse, and fits the normal ML workflow with `.fit()` and `.predict()`.

**How used in the assignment:** The notebook trains `LinearRegression()` from Scikit-learn and evaluates it with the same regression metrics. This connects the linear algebra formula to practical library usage.

## 6. Regression Metrics

**Question:** What are MSE, RMSE, MAE, and R-squared?

**Theory answer:** MSE is the average squared error, so large mistakes are punished strongly. RMSE is the square root of MSE, so it is in the same unit as the target. MAE is the average absolute error and is easier to interpret. R-squared explains how much variance in the target is explained by the model.

**How used in the assignment:** The notebook reports regression metrics for price prediction. RMSE and MAE are in DKK, so they can be explained as average prediction error in price units. R-squared gives a relative view of fit quality.

## 7. Regularization

**Question:** What is regularization?

**Theory answer:** Regularization adds a penalty for large model coefficients. It reduces overfitting by making the model simpler, but too much regularization can cause underfitting.

**How used in the assignment:** The notebook compares Ridge, Lasso, and Elastic Net with different `alpha` values. `alpha` controls penalty strength: larger `alpha` means stronger regularization.

## 8. Ridge, Lasso, Elastic Net

**Question:** What is the difference between Ridge, Lasso, and Elastic Net?

**Theory answer:** Ridge uses an L2 penalty and shrinks coefficients toward zero. Lasso uses an L1 penalty and can set some coefficients exactly to zero, so it can act like feature selection. Elastic Net combines L1 and L2 penalties.

**How used in the assignment:** The notebook trains all three models on scaled features and compares their test performance. This shows whether regularization improves generalization compared with ordinary linear regression.

## 9. Scaling

**Question:** Why do you scale features?

**Theory answer:** Scaling makes numerical features comparable. It is important for algorithms based on distances or coefficient penalties, because large-scale features can dominate the model.

**How used in the assignment:** Scaling is used before Ridge, Lasso, Elastic Net, and kNN. The scaler is fitted on training data and then applied to test data, which avoids data leakage.

## 10. Data Leakage

**Question:** What would data leakage look like here?

**Theory answer:** Data leakage happens when information from validation/test data influences training. It makes evaluation too optimistic.

**How used in the assignment:** The correct approach is to fit preprocessing only on the training data. In this notebook, scaling is not learned from the full dataset before splitting.

## 11. kNN Classification

**Question:** How does kNN classification work?

**Theory answer:** kNN classifies a new point by looking at the `k` nearest training points according to a distance metric, usually Euclidean distance. The majority class among neighbors becomes the prediction.

**How used in the assignment:** The notebook converts price into two classes, cheap and expensive, using the median price. Then it trains a kNN classifier to predict the price class from car features.

## 12. Choosing k

**Question:** Why does the value of `k` matter in kNN?

**Theory answer:** Small `k` can overfit because predictions react strongly to noise. Large `k` can underfit because local structure is smoothed too much. Odd `k` is often preferred in binary classification to reduce ties.

**How used in the assignment:** The notebook tests kNN after scaling and evaluates classification performance. The key explanation is that kNN is distance-based, so both `k` and scaling matter.

## 13. Classification Metrics

**Question:** Why not only use accuracy?

**Theory answer:** Accuracy can be misleading if classes are imbalanced. Precision measures how many predicted positives are correct. Recall measures how many actual positives were found. F1 balances precision and recall.

**How used in the assignment:** In the kNN part, the notebook evaluates the cheap/expensive classifier using classification metrics and a confusion-matrix style interpretation.

## 14. Limitations

**Question:** What are the limitations of this assignment?

**Theory answer:** Linear models assume mostly linear relationships. Price data can contain outliers, correlated features, missing market context, and nonlinear effects. kNN can be sensitive to scaling, irrelevant features, and the choice of `k`.

**How used in the assignment:** The notebook explicitly discusses limitations: price prediction is simplified, the dataset may not contain every important factor, and classification by median price is artificial but useful for demonstrating kNN.

## Fast Last-Minute Answers

- **Target:** `Price (DKK)`.
- **Main task:** supervised regression.
- **Manual theory part:** normal equation for linear regression.
- **Library part:** Scikit-learn `LinearRegression`, Ridge, Lasso, Elastic Net, kNN.
- **Most important risk:** data leakage from fitting preprocessing on all data.
- **Best one-sentence defense:** I compare a theory-first linear regression implementation with practical library models, evaluate on unseen data, then show how the same dataset can be reframed as a binary kNN classification problem.

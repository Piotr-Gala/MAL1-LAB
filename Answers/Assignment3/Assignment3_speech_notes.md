# Assignment 3: Mushroom Foraging - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with the mushroom dataset.
The goal was to build a supervised machine learning pipeline that classifies mushrooms as edible or poisonous.

I started with exploratory data analysis: I checked the dataset shape, column types, missing values, target distribution and duplicate rows.
Then I split the data into training, validation and test sets using stratification before fitting any preprocessing steps.

After the split, I selected columns with more than 80% missing values using only the training data and removed the same columns from validation and test sets.
For preprocessing, I used a pipeline with median imputation and standard scaling for numerical features, and most-frequent imputation with one-hot encoding for categorical features.

Finally, I trained Logistic Regression and compared three validation approaches: a single validation split, Stratified K-Fold cross-validation and nested cross-validation.
The final model achieved about 82% accuracy and about 0.83 F1-score for the poisonous class, but I treat the result carefully because false negatives are dangerous in this problem.
```

## Cells 1-4: Title And Imports

```text
At the beginning I import the libraries needed for the full machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for splitting, preprocessing, modelling and evaluation.
```

## Cells 5-8: Load Data

```text
In this step I load the mushroom dataset from a CSV file using a semicolon delimiter.
Then I check the dataset shape, data types, missing values and summary statistics.
The dataset contains 61069 rows and 21 columns.
```

## Cells 9-12: Initial Overview

```text
Here I inspect the column data types, missing values and target distribution.
The target variable is class, where p means poisonous and e means edible.
The dataset is slightly imbalanced, with about 55.5% poisonous mushrooms and 44.5% edible mushrooms.
```

## Cells 13-21: Data Cleaning And Preparation

```text
In this section I check duplicates, prepare X and y, and split the data before preprocessing.
This is important because fitting preprocessing steps before the split would cause data leakage.

I use stratified train-validation-test splitting, so the edible and poisonous class proportions stay similar in all subsets.
Then I calculate missing-value ratios only on the training data.
Columns with more than 80% missing values in the training set are removed from train, validation and test sets.
```

## Cells 23-25: Preprocessing And Model

```text
I create separate preprocessing pipelines for numerical and categorical features.
Numerical features are imputed with the median and scaled with StandardScaler.
Categorical features are imputed with the most frequent value and encoded with OneHotEncoder.

The preprocessing is combined with Logistic Regression in one pipeline.
This means the transformations are fitted only on the training data during validation and then applied to validation or test data.
```

## Cells 26-33: Single Validation Split

```text
The first validation method is a single validation split.
I train Logistic Regression with several C values and choose the one with the best F1-score for the poisonous class.

The best C from this split is 0.1.
The validation F1-score for the poisonous class is about 0.83, and recall for the poisonous class is about 0.825.
The confusion matrix shows that some poisonous mushrooms are still predicted as edible, which is the most dangerous error here.
```

## Cells 34-38: Stratified K-Fold Cross-Validation

```text
The second validation method is Stratified K-Fold cross-validation.
It trains and evaluates the model on several different folds while preserving the class distribution in each fold.

I use GridSearchCV to tune the C hyperparameter using F1-score for the poisonous class.
The best C is 0.1 and the mean cross-validation F1-score is about 0.830.
This method is more stable than a single validation split, but it takes more computation time.
```

## Cells 39-41: Nested Cross-Validation

```text
The third validation method is nested cross-validation.
The inner loop tunes the hyperparameter, and the outer loop estimates model performance on data not used for that inner tuning.

This gives a less optimistic estimate than normal cross-validation after hyperparameter tuning.
The nested CV mean F1-score is about 0.829.
The downside is that nested cross-validation is more computationally expensive.
```

## Cells 42-49: Final Model And Metrics

```text
After validation, I use the best model from GridSearchCV and fit it on the combined training and validation data.
Then I evaluate it once on the untouched test set.

On the test set, the model achieves about 0.817 accuracy, 0.842 precision, 0.824 recall and 0.833 F1-score for the poisonous class.
The ROC AUC is about 0.882.
I also plot the confusion matrix, ROC curve and precision-recall curve to inspect performance in more detail.
```

## Cells 50-51: Interpretation And Limitations

```text
The final model performs reasonably well, but it still predicts some poisonous mushrooms as edible.
These false negatives are the most important limitation, because they would be dangerous in a real mushroom-foraging context.

For this task, recall for the poisonous class is the most important metric.
F1-score is also useful because it balances recall and precision.
Accuracy alone is not enough because it treats all mistakes as equally important.
```

## Metrics Speech

```text
In this task, precision tells me how many mushrooms predicted as poisonous are actually poisonous.
Recall tells me how many actual poisonous mushrooms the model detects.
F1-score balances precision and recall.

A false positive means that an edible mushroom is predicted as poisonous.
A false negative means that a poisonous mushroom is predicted as edible.
False negatives are more dangerous here, so recall for the poisonous class is the key metric.
```

## Final Sentence

```text
The model is useful for learning and comparison, but because it still produces false negatives, it should not be treated as safe for real-world mushroom foraging.
```

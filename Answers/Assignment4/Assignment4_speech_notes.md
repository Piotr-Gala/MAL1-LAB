# Assignment 4: Detecting Exoplanets - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with the Kepler exoplanet dataset.
The goal was to build a supervised machine learning pipeline that classifies Kepler objects as planet candidates or false positives.

I started with exploratory data analysis: I checked the dataset shape, column types, missing values, target distribution, outliers and correlations.
Then I cleaned the data by removing completely empty columns and identifier columns, encoded the target labels, and prepared the feature matrix.

For modelling, I split the data into training, validation and test sets using stratification.
After the split, I applied median imputation and standard scaling, fitting both steps only on the training data to avoid data leakage.

Finally, I trained and compared Logistic Regression and SVM models.
Logistic Regression performed slightly better on the final test set, with about 90% accuracy and F1-score.
I also treated the result carefully, because some false-positive flag features may act as proxy information for the target label.
```

## Cells 1-3: Title And Imports

```text
At the beginning I import the libraries needed for the whole machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for preprocessing, modelling and evaluation.
```

## Cells 4-5: Load Data

```text
In this step I load the exoplanet dataset from a CSV file into a pandas DataFrame.
Then I check the shape of the dataset and display the first few rows to make sure the data was loaded correctly.
The dataset contains 9564 rows and 49 columns.
```

## Cells 6-8: Initial Overview And Renaming

```text
After loading the data, I perform an initial overview.
I use info() to check column types and missing values, and describe() to see basic statistics.
Then I rename the columns to make them easier to understand.
```

## Cells 9-13: Missing Values And Target Encoding

```text
I calculated missing value percentages for all columns.
Then I removed two columns with 100% missing values and several identifier columns.
After that, I encoded the disposition labels numerically.
The final target based on Kepler data is almost balanced, with false positives and candidates in similar numbers.
```

## Cells 14-18: Outliers And Missing Values Before Split

```text
I analyzed outliers using the IQR method, but I decided not to remove them because extreme astronomical values may be valid observations.
I also checked that dropping all rows with missing values would remove 1761 rows, so imputation is a better choice.
However, I postponed imputation until after the train-validation-test split to avoid data leakage.
```

## Cells 19-22: Prepare Features And Split

```text
I prepared X and y by selecting KeplerDispositionStatus as the target and removing target-related columns from the features.
Then I split the data into train, validation and test sets using stratification.
This keeps the class distribution similar in all subsets and prepares the data for train-only feature selection.
```

## Cells 23-28: Train-Only Correlation, Imputation And Scaling

```text
I analyzed correlations only on the training features and removed two highly correlated columns from all splits.
This avoids using validation or test information during feature selection.
After that, I applied median imputation and standard scaling.
Both preprocessing steps were fitted only on the training set to avoid data leakage.
```

## Metrics Speech

```text
In this task, precision tells me how reliable the predicted planet candidates are.
Recall tells me how many actual candidates the model managed to detect.
F1-score balances these two aspects.
A false positive means that a false-positive object was predicted as a candidate, while a false negative means that a real candidate was missed.
```

## Cells 29-34: Logistic Regression

```text
I trained Logistic Regression with several C values and selected the one with the best validation F1-score.
The best C was 0.1.
The model achieved very high validation performance, with F1 around 0.994.
I also checked the confusion matrix to see the types of errors.
However, I interpret the result carefully because some false-positive flags may be strongly related to the target.
```

## Cells 35-39: SVM

```text
I trained and tuned an SVM model using different C values and two kernels: linear and RBF.
The best model used a linear kernel with C equal to 0.01.
It achieved a validation F1-score around 0.993, which is very high but slightly lower than Logistic Regression.
I also used a confusion matrix to inspect the classification errors.
```

## Cells 40-45: Final Model, Test Evaluation And Limitations

```text
After tuning, I retrained both models on the combined training and validation data and evaluated them on the untouched test set.
Logistic Regression achieved about 90.2% accuracy and F1-score, while SVM achieved about 89.9%.
I selected Logistic Regression because it was slightly better and easier to explain.
However, I mention one important limitation: some false-positive flags may act as proxy variables for the target.
```

## Final Sentence

```text
The model performs well, but I treat the result carefully because the dataset contains features that may be strongly related to the labelling process itself.
```

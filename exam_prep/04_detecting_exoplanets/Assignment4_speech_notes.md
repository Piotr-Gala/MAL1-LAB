# Assignment 4: Detecting Exoplanets - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with the Kepler exoplanet dataset.
The goal was to build a supervised machine learning pipeline that classifies Kepler objects as planet candidates or false positives.

I started with exploratory data analysis: I checked the dataset shape, column types, missing values, target distribution, outliers and correlations.
Then I cleaned the data by removing completely empty columns and identifier columns, encoded the target labels, and prepared the feature matrix.

For modelling, I split the data into training, validation and test sets using stratification.
Before modelling, I removed target-related columns and the explicit false-positive flag columns from the feature matrix.
This made the experiment more conservative, because the model could not rely on variables that are very close to the target labelling process.

After the split, I applied median imputation and standard scaling, fitting both steps only on the training data to avoid data leakage.
Finally, I trained and compared Logistic Regression and SVM models.
SVM performed better on the final test set, with about 85.5% accuracy and F1-score.
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
I prepared X and y by selecting KeplerDispositionStatus as the target.
I removed target-related columns, including DispositionScore and ArchiveDispositionStatus.
I also removed the four explicit false-positive flag columns before modelling.
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
The best C was 10.
The model achieved about 80.3% validation F1-score.
On the final test set it achieved about 82.1% accuracy and about 0.820 F1-score.
This is lower than the original flag-based version, but it is more realistic because the model cannot use explicit false-positive flags.
```

## Cells 35-39: SVM

```text
I trained and tuned an SVM model using different C values and two kernels: linear and RBF.
The best model used an RBF kernel with C equal to 10.
It achieved about 82.6% validation F1-score.
On the final test set it achieved about 85.5% accuracy and about 0.855 F1-score.
This was the best final model.
```

## Cells 40-45: Final Model, Test Evaluation And Limitations

```text
After tuning, I retrained both models on the combined training and validation data and evaluated them on the untouched test set.
Logistic Regression achieved about 82.1% accuracy and 0.820 F1-score.
SVM achieved about 85.5% accuracy and 0.855 F1-score.
I selected SVM because it performed better on the final test set.
The main limitation is that removing explicit false-positive flags reduces proxy risk, but it does not prove that all indirect proxy information is removed.
```

## Final Sentence

```text
After removing the explicit false-positive flag columns, the model performs less perfectly but the result is easier to defend because it relies more on measurement-based features.
```

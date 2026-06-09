# Assignment 4: Detecting Exoplanets - Oral Exam Notes

## Opening Speech

```text
Good morning. In this assignment I worked with the Kepler exoplanet dataset.
The goal was to build a supervised machine learning pipeline that classifies Kepler objects as planet candidates or false positives.

I started with exploratory data analysis: I checked the dataset shape, column types, missing values, target distribution, outliers and correlations.
Then I cleaned the data by removing completely empty columns and identifier columns, encoded the target labels, and prepared the feature matrix.

Before modelling, I removed target-related columns and the explicit false-positive flag columns.
This was important because those flags are very close to the target labelling process and could give the model an unrealistic shortcut.

For modelling, I split the data into training, validation and test sets using stratification.
After the split, I applied correlation filtering, median imputation and standard scaling in a leakage-safe way.
Finally, I trained and compared Logistic Regression and SVM models.
SVM performed better on the final test set, with about 85.5% accuracy and F1-score.
```

---

## Cells 1-3: Imports

**What to say**

```text
At the beginning I import the libraries needed for the whole machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for preprocessing, modelling and evaluation.
```

**Most important**

- `pandas` - working with tabular data.
- `numpy` - numerical operations.
- `matplotlib`, `seaborn` - plots and visualizations.
- `train_test_split` - data splitting.
- `SimpleImputer` - filling missing values.
- `StandardScaler` - feature scaling.
- `LogisticRegression`, `SVC` - models.
- metrics - model evaluation.

**Question**

```text
Why do you use StandardScaler?
```

**Answer**

```text
I use StandardScaler because Logistic Regression and SVM are sensitive to feature scale. Scaling makes features comparable.
```

---

## Cells 4-5: Loading The Dataset

**What to say**

```text
In this step I load the exoplanet dataset from a CSV file into a pandas DataFrame.
Then I check the shape of the dataset and display the first few rows to make sure the data was loaded correctly.
```

**Notebook details**

```text
9564 rows and 49 columns
```

**Question**

```text
What does one row represent?
```

**Answer**

```text
One row represents one Kepler object of interest, with its measured astronomical properties and disposition labels.
```

---

## Cells 6-8: Initial Overview And Renaming Columns

**What to say**

```text
After loading the data, I perform an initial overview.
I use info() to check column types and missing values, and describe() to see basic statistics.
Then I rename the columns to make them easier to understand.
```

**Concrete details**

- the dataset has 49 columns,
- there are numerical and text columns,
- some columns have missing values,
- `koi_teq_err1` and `koi_teq_err2` are completely empty.

**Note**

There is a typo in the notebook: `ImpactParamete`. It is only an error in the column name and does not change the data or the model.

---

## Cells 9-13: Missing Values And Encoding Target

**What to say**

```text
I calculated missing value percentages for all columns.
Then I removed two columns with 100% missing values and several identifier columns.
After that, I encoded the disposition labels numerically.
The final target based on Kepler data is almost balanced, with false positives and candidates in similar numbers.
```

**Removed columns**

- `EquilibriumTemperatureUpperUnc, K`
- `EquilibriumTemperatureLowerUnc, K`
- `KepID`
- `KOIName`
- `KeplerName`
- `TCEDeliver`

**Encoding**

```text
FALSE POSITIVE -> 0
CANDIDATE -> 1
CONFIRMED -> 2
```

**Target details**

```text
KeplerDispositionStatus:
0 -> 4847
1 -> 4717
```

This is an almost balanced binary classification problem.

**Question**

```text
Why remove ID columns?
```

**Answer**

```text
Because identifiers do not describe physical properties and may cause the model to memorize objects instead of learning patterns.
```

---

## Cells 14-18: Outliers And Missing Values Before Split

**What to say**

```text
I analyzed outliers using the IQR method, but I decided not to remove them because extreme astronomical values may be valid observations.
I also checked that dropping all rows with missing values would remove 1761 rows, so imputation is a better choice.
However, I postponed imputation until after the train-validation-test split to avoid data leakage.
```

**Dropna check**

```text
Original shape: (9564, 43)
Shape after dropna: (7803, 43)
Rows removed: 1761
```

**Question**

```text
Why is IQR not ideal for binary flags?
```

**Answer**

```text
Because binary flags only have values 0 and 1, so the IQR method may mark valid flag values as outliers.
In this notebook the flags are still visible during EDA, but they are removed before modelling.
```

---

## Cells 19-22: Features, Proxy Removal And Split

**What to say**

```text
I prepared X and y by selecting KeplerDispositionStatus as the target.
Then I removed target-related columns and the explicit false-positive flag columns from X.
After that, I split the data into train, validation and test sets using stratification.
```

**X and y**

```text
X = features
y = target
```

**Target**

```text
KeplerDispositionStatus
```

**Removed target-related columns**

- `DispositionScore`
- `KeplerDispositionStatus`
- `ArchiveDispositionStatus`

**Removed false-positive flag columns**

- `NotTransit-LikeFalsePositiveFlag`
- `koi_fpflag_ss`
- `CentroidOffsetFalsePositiveFlag`
- `EphemerisMatchIndicatesContaminationFalsePositiveFlag`

**Shapes**

```text
X shape before correlation filtering: (9564, 36)
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

**Question**

```text
Why remove the false-positive flags?
```

**Answer**

```text
Because they are likely very close to the target labelling process.
If the model uses them, it may learn a shortcut instead of learning from physical and measurement-based features.
```

**Question**

```text
Did removing the flags make the model worse?
```

**Answer**

```text
Numerically yes, the scores became lower.
But that is expected and useful, because the new result is more conservative and easier to defend.
```

---

## Cells 23-28: Train-Only Correlation, Imputation And Scaling

**What to say**

```text
After the split, I analyzed correlations only on the training features and removed two highly correlated columns from all splits.
Finally, I applied median imputation and standard scaling, fitting both preprocessing steps only on the training data to avoid data leakage.
```

**Removed because of correlation**

- `PlanetaryRadiusLowerUnc, Earthradii`
- `InsolationFluxLowerUnc, Earthflux`

**Shape after correlation filtering**

```text
X shape after correlation filtering: (9564, 34)
X_train: (6120, 34)
X_val: (1531, 34)
X_test: (1913, 34)
```

**Question**

```text
Was this step leakage-safe?
```

**Answer**

```text
Yes. Correlation-based feature selection is fitted on the training set only and the same selected columns are removed from validation and test sets.
```

**Question**

```text
Why fit imputer and scaler only on training data?
```

**Answer**

```text
To avoid using information from validation or test data during preprocessing.
```

---

## Metrics: FP, FN, Precision, Recall, F1

**Classes**

```text
0 = FALSE POSITIVE
1 = CANDIDATE
```

**Confusion matrix**

```text
                 Predicted 0        Predicted 1
Actual 0         TN                 FP
Actual 1         FN                 TP
```

**Accuracy**

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**

```text
precision = TP / (TP + FP)
```

When the model says "candidate", how often is it correct?

**Recall**

```text
recall = TP / (TP + FN)
```

How many true candidates did the model find?

**F1**

```text
F1 = 2 * precision * recall / (precision + recall)
```

Balance between precision and recall.

---

## Cells 29-34: Logistic Regression

**What to say**

```text
I trained Logistic Regression with several C values and selected the one with the best validation F1-score.
The best C was 10.
The validation F1-score was about 0.803.
On the final test set, Logistic Regression achieved about 82.1% accuracy and about 0.820 F1-score.
```

**Result**

```text
Best Logistic Regression C: 10
Train F1: about 0.804
Validation F1: about 0.803
Final test accuracy: about 0.821
Final test F1: about 0.820
```

**Question**

```text
What does C control?
```

**Answer**

```text
C controls regularization strength. A smaller C means stronger regularization, while a larger C means weaker regularization.
```

---

## Cells 35-39: SVM

**What to say**

```text
I trained and tuned an SVM model using different C values and two kernels: linear and RBF.
The best model used an RBF kernel with C equal to 10.
The validation F1-score was about 0.826.
On the final test set, SVM achieved about 85.5% accuracy and about 0.855 F1-score.
```

**Result**

```text
Best SVM C: 10
Best SVM kernel: rbf
Train F1: about 0.881
Validation F1: about 0.826
Final test accuracy: about 0.855
Final test F1: about 0.855
```

**Question**

```text
What is the difference between linear and RBF kernel?
```

**Answer**

```text
A linear kernel creates a linear decision boundary, while an RBF kernel can model nonlinear relationships.
```

**Question**

```text
Why did RBF perform best?
```

**Answer**

```text
It suggests that the class boundary may be nonlinear after removing the explicit false-positive flags.
```

---

## Cells 40-45: Final Model, Test Evaluation, Limitations

**What to say**

```text
After tuning, I retrained both models on the combined training and validation data and evaluated them on the untouched test set.
Logistic Regression achieved about 82.1% accuracy and 0.820 F1-score.
SVM achieved about 85.5% accuracy and 0.855 F1-score.
I selected SVM because it performed better on the final test set.
```

**Final results**

```text
Logistic Regression:
Accuracy: 0.821
F1-score: 0.820

SVM:
Accuracy: 0.855
F1-score: 0.855
```

**Selected model**

```text
SVM
```

Because it performs better on the final test set.

**Main limitation**

```text
The explicit false-positive flags were removed, which reduces target-proxy risk.
However, this does not prove that all indirect proxy information is gone.
Some remaining measurement features may still be related to the labelling process.
```

**Question**

```text
Why are the scores lower now?
```

**Answer**

```text
Because the model can no longer use the explicit false-positive flag columns.
The lower score is more realistic and easier to defend.
```

**Question**

```text
Which model was selected and why?
```

**Answer**

```text
I selected SVM because it achieved better final test performance than Logistic Regression.
```

**Final defense**

```text
The model performs less perfectly after removing the false-positive flags, but the result is more honest.
It better tests whether the remaining physical and measurement-based features can predict the class.
```

# Assignment 4: Detecting Exoplanets - Oral Exam Notes

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

---

## Cells 1-3: Imports

**What to say**

```text
At the beginning I import the libraries needed for the whole machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for preprocessing, modelling and evaluation.
```

**Plain English**

This is only tool setup. Nothing has happened to the data yet.

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
Why do you use pandas?
```

**Answer**

```text
I use pandas because the dataset is tabular, so a DataFrame is convenient for loading, inspecting and transforming the data.
```

**Question**

```text
Why do you use seaborn?
```

**Answer**

```text
I use seaborn to create clearer statistical visualizations, for example the correlation heatmap.
```

**Question**

```text
Why do you import evaluation metrics?
```

**Answer**

```text
Because after training the models I need to measure their performance using accuracy, precision, recall, F1-score and confusion matrices.
```

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

**Plain English**

`shape` shows how many rows and columns there are.  
`head()` shows the first rows.

**Question**

```text
Why do you check the shape of the dataset?
```

**Answer**

```text
I check the shape to know how many observations and features are available in the dataset.
```

**Question**

```text
Why do you use head()?
```

**Answer**

```text
I use head() to quickly inspect the first rows and verify that the dataset was loaded correctly.
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

**Plain English**

This checks what is in the table: data types, missing values and basic statistics.

**Concrete details**

- the dataset has 49 columns,
- there are numerical and text columns,
- some columns have missing values,
- `koi_teq_err1` and `koi_teq_err2` are completely empty.

**Question**

```text
What is the purpose of info()?
```

**Answer**

```text
info() shows the data types, number of non-null values and memory usage, so it helps identify missing values and categorical columns.
```

**Question**

```text
What is the purpose of describe()?
```

**Answer**

```text
describe() gives summary statistics, such as mean, standard deviation, minimum, maximum and quartiles.
```

**Question**

```text
Why did you rename the columns?
```

**Answer**

```text
I renamed the columns to make them more descriptive and easier to interpret during analysis.
```

**Question**

```text
Does renaming columns affect the model?
```

**Answer**

```text
No, renaming only changes column labels. It does not change the data values.
```

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

**Plain English**

First I check missing values, then remove irrelevant columns, then convert text labels into numbers.

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
Why analyze missing values first?
```

**Answer**

```text
Because missing values affect preprocessing decisions. Some columns may need to be removed, while others can be imputed.
```

**Question**

```text
Why remove columns with 100% missing values?
```

**Answer**

```text
Because they contain no information and cannot help the model.
```

**Question**

```text
Why remove ID columns?
```

**Answer**

```text
Because identifiers do not describe physical properties and may cause the model to memorize objects instead of learning patterns.
```

**Question**

```text
Why encode labels?
```

**Answer**

```text
Because machine learning models need numerical labels instead of text categories.
```

**Question**

```text
Is the target balanced?
```

**Answer**

```text
Yes, the Kepler disposition target is almost balanced: about 4847 false positives and 4717 candidates.
```

---

## Cells 14-18: Outliers And Missing Values Before Split

**What to say**

```text
I analyzed outliers using the IQR method, but I decided not to remove them because extreme astronomical values may be valid observations.
I also checked that dropping all rows with missing values would remove 1761 rows, so imputation is a better choice.
However, I postponed imputation until after the train-validation-test split to avoid data leakage.
```

**IQR**

```text
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower bound = Q1 - 1.5 * IQR
upper bound = Q3 + 1.5 * IQR
```

**Plain English**

The outliers are kept because in astronomy extreme values may be real observations, not errors.

**Dropna check**

```text
Original shape: (9564, 43)
Shape after dropna: (7803, 43)
Rows removed: 1761
```

**Question**

```text
What is an outlier?
```

**Answer**

```text
An outlier is a value that lies far away from the typical range of a feature.
```

**Question**

```text
How does the IQR method work?
```

**Answer**
```text
It uses the first and third quartiles. Values below Q1 minus 1.5 IQR or above Q3 plus 1.5 IQR are treated as outliers.
```

**Question**

```text
Why is IQR not ideal for binary flags?
```

**Answer**

```text
Because binary flags only have values 0 and 1, so the IQR method may mark valid flag values as outliers.
```

**Question**

```text
Why did you not remove outliers?
```

**Answer**

```text
Because extreme values in astronomical data may be valid physical observations, not errors. Removing them could remove useful information.
```

**Question**

```text
Why not use dropna()?
```

**Answer**

```text
Because it would remove 1761 rows, which is a significant amount of data.
```

**Question**

```text
What is data leakage?
```

**Answer**

```text
Data leakage happens when information from validation or test data is used during training or preprocessing.
```

---

## Cells 19-28: Features, Split, Train-Only Correlation, Imputation And Scaling

**What to say**

```text
I prepared X and y by selecting KeplerDispositionStatus as the target and removing target-related columns from the features.
Then I split the data into train, validation and test sets using stratification.
After the split, I analyzed correlations only on the training features and removed two highly correlated columns from all splits.
Finally, I applied median imputation and standard scaling, fitting both preprocessing steps only on the training data to avoid data leakage.
```

**Plain English**

First I create `X` and `y`, then split the data, then inspect correlations only on `X_train`. Only after that do I remove selected columns from all splits and apply imputation and scaling.

**X and y**

```text
X = features
y = target
```

**Target**

```text
KeplerDispositionStatus
```

**Removed from X**

- `DispositionScore`
- `KeplerDispositionStatus`
- `ArchiveDispositionStatus`

**Split**

```text
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

**Why stratify**

It preserves a similar class distribution in train, validation and test.

**Correlation analysis**

Correlation is calculated only on `X_train`, not on the full dataset. This is important because feature selection should not use information from the validation or test set.

**Removed because of correlation**

- `PlanetaryRadiusLowerUnc, Earthradii`
- `InsolationFluxLowerUnc, Earthflux`

**Correlation theory**

Pearson correlation:

```text
+1 = strong positive linear relation
 0 = no linear relation
-1 = strong negative linear relation
```

**Imputation**

Missing values are filled with the median. The median is better than the mean with outliers because it is more robust to extreme values.

**Scaling**

```text
x_scaled = (x - mean) / standard_deviation
```

Scaling is needed because Logistic Regression and SVM are sensitive to feature scales.

**Question**

```text
What are X and y?
```

**Answer**

```text
X is the feature matrix and y is the target variable.
```

**Question**

```text
Why remove ArchiveDispositionStatus?
```

**Answer**

```text
Because it is another disposition label and could leak information about the target.
```

**Question**

```text
Why remove DispositionScore?
```

**Answer**

```text
It is likely too directly related to the disposition label and could leak target information.
```

**Question**

```text
Why use train, validation and test sets?
```

**Answer**

```text
The training set is used to fit the model, the validation set to tune hyperparameters, and the test set for final unbiased evaluation.
```

**Question**

```text
Why use stratified split?
```

**Answer**

```text
To keep the class proportions similar in all subsets.
```

**Question**

```text
What is correlation?
```

**Answer**

```text
Correlation measures the strength and direction of a linear relationship between two variables.
```

**Question**

```text
Why did you use absolute correlation?
```

**Answer**

```text
Because both strong positive and strong negative correlations can indicate redundant information.
```

**Question**

```text
Why remove highly correlated features?
```

**Answer**

```text
To reduce redundancy and multicollinearity, especially for Logistic Regression.
```

**Question**

```text
What is multicollinearity?
```

**Answer**

```text
Multicollinearity occurs when two or more features are strongly correlated and carry similar information.
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
Why use median imputation?
```

**Answer**

```text
Median imputation is robust to outliers and allows us to keep rows with missing values.
```

**Question**

```text
Why fit imputer only on training data?
```

**Answer**

```text
To avoid using information from validation or test data during preprocessing.
```

**Question**

```text
What does StandardScaler do?
```

**Answer**

```text
It subtracts the mean and divides by the standard deviation for each feature.
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

**TP**

```text
Actual: CANDIDATE
Predicted: CANDIDATE
```

**TN**

```text
Actual: FALSE POSITIVE
Predicted: FALSE POSITIVE
```

**FP**

```text
Actual: FALSE POSITIVE
Predicted: CANDIDATE
```

False alarm.

**FN**

```text
Actual: CANDIDATE
Predicted: FALSE POSITIVE
```

The model missed a potential planet.

**Accuracy**

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Overall correctness.

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

**What to say**

```text
In this task, precision tells me how reliable the predicted planet candidates are.
Recall tells me how many actual candidates the model managed to detect.
F1-score balances these two aspects.
A false positive means that a false-positive object was predicted as a candidate, while a false negative means that a real candidate was missed.
```

**Intuition**

```text
Precision asks: when the model says "candidate", can I trust it?
Recall asks: did the model find most real candidates?
Accuracy asks: how often is the model correct overall?
F1 asks: is there a good balance between precision and recall?
```

**Mini example**

```text
TP = 90
TN = 80
FP = 10
FN = 20

accuracy = (90 + 80) / 200 = 0.85
precision = 90 / (90 + 10) = 0.90
recall = 90 / (90 + 20) = 0.818
F1 is about 0.857
```

**Most important**

```text
Precision cares about FP.
Recall cares about FN.
F1 balances both.
```

---

## Cells 29-34: Logistic Regression

**What to say**

```text
I trained Logistic Regression with several C values and selected the one with the best validation F1-score.
The best C was 0.1.
The model achieved very high validation performance, with F1 around 0.994.
I also checked the confusion matrix to see the types of errors.
However, I interpret the result carefully because some false-positive flags may be strongly related to the target.
```

**Plain English**

Logistic Regression is a classification model despite the word "regression" in its name.

**C**

```text
small C = stronger regularization
large C = weaker regularization
```

**Result**

```text
Best Logistic Regression C: 0.1
Validation F1: about 0.994
```

**Question**

```text
Why is Logistic Regression used for classification?
```

**Answer**

```text
Because it estimates class probabilities and then assigns observations to classes based on a decision boundary.
```

**Question**

```text
What does C control?
```

**Answer**

```text
C controls regularization strength. Smaller C means stronger regularization.
```

**Question**

```text
Why tune C?
```

**Answer**

```text
Because different regularization strengths can affect generalization performance.
```

**Question**

```text
Why use F1-score?
```

**Answer**

```text
F1-score balances precision and recall, so it is useful when we care about both false positives and false negatives.
```

**Question**

```text
What is a confusion matrix?
```

**Answer**

```text
A confusion matrix shows how many samples were classified correctly and incorrectly for each class.
```

**Question**

```text
What is overfitting?
```

**Answer**

```text
Overfitting happens when a model learns the training data too closely and performs poorly on unseen data.
```

**Question**

```text
Why should the high validation score be interpreted carefully?
```

**Answer**

```text
Because some false-positive flags may act as proxy information for the target label.
```

**Important note**

False-positive flags may act as proxies for the target, so the model may receive a very strong hint.

---

## Cells 35-39: SVM

**What to say**

```text
I trained and tuned an SVM model using different C values and two kernels: linear and RBF.
The best model used a linear kernel with C equal to 0.01.
It achieved a validation F1-score around 0.993, which is very high but slightly lower than Logistic Regression.
I also used a confusion matrix to inspect the classification errors.
```

**Plain English**

SVM looks for a decision boundary that separates classes with the largest possible margin.

**Kernel**

- `linear` - a linear decision boundary.
- `rbf` - a nonlinear, more flexible boundary.

**Result**

```text
Best SVM C: 0.01
Best SVM kernel: linear
Validation F1: about 0.993
```

**Question**

```text
What is SVM?
```

**Answer**

```text
SVM is a classification algorithm that tries to find a decision boundary with the maximum margin between classes.
```

**Question**

```text
What is a margin?
```

**Answer**

```text
The margin is the distance between the decision boundary and the closest training points.
```

**Question**

```text
What are support vectors?
```

**Answer**

```text
Support vectors are the closest points to the decision boundary, and they determine the position of that boundary.
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
Why did the linear kernel perform best?
```

**Answer**

```text
It suggests that the classes may be separated well using a relatively linear boundary in the scaled feature space.
```

**Question**

```text
What does C do in SVM?
```

**Answer**

```text
C controls the trade-off between a wider margin and fewer training errors. Smaller C means stronger regularization.
```

---

## Cells 40-45: Final Model, Test Evaluation, Limitations

**What to say**

```text
After tuning, I retrained both models on the combined training and validation data and evaluated them on the untouched test set.
Logistic Regression achieved about 90.2% accuracy and F1-score, while SVM achieved about 89.9%.
I selected Logistic Regression because it was slightly better and easier to explain.
However, I mention one important limitation: some false-positive flags may act as proxy variables for the target.
```

**Plain English**

After choosing hyperparameters, the final model is trained on `train + validation`, while the test set is kept only for the final evaluation.

**Final results**

```text
Logistic Regression:
Accuracy: 0.902
F1-score: 0.902

SVM:
Accuracy: 0.899
F1-score: 0.898
```

**Selected model**

```text
Logistic Regression
```

Because it is slightly better on the test set and easier to explain.

**Question**

```text
Why retrain on train plus validation?
```

**Answer**

```text
Because after hyperparameters are selected, using more training data can improve the final model.
```

**Question**

```text
Why is the test set used only at the end?
```

**Answer**

```text
Because it should represent unseen data and provide an unbiased estimate of final performance.
```

**Question**

```text
Which model was selected and why?
```

**Answer**

```text
I selected Logistic Regression because it achieved slightly better test performance and is simpler to interpret.
```

**Question**

```text
Why did validation performance differ from test performance?
```

**Answer**

```text
The validation result was probably optimistic. The test set gives a more realistic estimate of generalization.
```

**Question**

```text
Is 90% accuracy good?
```

**Answer**

```text
Yes, it is a strong result, but it should be interpreted carefully because of possible proxy features and dataset-specific limitations.
```

**Main limitation**

```text
Some false-positive flags may act as proxy information for the target.
```

**Final defense**

```text
The model performs well, but I treat the result carefully because the dataset contains features that may be strongly related to the labelling process itself.
```

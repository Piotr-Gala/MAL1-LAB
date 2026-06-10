# Assignment 3: Mushroom Foraging - Oral Exam Notes

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

---

## Cells 1-4: Title And Imports

**What to say**

```text
At the beginning I import the libraries needed for the full machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for splitting, preprocessing, modelling and evaluation.
```

**Plain English**

This is only tool setup. Nothing has happened to the data yet.

**Most important**

- `pandas` - working with tabular data.
- `numpy` - numerical operations.
- `matplotlib`, `seaborn` - plots and visualizations.
- `train_test_split` - data splitting.
- `StratifiedKFold` - cross-validation that preserves class proportions.
- `SimpleImputer` - filling missing values.
- `OneHotEncoder` - converting categories into numerical columns.
- `StandardScaler` - scaling numerical features.
- `LogisticRegression` - classification model.
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
I use seaborn to create clearer visualizations, for example confusion matrices.
```

**Question**

```text
Why do you import evaluation metrics?
```

**Answer**

```text
Because after training the model I need to measure its performance using accuracy, precision, recall, F1-score, ROC AUC and confusion matrices.
```

---

## Cells 5-8: Loading The Dataset

**What to say**

```text
In this step I load the mushroom dataset from a CSV file using a semicolon delimiter.
Then I check the shape of the dataset, display the first rows, inspect column types and look at summary statistics.
```

**Notebook details**

```text
61069 rows and 21 columns
```

**Plain English**

`shape` shows how many rows and columns there are.  
`head()` shows the first rows.  
`info()` shows data types and missing values.  
`describe()` shows basic summary statistics.

**Question**

```text
Why do you check the shape?
```

**Answer**

```text
I check the shape to know how many observations and features are available.
```

**Question**

```text
What does one row represent?
```

**Answer**

```text
One row represents one mushroom described by physical characteristics and the target class.
```

**Question**

```text
What is the target variable?
```

**Answer**

```text
The target variable is class. It tells whether a mushroom is edible or poisonous.
```

---

## Cells 9-12: Initial Overview

**What to say**

```text
Here I inspect data types, missing values and the target distribution.
The target has two classes: p for poisonous and e for edible.
The dataset is slightly imbalanced, with about 55.5% poisonous and 44.5% edible mushrooms.
```

**Concrete details**

```text
p = 33888
e = 27181
```

**Plain English**

This shows which columns are present, where values are missing, and whether the classes are roughly balanced.

**Question**

```text
Why check missing values?
```

**Answer**

```text
Because missing values affect preprocessing decisions. Some columns may need to be removed, while others can be imputed.
```

**Question**

```text
Why check target distribution?
```

**Answer**

```text
Because if the classes are imbalanced, accuracy may be misleading and stratified splitting becomes important.
```

**Question**

```text
Is the dataset balanced?
```

**Answer**

```text
It is slightly imbalanced. Poisonous mushrooms are about 55.5% and edible mushrooms are about 44.5%.
```

---

## Cells 13-21: Data Cleaning And Preparation

**What to say**

```text
In this section I first check duplicates, but I do not drop them before splitting.
Then I prepare X and y and split the data into train, validation and test sets using stratification.

After the split, I check the class distribution in all subsets.
Then I calculate missing-value ratios only on the training data and remove columns with more than 80% missing values from all splits.
```

**Plain English**

The most important rule is: split first, then preprocess.  
The imputer, scaler and column-removal decisions are not fitted on the whole dataset.

**Concrete details**

```text
Duplicate rows: 146

X_train: 36641 rows
X_val: 12214 rows
X_test: 12214 rows

Dropped columns:
veil-type
spore-print-color
veil-color
stem-root
```

**Why stratify**

Stratification keeps a similar `p/e` class distribution in train, validation and test.

**Question**

```text
Why do you split before preprocessing?
```

**Answer**

```text
To avoid data leakage. Validation and test data should not influence preprocessing decisions or fitted transformations.
```

**Question**

```text
What is data leakage?
```

**Answer**

```text
Data leakage happens when information from validation or test data is used during training or preprocessing.
```

**Question**

```text
Why did you not drop duplicates before splitting?
```

**Answer**

```text
I checked duplicates, but I avoided changing rows before the split. The key exam rule is to split before changing values or fitting preprocessing steps.
```

**Question**

```text
Why drop columns with more than 80% missing values?
```

**Answer**

```text
Because those columns contain too little observed information to be reliable, and imputing such columns would mostly create artificial values.
```

**Question**

```text
Why decide dropped columns only on training data?
```

**Answer**

```text
Because selecting columns using the full dataset would use information from validation and test data, which would be data leakage.
```

**Question**

```text
What are X and y?
```

**Answer**

```text
X is the feature matrix and y is the target variable.
```

---

## Cells 23-25: Preprocessing And Model

**What to say**

```text
I create separate preprocessing pipelines for numerical and categorical features.
Numerical features are imputed with the median and scaled.
Categorical features are imputed with the most frequent value and one-hot encoded.
Then I combine preprocessing with Logistic Regression in one pipeline.
```

**Plain English**

The pipeline ensures that preprocessing is fitted only on the training data during validation.

**Numerical features**

```text
cap-diameter
stem-height
stem-width
```

**Categorical features**

```text
cap-shape, cap-surface, cap-color, does-bruise-or-bleed,
gill-attachment, gill-spacing, gill-color, stem-surface,
stem-color, has-ring, ring-type, habitat, season
```

**Question**

```text
Why use median imputation for numerical features?
```

**Answer**

```text
Median imputation is more robust to outliers than mean imputation.
```

**Question**

```text
Why use most frequent imputation for categorical features?
```

**Answer**

```text
Because categorical values do not have a numerical median, so the most common category is a simple replacement.
```

**Question**

```text
What does OneHotEncoder do?
```

**Answer**

```text
It converts categorical variables into binary numerical columns so the model can use them.
```

**Question**

```text
Why use handle_unknown='ignore'?
```

**Answer**

```text
Because validation or test data may contain categories not seen during training. Ignoring them prevents errors.
```

**Question**

```text
What does StandardScaler do?
```

**Answer**

```text
It subtracts the mean and divides by the standard deviation for each numerical feature.
```

**Question**

```text
Why use a pipeline?
```

**Answer**

```text
A pipeline keeps preprocessing and modelling together, and helps prevent data leakage during validation.
```

---

## Cells 26-33: Single Validation Split

**What to say**

```text
The first validation method is a single validation split.
I train Logistic Regression with several C values and choose the one with the best F1-score for the poisonous class.
The best C is 0.1.
Then I fit the model using that C, print the classification report and inspect the confusion matrix.
```

**Concrete details**

```text
Best C: 0.1
Validation accuracy: about 0.814
Validation F1 for poisonous: about 0.831
Validation recall for poisonous: about 0.825
```

**Plain English**

This is a quick method: one train set and one validation set. It is simple to explain, but depends on one specific split.

**C**

```text
small C = stronger regularization
large C = weaker regularization
```

**Question**

```text
Why use Logistic Regression?
```

**Answer**

```text
Because the assignment requires Logistic Regression, and it is a simple classification model that can estimate class probabilities.
```

**Question**

```text
Why is Logistic Regression used for classification?
```

**Answer**

```text
Because it estimates probabilities for classes and then assigns observations based on a decision threshold.
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
Why choose F1-score for the poisonous class?
```

**Answer**

```text
Because I care about the poisonous class specifically, and F1-score balances precision and recall for that class.
```

**Question**

```text
What is the weakness of a single validation split?
```

**Answer**

```text
It depends on one specific split. If the split is unlucky, the result may not be representative.
```

---

## Metrics: Confusion Matrix, Precision, Recall, F1, ROC

**Classes**

```text
e = edible
p = poisonous
```

**Confusion matrix for poisonous as the positive class**

```text
                 Predicted edible     Predicted poisonous
Actual edible    TN                  FP
Actual poisonous FN                  TP
```

**TP**

```text
Actual: poisonous
Predicted: poisonous
```

**TN**

```text
Actual: edible
Predicted: edible
```

**FP**

```text
Actual: edible
Predicted: poisonous
```

False alarm. The model predicts "poisonous", but the mushroom is actually edible.

**FN**

```text
Actual: poisonous
Predicted: edible
```

This is the worst error in this assignment.

**Accuracy**

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**

```text
precision = TP / (TP + FP)
```

When the model says "poisonous", how often is it correct?

**Recall**

```text
recall = TP / (TP + FN)
```

How many truly poisonous mushrooms did the model detect?

**F1**

```text
F1 = 2 * precision * recall / (precision + recall)
```

Balance between precision and recall.

**ROC curve**

```text
ROC shows the trade-off between true positive rate and false positive rate across thresholds.
```

**Precision-recall curve**

```text
Precision-recall curve shows the trade-off between precision and recall across thresholds.
```

**Most important**

```text
Recall for poisonous is the key metric because false negatives are dangerous.
```

---

## Cells 34-38: Stratified K-Fold Cross-Validation

**What to say**

```text
The second validation method is Stratified K-Fold cross-validation.
I use GridSearchCV with F1-score for the poisonous class to tune C.
Stratification keeps the class distribution similar in each fold.
```

**Concrete details**

```text
Best C: 0.1
Mean F1: about 0.830
Standard deviation: about 0.002
```

**Plain English**

The model is trained and evaluated several times on different folds. The result is more stable than a single split.

**Question**

```text
What is K-Fold cross-validation?
```

**Answer**

```text
K-Fold cross-validation splits the data into K parts. The model is trained K times, each time using one fold for validation and the remaining folds for training.
```

**Question**

```text
Why use Stratified K-Fold?
```

**Answer**

```text
Because this is a classification problem, and stratification keeps class proportions similar in each fold.
```

**Question**

```text
What is GridSearchCV?
```

**Answer**

```text
GridSearchCV tests different hyperparameter values using cross-validation and selects the best one based on a scoring metric.
```

**Question**

```text
Why is K-Fold more stable than a single split?
```

**Answer**

```text
Because the model is evaluated on several different validation folds instead of only one validation set.
```

---

## Cells 39-41: Nested Cross-Validation

**What to say**

```text
The third validation method is nested cross-validation.
The inner loop tunes the hyperparameter, and the outer loop estimates performance.
This separates model selection from model evaluation.
```

**Concrete details**

```text
Nested CV mean F1: about 0.829
Nested CV std: about 0.0025
```

**Plain English**

Nested CV is methodologically cleaner, but slower.

**Question**

```text
Why use nested cross-validation?
```

**Answer**

```text
Because it gives a less biased estimate when hyperparameters are tuned. The outer loop evaluates data not used in the inner tuning.
```

**Question**

```text
What is the downside of nested cross-validation?
```

**Answer**

```text
It is computationally expensive because it runs many cross-validation procedures.
```

**Question**

```text
Why use 3 outer and 3 inner folds?
```

**Answer**

```text
To keep the notebook practical while still demonstrating nested cross-validation.
```

---

## Cells 42-49: Final Model And Test Evaluation

**What to say**

```text
After validation, I use the best model selected by GridSearchCV.
I fit it on the combined training and validation data and evaluate it once on the untouched test set.
Then I show the confusion matrix, ROC curve, precision-recall curve and final metrics.
```

**Final results**

```text
Accuracy: 0.8167
Precision for poisonous: 0.8422
Recall for poisonous: 0.8241
F1 for poisonous: 0.8330
ROC AUC: 0.8816
```

**Question**

```text
Why evaluate on the test set only at the end?
```

**Answer**

```text
Because the test set should represent unseen data and provide a final unbiased performance estimate.
```

**Question**

```text
Why fit the final model on train plus validation?
```

**Answer**

```text
Because after hyperparameters are selected, using more data can improve the final model.
```

**Question**

```text
Why check best_model.classes_?
```

**Answer**

```text
To make sure I use the probability column for the poisonous class when plotting ROC and precision-recall curves.
```

**Question**

```text
What does ROC AUC mean?
```

**Answer**

```text
ROC AUC measures how well the model separates the positive and negative classes across different thresholds.
```

---

## Cells 50-51: Interpretation And Limitations

**What to say**

```text
The final model performs reasonably well, with about 82% accuracy and 0.83 F1-score for the poisonous class.
However, it still produces false negatives, meaning some poisonous mushrooms are predicted as edible.
This is the main limitation because such mistakes would be dangerous in a real mushroom-foraging context.
```

**Plain English**

The model is acceptable for the assignment, but it is not safe for real-life use.

**Main limitation**

```text
Some poisonous mushrooms are still classified as edible.
```

**Question**

```text
Which metric is most important here?
```

**Answer**

```text
Recall for the poisonous class is the most important metric, because it measures how many dangerous mushrooms are detected.
```

**Question**

```text
Is accuracy enough here?
```

**Answer**

```text
No. Accuracy treats all mistakes equally, but false negatives are much more dangerous in this problem.
```

**Question**

```text
Would you use this model for real mushroom foraging?
```

**Answer**

```text
No. It is useful for learning and comparison, but it still makes dangerous false-negative errors.
```

**Question**

```text
How could the model be improved?
```

**Answer**

```text
I could tune the classification threshold, compare additional models, add domain knowledge, or optimize specifically for higher recall of the poisonous class.
```

**Final defense**

```text
The model is useful for learning and comparison, but because it still produces false negatives, it should not be treated as safe for real-world mushroom foraging.
```

# Assignment 3: Mushroom Foraging - ADHD-Friendly Oral Exam Lesson

## How To Use This File

To jest wersja do nauki ustnej: najpierw masz normalny kawałek lekcji, a dopiero pod spodem wyjaśnienia słów z tej sekcji.

Schemat:

```text
section
lesson text
term explanations
exam sentences
where in code
```

Main exam pattern:

```text
What does it mean?
Where is it in your code?
Why did you do it?
```

---

# 0. Big Picture

Assignment 3 is about mushroom foraging.

The goal is to classify mushrooms as edible or poisonous.

This is a supervised binary classification problem.

The notebook loads `secondary_data.csv`, prepares the data, trains Logistic Regression, and evaluates the model using validation methods, a confusion matrix and classification metrics.

The most important practical point is safety: predicting a poisonous mushroom as edible is the dangerous mistake.

## Terms From This Section

`supervised`

```text
The model learns from examples where the correct answer is already known.
Here, every mushroom has a known class: edible or poisonous.
```

`binary classification`

```text
Classification with two possible classes.
Here: edible or poisonous.
```

`edible`

```text
Safe to eat.
In the dataset this is class e.
```

`poisonous`

```text
Dangerous/toxic.
In the dataset this is class p.
```

`classification metrics`

```text
Numbers used to evaluate a classifier.
Examples: accuracy, precision, recall, F1-score.
```

## Exam Sentences

```text
This is a supervised binary classification problem.
The goal is to predict whether a mushroom is edible or poisonous from its features.
The most dangerous error is a false negative: a poisonous mushroom predicted as edible.
```

---

# 1. Loading And Initial Data Check

The notebook starts by loading the mushroom dataset from `secondary_data.csv`.

The dataset has:

```text
61069 rows
21 columns
```

One row represents one mushroom.

The notebook checks shape, first rows, column types, missing values and summary statistics.

This is basic exploratory data analysis before modelling.

## Terms From This Section

`row`

```text
One observation.
Here: one mushroom.
```

`column`

```text
One variable.
Here: a mushroom characteristic or the target class.
```

`shape`

```text
Number of rows and columns in the dataset.
```

`info()`

```text
Pandas method showing column types and missing values.
```

`describe()`

```text
Pandas method showing summary statistics.
```

`exploratory data analysis`

```text
Initial inspection of the dataset before modelling.
```

## Exam Sentences

```text
I first load the dataset and inspect its shape, data types, missing values and basic statistics.
This helps me understand the structure of the data before building a model.
```

## Where In Code

```text
Cells 5-8: loading secondary_data.csv, checking shape, info() and describe().
```

---

# 2. Target Distribution

The target variable is `class`.

It has two values:

```text
p = poisonous
e = edible
```

The dataset is slightly imbalanced:

```text
poisonous: about 55.5%
edible: about 44.5%
```

This matters because accuracy alone can be misleading, and stratified splitting is useful.

## Terms From This Section

`target variable`

```text
The thing the model should predict.
Here: class.
```

`class`

```text
The category assigned to an example.
Here: edible or poisonous.
```

`class distribution`

```text
How many examples belong to each class.
```

`imbalanced dataset`

```text
One class has more examples than another.
Here the imbalance is not extreme, but poisonous is more common.
```

`accuracy can be misleading`

```text
If classes are imbalanced, a model can get decent accuracy by mostly predicting the majority class.
```

## Exam Sentences

```text
The target variable is class, where p means poisonous and e means edible.
I check the target distribution because class imbalance affects splitting and evaluation.
The dataset is slightly imbalanced, so I use stratification and also look at precision, recall and F1-score.
```

## Where In Code

```text
Cells 9-12: checking data types, missing values and target distribution.
```

---

# 3. X, y And Splitting

After inspecting the data, the notebook separates `X` and `y`.

`X` contains mushroom features.

`y` contains the target:

```text
class
```

Then the data is split into training, validation and test sets.

The split is done before fitting preprocessing steps.

This is important because preprocessing should not learn from validation or test data.

## Terms From This Section

`X`

```text
The feature matrix.
Here: mushroom characteristics used as input.
```

`y`

```text
The target vector.
Here: edible/poisonous class.
```

`training set`

```text
Data used to fit the model.
```

`validation set`

```text
Data used for model selection and hyperparameter tuning.
```

`test set`

```text
Data used only once at the end for final evaluation.
```

`preprocessing`

```text
Data transformations before modelling.
Examples: imputation, scaling, one-hot encoding.
```

## Exam Sentences

```text
X contains the mushroom features, and y contains the class label.
I split the data into train, validation and test sets before fitting preprocessing.
The training set fits the model, the validation set helps choose hyperparameters, and the test set is used only for final evaluation.
```

## Where In Code

```text
Cells 13-21: duplicates, X/y creation, train-validation-test split and stratification.
```

---

# 4. Stratification

The notebook uses stratified splitting.

Stratification keeps the edible/poisonous proportions similar in train, validation and test sets.

This matters because the dataset is slightly imbalanced.

In this notebook, train, validation and test all keep approximately the same class proportions:

```text
about 55.5% poisonous
about 44.5% edible
```

## Terms From This Section

`stratification`

```text
Keeping class proportions similar across splits.
```

`stratify=y`

```text
Argument used in train_test_split to preserve target class proportions.
```

`representative split`

```text
A split that has a similar class distribution to the full dataset.
```

## Exam Sentences

```text
Stratification keeps the class proportions similar in train, validation and test sets.
I use it because this is a classification problem and the edible/poisonous classes are slightly imbalanced.
```

## Where In Code

```text
Cell 16: train_test_split with stratify.
Cell 17: checking class distribution in the splits.
```

---

# 5. Missing Values And Column Removal

The notebook checks missing values.

Some columns have too many missing values.

After splitting, columns with more than 80% missing values are selected using only the training data and removed from train, validation and test sets.

Dropped columns:

```text
veil-type
spore-print-color
veil-color
stem-root
```

This is done after the split to avoid data leakage.

## Terms From This Section

`missing value`

```text
A value that is absent or unknown.
```

`column removal`

```text
Dropping a feature from the dataset.
```

`80% missing`

```text
If more than 80% of a column is missing, it may contain too little useful information.
```

`training-data-only decision`

```text
A preprocessing decision made using only the training set.
This avoids leaking validation/test information.
```

`data leakage`

```text
When information from validation or test data influences training or preprocessing.
It makes evaluation too optimistic.
```

## Exam Sentences

```text
I remove columns with more than 80% missing values because they contain too little observed information to be reliable.
I decide which columns to remove using only the training data to avoid data leakage.
The same columns are then removed from validation and test sets.
```

## Where In Code

```text
Cells 18-20: missing-value ratios calculated on training data and high-missing columns removed from all splits.
```

---

# 6. Imputation

After removing columns with too many missing values, remaining missing values are handled with imputation.

Numerical features are imputed with the median.

Categorical features are imputed with the most frequent value.

This happens inside the preprocessing pipeline.

## Terms From This Section

`imputation`

```text
Filling in missing values.
```

`numerical feature`

```text
A feature with numbers.
Examples: cap-diameter, stem-height, stem-width.
```

`categorical feature`

```text
A feature with categories/labels.
Examples: cap-shape, cap-color, habitat.
```

`median imputation`

```text
Replacing missing numerical values with the median.
Median is robust to outliers.
```

`most frequent imputation`

```text
Replacing missing categorical values with the most common category.
```

## Exam Sentences

```text
For numerical features, I use median imputation because it is robust to outliers.
For categorical features, I use most-frequent imputation because categories do not have a numerical median.
```

## Where In Code

```text
Cells 23-25: numerical and categorical preprocessing pipelines.
```

---

# 7. One-Hot Encoding And Scaling

Most mushroom features are categorical.

Machine learning models need numerical input, so categorical variables are converted with one-hot encoding.

Numerical features are scaled with `StandardScaler`.

Both encoding and scaling are placed inside the pipeline, so they are fitted only on training data during validation.

## Terms From This Section

`one-hot encoding`

```text
Converts categories into binary 0/1 columns.
It avoids creating a fake order between category names.
```

`binary column`

```text
A column with values 0 or 1.
```

`false ordinal relationship`

```text
An incorrect order created by encoding categories as 1, 2, 3.
Example: red=1, blue=2, green=3 would suggest green > blue > red, which is nonsense.
```

`StandardScaler`

```text
Scales numerical features by subtracting mean and dividing by standard deviation.
```

`handle_unknown='ignore'`

```text
Prevents errors if validation/test data has a category not seen during training.
```

## Exam Sentences

```text
I use one-hot encoding because Logistic Regression needs numerical input, and mushroom features are mostly categorical.
One-hot encoding avoids creating a false order between categories.
I scale numerical features because scaling helps optimization for Logistic Regression.
```

## Where In Code

```text
Cells 23-25: OneHotEncoder for categorical features and StandardScaler for numerical features.
```

---

# 8. Pipeline

The notebook combines preprocessing and Logistic Regression in one pipeline.

This means imputation, scaling, one-hot encoding and modelling are treated as one workflow.

The main benefit is leakage-safe validation.

During validation, the pipeline fits preprocessing only on the training fold and then applies it to validation data.

## Terms From This Section

`pipeline`

```text
A sequence of preprocessing steps and a model connected together.
```

`fit on train, transform validation/test`

```text
The safe preprocessing rule.
Learn transformations from training data only.
```

`training fold`

```text
The part of data used for training inside cross-validation.
```

`validation fold`

```text
The part used for validation inside cross-validation.
```

## Exam Sentences

```text
I use a pipeline to keep preprocessing and modelling together.
This helps prevent data leakage because preprocessing is fitted only on training data during validation.
```

## Where In Code

```text
Cells 23-25: preprocessing pipeline combined with LogisticRegression.
```

---

# 9. Logistic Regression

Logistic Regression is the main model in this assignment.

Even though the name contains "regression", it is used for classification.

It takes a linear combination of features and passes it through a sigmoid function.

The sigmoid output is a probability between 0 and 1.

Then the model uses a threshold, usually 0.5, to choose the final class.

## Terms From This Section

`Logistic Regression`

```text
A classification algorithm that estimates class probabilities.
```

`sigmoid function`

```text
Function that maps any real number to a value between 0 and 1.
Useful for probability-like outputs.
```

`probability`

```text
A value between 0 and 1 representing model confidence.
```

`threshold`

```text
Cutoff used to convert probability into a class.
Usually 0.5.
```

`linear combination`

```text
Features multiplied by coefficients and added together.
```

## Exam Sentences

```text
Logistic Regression is used because this is a binary classification task.
It estimates class probabilities using a sigmoid function.
The final class is chosen using a decision threshold, usually 0.5.
```

## Where In Code

```text
Cells 23-25: base Logistic Regression pipeline.
Cells 26-33: Logistic Regression with different C values.
Cells 42-49: final Logistic Regression model and test evaluation.
```

---

# 10. C And Regularization

The notebook tunes the Logistic Regression hyperparameter `C`.

`C` controls regularization strength.

Important:

```text
small C = stronger regularization
large C = weaker regularization
```

The best `C` in the notebook is:

```text
C = 0.1
```

The model is selected using F1-score for the poisonous class.

## Terms From This Section

`hyperparameter`

```text
A setting chosen before or during model selection.
It is not learned directly like coefficients.
```

`C`

```text
Inverse regularization strength in Logistic Regression.
Smaller C means stronger regularization.
```

`regularization`

```text
A penalty that helps reduce overfitting.
```

`model selection`

```text
Choosing the best model or hyperparameter setting.
```

## Exam Sentences

```text
C controls regularization strength.
A smaller C means stronger regularization, while a larger C means weaker regularization.
I tune C because different regularization strengths can change model performance.
```

## Where In Code

```text
Cells 26-33: testing C values with a single validation split.
Cells 34-38: GridSearchCV tuning C with Stratified K-Fold cross-validation.
```

---

# 11. Single Validation Split

The first validation method is a single validation split.

The model is trained on the training set and evaluated on the validation set.

Several `C` values are tested.

The best value is chosen using F1-score for the poisonous class.

In the notebook:

```text
Best C: 0.1
Validation accuracy: about 0.814
Validation F1 for poisonous: about 0.831
Validation recall for poisonous: about 0.825
```

This method is simple, but it depends on one specific split.

## Terms From This Section

`validation split`

```text
One fixed split used to choose model settings.
```

`F1-score for poisonous`

```text
F1-score calculated specifically for the poisonous class.
Important because poisonous is the dangerous class.
```

`representative`

```text
Similar enough to the real data distribution.
An unlucky split may not be representative.
```

## Exam Sentences

```text
The single validation split is simple and fast, but it depends on one specific split.
I choose the best C using F1-score for the poisonous class because that class is safety-critical.
```

## Where In Code

```text
Cells 26-33: single validation split, C tuning, classification report and validation confusion matrix.
```

---

# 12. Stratified K-Fold Cross-Validation

The second validation method is Stratified K-Fold cross-validation.

The training data is split into several folds.

The model is trained and validated several times, each time with a different validation fold.

The notebook uses `GridSearchCV` to test different `C` values.

In the notebook:

```text
Best C: 0.1
Mean F1: about 0.830
Standard deviation: about 0.002
```

This is more stable than a single validation split, but more computationally expensive.

## Terms From This Section

`K-Fold cross-validation`

```text
Split data into K parts.
Train K times, each time validating on a different fold.
```

`StratifiedKFold`

```text
K-Fold cross-validation that preserves class proportions in each fold.
```

`GridSearchCV`

```text
Tool that tests hyperparameter values using cross-validation.
```

`mean F1`

```text
Average F1-score across folds.
```

`standard deviation`

```text
Shows how much scores vary across folds.
Small std means results are stable.
```

## Exam Sentences

```text
Cross-validation trains and validates the model several times on different folds.
This gives a more stable estimate than one validation split.
I use stratified folds to preserve the edible and poisonous class proportions.
```

## Where In Code

```text
Cells 34-38: StratifiedKFold and GridSearchCV.
```

---

# 13. Nested Cross-Validation

The third validation method is nested cross-validation.

Nested CV has two loops:

```text
inner loop -> hyperparameter tuning
outer loop -> performance estimation
```

The inner loop chooses the best `C`.

The outer loop evaluates performance on data not used for that tuning.

In the notebook:

```text
Nested CV mean F1: about 0.829
Nested CV std: about 0.0025
```

Nested CV is methodologically cleaner, but slower.

## Terms From This Section

`nested cross-validation`

```text
Cross-validation inside cross-validation.
Used to separate tuning from evaluation.
```

`inner loop`

```text
Used for hyperparameter tuning.
```

`outer loop`

```text
Used for estimating model performance.
```

`less biased estimate`

```text
A more honest performance estimate because evaluation data was not used for tuning.
```

`computationally expensive`

```text
Slow or costly because many models must be trained.
```

## Exam Sentences

```text
Nested cross-validation separates model selection from model evaluation.
The inner loop tunes the hyperparameter, and the outer loop estimates performance.
The downside is that it is computationally expensive.
```

## Where In Code

```text
Cells 39-41: nested cross-validation.
```

---

# 14. Final Model And Test Evaluation

After validation, the final model uses the best settings from `GridSearchCV`.

The final model is fitted on combined training and validation data.

Then it is evaluated once on the untouched test set.

Final test results:

```text
Accuracy: 0.8167
Precision for poisonous: 0.8422
Recall for poisonous: 0.8241
F1 for poisonous: 0.8330
ROC AUC: 0.8816
```

The test set is not used for tuning.

It is only used at the end to estimate final generalization performance.

## Terms From This Section

`final model`

```text
The model chosen after validation/model selection.
```

`train plus validation`

```text
After choosing hyperparameters, training on more data can improve the final model.
```

`untouched test set`

```text
Test data that was not used during training, preprocessing decisions or tuning.
```

`generalization performance`

```text
How well the model works on unseen data.
```

`ROC AUC`

```text
Measures how well the model separates the two classes across thresholds.
```

## Exam Sentences

```text
After choosing the best hyperparameter, I fit the final model on train plus validation data.
The test set is used only once at the end for final unbiased evaluation.
The final model gets about 82% accuracy and about 0.83 F1-score for the poisonous class.
```

## Where In Code

```text
Cells 42-49: final model, test confusion matrix, ROC curve, precision-recall curve and final metrics.
```

---

# 15. Confusion Matrix

The confusion matrix shows the types of classification errors.

For poisonous as the positive class:

```text
                 Predicted edible     Predicted poisonous
Actual edible    TN                  FP
Actual poisonous FN                  TP
```

The most important error is:

```text
False negative = actual poisonous, predicted edible
```

This is dangerous because the model says a poisonous mushroom is edible.

## Terms From This Section

`confusion matrix`

```text
A table showing correct and incorrect predictions by class.
```

`TP`

```text
True positive.
Actual poisonous, predicted poisonous.
```

`TN`

```text
True negative.
Actual edible, predicted edible.
```

`FP`

```text
False positive.
Actual edible, predicted poisonous.
False alarm.
```

`FN`

```text
False negative.
Actual poisonous, predicted edible.
This is the dangerous mistake.
```

## Exam Sentences

```text
The confusion matrix shows the types of errors, not only the total score.
In this problem, false negatives are the most dangerous because they mean poisonous mushrooms predicted as edible.
```

## Where In Code

```text
Cell 30: validation confusion matrix.
Cell 45: test confusion matrix.
```

---

# 16. Classification Metrics

Accuracy measures overall correctness.

Precision for poisonous asks: when the model predicts poisonous, how often is it correct?

Recall for poisonous asks: out of all actually poisonous mushrooms, how many did the model detect?

F1-score balances precision and recall.

For this task, recall for the poisonous class is especially important because false negatives are dangerous.

## Terms From This Section

`accuracy`

```text
Correct predictions divided by all predictions.
```

`precision`

```text
TP / (TP + FP)
When the model predicts poisonous, how often is it right?
```

`recall`

```text
TP / (TP + FN)
How many actual poisonous mushrooms did the model detect?
```

`F1-score`

```text
Balance between precision and recall.
```

`precision-recall trade-off`

```text
Changing the classification threshold can improve recall but reduce precision, or the other way around.
```

`threshold`

```text
The probability cutoff used to choose a class.
```

## Exam Sentences

```text
Accuracy is useful as an overall score, but it is not enough here because all mistakes are not equally dangerous.
Recall for the poisonous class is the key metric because it measures how many dangerous mushrooms are detected.
F1-score is also useful because it balances recall and precision.
```

## Where In Code

```text
Cells 29 and 32: validation classification report and recall.
Cells 47-49: final accuracy, precision, recall, F1 and ROC AUC.
```

---

# 17. ROC And Precision-Recall Curves

The notebook also plots ROC and precision-recall curves.

The ROC curve shows the trade-off between true positive rate and false positive rate across thresholds.

The precision-recall curve shows the trade-off between precision and recall across thresholds.

These curves are useful because classification performance changes when the decision threshold changes.

## Terms From This Section

`ROC curve`

```text
Shows true positive rate vs false positive rate across thresholds.
```

`true positive rate`

```text
Same idea as recall.
How many positives were found.
```

`false positive rate`

```text
How many negatives were incorrectly predicted as positive.
```

`precision-recall curve`

```text
Shows precision vs recall across thresholds.
```

`probability column`

```text
The model returns probabilities for each class.
For ROC/PR curves, we need the probability for the poisonous class.
```

## Exam Sentences

```text
ROC AUC measures how well the model separates the positive and negative classes across different thresholds.
The precision-recall curve is useful here because the poisonous class is safety-critical.
```

## Where In Code

```text
Cells 47-49: ROC curve, precision-recall curve and probability-based evaluation.
```

---

# 18. Limitations

The final model performs reasonably well, but it still produces false negatives.

That means some poisonous mushrooms are predicted as edible.

This is the main limitation because it would be dangerous in a real mushroom-foraging context.

Also, Logistic Regression is relatively simple.

It may not capture every complex relationship unless the encoded features make those patterns visible.

The model is useful for learning and comparison, but it should not be used as a real-world mushroom safety system.

## Terms From This Section

`limitation`

```text
A weakness or boundary of the solution.
```

`false negative risk`

```text
The risk that a dangerous mushroom is classified as edible.
```

`linear model`

```text
A model based on a linear combination of features.
Logistic Regression is linear before the sigmoid.
```

`real-world use`

```text
Using the model outside the assignment.
Here, that would be unsafe because mistakes can be dangerous.
```

## Exam Sentences

```text
The final model performs reasonably well, but it still predicts some poisonous mushrooms as edible.
These false negatives are the most important limitation.
The model is useful for learning and comparison, but it should not be treated as safe for real-world mushroom foraging.
```

## Where In Code

```text
Cells 50-51: interpretation and limitations.
```

---

# 19. Leave-One-Out Cross-Validation

Leave-One-Out cross-validation is not implemented in the notebook.

It is a theory topic only.

Leave-One-Out uses one observation as validation data and all remaining observations as training data.

This would require one model fit per observation.

Because this dataset has over 61,000 rows, Leave-One-Out would be computationally impractical.

Stratified K-Fold gives a better practical trade-off.

## Terms From This Section

`Leave-One-Out cross-validation`

```text
Validation method where one observation is used as validation and all others as training.
Repeated for every observation.
```

`one model fit per observation`

```text
With 61,069 rows, this would mean 61,069 model fits.
```

`practical trade-off`

```text
A balance between quality of estimate and computation time.
```

## Exam Sentences

```text
Leave-One-Out cross-validation is not used because it would require one model fit per observation.
For this dataset with over 61,000 rows, it is computationally impractical.
Stratified K-Fold is a better practical trade-off.
```

---

# 20. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Data loading:
Cells 5-8, where I load secondary_data.csv and inspect shape, info and describe.

Target distribution:
Cells 9-12, where I check class values p and e.

X and y:
Cells 13-21, where I prepare the feature matrix and target vector.

Train-validation-test split:
Cell 16, where I split data with stratification.

Missing-value column removal:
Cells 18-20, where I use only training data to select columns with more than 80% missing values.

Preprocessing pipeline:
Cells 23-25, where I define numerical and categorical preprocessing.

Logistic Regression:
Cells 23-25 for the base pipeline, then Cells 26-33 and 34-38 for tuning.

Single validation split:
Cells 26-33.

Stratified K-Fold and GridSearchCV:
Cells 34-38.

Nested cross-validation:
Cells 39-41.

Final test evaluation:
Cells 42-49.

Confusion matrix:
Cell 30 for validation and Cell 45 for test.

Limitations:
Cells 50-51.
```

---

# 21. A3 In 30 Seconds

```text
Assignment 3 is about classifying mushrooms as edible or poisonous.
This is supervised binary classification, where the target is class.

I first inspect the data, check missing values and target distribution, and then split the data into train, validation and test sets using stratification.

After the split, I remove columns with more than 80% missing values using only training data.
Then I use a pipeline with imputation, scaling, one-hot encoding and Logistic Regression.

I compare validation methods: a single validation split, Stratified K-Fold cross-validation and nested cross-validation.
The final model gets about 82% accuracy and about 0.83 F1-score for the poisonous class, but false negatives are dangerous, so the model is not safe for real-world mushroom foraging.
```

---

# 22. Emergency Speaking Pattern

If you forget a formal definition, use this pattern:

```text
[Term] means [simple meaning].
In my assignment, I used it for [specific thing].
The reason is [why].
```

Example:

```text
Stratification means keeping class proportions similar across splits.
In my assignment, I used it for train, validation and test sets.
The reason is that the edible and poisonous classes are slightly imbalanced.
```

---

# 23. Top Words To Memorize

```text
target = class
binary classification = two classes
edible = safe class, e
poisonous = dangerous class, p
imputation = filling missing values
one-hot encoding = categories to 0/1 columns
pipeline = preprocessing plus model together
Logistic Regression = probability-based classifier
sigmoid = maps values to 0-1
C = inverse regularization strength
stratification = same class proportions in splits
confusion matrix = table of mistake types
false negative = poisonous predicted edible
recall = how many poisonous mushrooms were detected
F1 = balance between precision and recall
```

Final survival sentence:

```text
The key point in this assignment is that I prepare the data in a leakage-safe way, train Logistic Regression as a binary classifier, and focus on false negatives because poisonous mushrooms predicted as edible are dangerous.
```

# Assignment 2: Candidates I - ADHD-Friendly Oral Exam Lesson

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

Assignment 2 uses Danish candidate test data from the 2022 election.

The assignment has two main parts:

```text
descriptive analysis and visualization
supervised multiclass classification
```

First, the notebook describes the candidate data and uses box plots.

Then it predicts party affiliation from candidate-test answer patterns.

The models are tree-based classifiers:

```text
Decision Tree
Random Forest
Gradient-Boosted Trees
```

The best model is Random Forest, with about 0.92 test accuracy and about 0.92 macro F1.

## Terms From This Section

`descriptive analysis`

```text
Summarizing and exploring the dataset before modelling.
Examples: shape, missing values, party counts, age summaries.
```

`visualization`

```text
Using plots to understand the data.
In A2, box plots are important.
```

`supervised classification`

```text
The model learns from examples with known labels.
Here, the known label is party affiliation.
```

`multiclass classification`

```text
Classification with more than two classes.
Here, there are many possible parties.
```

`tree-based classifier`

```text
A model based on decision trees.
Examples: Decision Tree, Random Forest, Gradient-Boosted Trees.
```

## Exam Sentences

```text
This assignment combines descriptive analysis and supervised multiclass classification.
The goal is to describe candidate data and then predict party affiliation from questionnaire answers.
I compare Decision Tree, Random Forest and Gradient-Boosted Trees, and Random Forest performs best.
```

---

# 1. Loading And Initial Overview

The notebook loads candidate response datasets and question metadata.

The main dataset is `all_df`.

It contains:

```text
867 candidates
53 columns
```

One row represents one candidate.

The notebook checks first rows, `info()`, `describe()`, data types, missing values and basic statistics.

This is done before modelling to understand the data structure.

## Terms From This Section

`all_df`

```text
The main dataset with all candidates and their answers.
```

`row`

```text
One observation.
Here: one candidate.
```

`column`

```text
One variable.
Here: metadata or a political answer.
```

`info()`

```text
Shows data types and non-null counts.
Useful for checking missing values and column types.
```

`describe()`

```text
Shows summary statistics.
Useful for checking ranges and unusual values.
```

## Exam Sentences

```text
I first load the candidate datasets and inspect the structure.
I use info() and describe() to check data types, missing values and summary statistics before modelling.
```

## Where In Code

```text
Cells 5-10: loading data, first overview, info() and describe().
```

---

# 2. Metadata And Response Features

The dataset contains metadata and response features.

Metadata includes information like:

```text
candidate name
party
district
age
```

Response features are the 49 political answer columns.

The model uses the response features to predict party.

Age is cleaned for descriptive analysis, but it is not used as a classification feature.

The notebook also checks that answer values are in the expected range:

```text
-2 to 2
```

## Terms From This Section

`metadata`

```text
Descriptive information about candidates.
Examples: name, party, district, age.
```

`response feature`

```text
One political answer column used as model input.
```

`feature_cols`

```text
The 49 response columns used as X for classification.
```

`response range`

```text
The expected answer scale from -2 to 2.
Negative and positive values represent disagreement/agreement.
```

`age value 0`

```text
Not realistic for election candidates, so it is replaced with missing for descriptive analysis.
```

## Exam Sentences

```text
I separate metadata columns from the 49 political response features.
The classification model uses only the answer columns as input.
Age is cleaned for descriptive analysis, but it is not used as a model feature.
```

## Where In Code

```text
Cells 11-14: clean age, define metadata columns and response features, check duplicates and response range.
```

---

# 3. Descriptive Analysis

Before modelling, the notebook performs descriptive analysis.

It counts candidates per party and summarizes age by party.

This shows that the party classes are imbalanced.

Some parties have many more candidates than others.

This matters for evaluation, because accuracy alone can hide poor performance on smaller parties.

That is why macro F1 is used later together with accuracy.

## Terms From This Section

`party distribution`

```text
How many candidates belong to each party.
```

`class imbalance`

```text
Some classes have more examples than others.
Here, some parties are much larger than others.
```

`accuracy`

```text
Overall percentage of correct predictions.
```

`macro F1`

```text
F1-score calculated separately for each class and then averaged equally.
Useful when classes are imbalanced.
```

## Exam Sentences

```text
Before modelling, I inspect party counts and age summaries.
The party counts show class imbalance, so I later use macro F1 together with accuracy.
Macro F1 gives each party equal weight.
```

## Where In Code

```text
Cells 15-18: party counts and age summary.
```

---

# 4. Box Plots

The exam checklist explicitly mentions box plots.

The notebook creates box plots for:

```text
age by party
confidence score by party
```

The confidence score is the proportion of strong answers.

Strong answers are:

```text
-2 or 2
```

These plots are descriptive.

The confidence score is not added as a model feature.

## Terms From This Section

`box plot`

```text
A plot showing median, quartiles, spread and possible outliers.
Good for comparing distributions between groups.
```

`median`

```text
The middle value.
```

`quartiles`

```text
Values that divide data into four parts.
```

`outlier`

```text
An unusually extreme value.
```

`confidence score`

```text
The proportion of strong answers, meaning answers equal to -2 or 2.
```

## Exam Sentences

```text
I use box plots to compare distributions across parties.
The age plot describes candidate demographics, while the confidence-score plot describes how often candidates give strong answers.
A box plot shows the median, quartiles, spread and possible outliers.
```

## Where In Code

```text
Cells 19-22: box plots for age and confidence score by party.
```

---

# 5. Classification Setup

After descriptive analysis, the notebook defines a supervised classification task.

The target is:

```text
party affiliation
```

The input features are:

```text
49 candidate-test answer columns
```

The independent candidate class is removed because it has only three observations.

Then party labels are encoded numerically and the data is split into train and test sets using stratification.

There is no separate validation set because cross-validation on the training data is used for model comparison.

## Terms From This Section

`target`

```text
The value the model should predict.
Here: party affiliation.
```

`X`

```text
The feature matrix.
Here: the 49 response columns.
```

`y`

```text
The target vector.
Here: party labels.
```

`LabelEncoder`

```text
Converts party names into numerical labels for modelling.
```

`stratified train-test split`

```text
A split that keeps party proportions similar in training and test data.
```

## Exam Sentences

```text
This is a multiclass classification problem.
The input features are candidate-test answers and the target is party affiliation.
I remove the independent candidate class because it has only three observations, which is too small for stable stratified classification.
```

## Where In Code

```text
Cells 23-26: classification setup, target definition, label encoding and stratified train-test split.
```

---

# 6. Cross-Validation Instead Of Separate Validation Set

The notebook uses a train-test split plus cross-validation on the training data.

Cross-validation is used for model comparison.

The test set is kept untouched until final evaluation.

This means the cross-validation folds play the role of validation data.

This is useful because the dataset is not very large and party classes are imbalanced.

## Terms From This Section

`cross-validation`

```text
The model is trained and validated several times on different folds.
This gives a more stable estimate than one split.
```

`fold`

```text
One part of the data used in cross-validation.
```

`StratifiedKFold`

```text
Cross-validation that preserves class proportions in each fold.
```

`untouched test set`

```text
Test data not used for model comparison or tuning.
Used only once at the end.
```

## Exam Sentences

```text
I do not use a separate validation set because cross-validation on the training data is used for model comparison.
The folds play the role of validation, while the test set is still untouched and used only once at the end.
```

## Where In Code

```text
Cells 30-31: shared Stratified K-Fold cross-validation setup.
Cells 43-45: model comparison using cross-validation.
```

---

# 7. Baseline Model

Before training real models, the notebook creates a baseline.

The baseline always predicts the most frequent party.

This gives a simple reference point.

In the notebook:

```text
Baseline accuracy: about 0.116
Baseline macro F1: about 0.015
```

Real models should clearly beat this baseline.

## Terms From This Section

`baseline`

```text
A simple reference model.
It shows how well we can do without learning useful patterns.
```

`DummyClassifier`

```text
Scikit-learn baseline model.
Here it predicts the most frequent party.
```

`most frequent class`

```text
The class with the most examples.
```

## Exam Sentences

```text
I use a baseline to check whether real models learn useful patterns.
The baseline always predicts the most frequent party.
If a model only performs close to the baseline, it is not useful.
```

## Where In Code

```text
Cells 27-29: baseline DummyClassifier.
```

---

# 8. Decision Tree

The first real classifier is a Decision Tree.

A decision tree predicts by following a sequence of feature-based splits.

Each split tries to make the child nodes more pure with respect to party labels.

The notebook limits the tree complexity using:

```text
max_depth = 6
min_samples_leaf = 5
```

This reduces overfitting.

The notebook also inspects feature importance and plots only the first levels of the tree.

In the notebook:

```text
Decision Tree CV accuracy: about 0.69
Decision Tree CV macro F1: about 0.68
```

## Terms From This Section

`Decision Tree`

```text
A model that makes predictions using a sequence of if-else splits.
```

`split`

```text
A rule that divides the data based on a feature value.
```

`node`

```text
A point in the tree where data is split.
```

`leaf`

```text
An endpoint of the tree that predicts a class.
```

`max_depth`

```text
Maximum number of split levels in the tree.
Limits complexity.
```

`min_samples_leaf`

```text
Minimum number of samples required in a leaf.
Prevents leaves from becoming too specific.
```

## Exam Sentences

```text
A decision tree predicts by following a sequence of feature-based splits.
I limit the tree depth and minimum leaf size to reduce overfitting.
The tree is interpretable because I can inspect the splits and feature importance.
```

## Where In Code

```text
Cells 32-35: Decision Tree model, cross-validation scores, feature importance and tree preview.
```

---

# 9. Gini Impurity And Overfitting In Trees

Decision trees choose splits that improve class purity.

Gini impurity measures how mixed the classes are in a node.

A pure node contains mostly or only one class.

A deep tree can overfit by memorizing training examples and noise.

Limiting `max_depth` and increasing `min_samples_leaf` makes the tree simpler.

## Terms From This Section

`Gini impurity`

```text
Measures how mixed the classes are in a node.
Lower impurity means purer groups.
```

`class purity`

```text
How much a node contains one class instead of many mixed classes.
```

`overfitting`

```text
The model performs well on training data but poorly on unseen data.
```

`generalization`

```text
How well the model works on new data.
```

## Exam Sentences

```text
Gini impurity measures how mixed the classes are in a node.
Decision trees prefer splits that reduce impurity.
A deep tree can overfit, so I limit depth and leaf size to help generalization.
```

---

# 10. Random Forest

The second model is Random Forest.

Random Forest is an ensemble of many decision trees.

Each tree is trained on a bootstrap sample and random subsets of features.

The final prediction is based on voting across trees.

In the notebook:

```text
n_estimators = 100
Random Forest CV accuracy: about 0.89
Random Forest CV macro F1: about 0.89
Final test accuracy: about 0.92
Final test macro F1: about 0.92
```

Random Forest performs best in this assignment.

## Terms From This Section

`Random Forest`

```text
An ensemble of many decision trees.
```

`ensemble`

```text
A model made from many smaller models.
```

`bootstrap sample`

```text
A random sample of training rows drawn with replacement.
Each tree sees a slightly different dataset.
```

`random feature subset`

```text
Each tree considers only some features at each split.
This makes trees more diverse.
```

`voting`

```text
Many trees vote for a class, and the majority class becomes the final prediction.
```

`variance`

```text
Sensitivity to training data changes.
A single tree has high variance; averaging many trees reduces it.
```

## Exam Sentences

```text
Random Forest combines many decision trees trained on different bootstrap samples and random feature subsets.
This reduces variance compared with one tree.
In this notebook, Random Forest performs best in cross-validation and on the final test set.
```

## Where In Code

```text
Cells 36-39: Random Forest model, cross-validation scores and feature importance.
Cells 46-49: final Random Forest test evaluation.
```

---

# 11. Gradient-Boosted Trees

The third model is Gradient-Boosted Trees.

Gradient boosting builds trees sequentially.

Each new tree tries to improve the errors made by the previous trees.

In the notebook:

```text
n_estimators = 80
learning_rate = 0.05
max_depth = 3
Gradient-Boosted Trees CV accuracy: about 0.84
Gradient-Boosted Trees CV macro F1: about 0.83
```

It performs better than a single Decision Tree, but worse than Random Forest in this setup.

## Terms From This Section

`Gradient Boosting`

```text
An ensemble method where trees are built one after another.
Each new tree focuses on improving previous errors.
```

`sequential`

```text
One after another, not independently.
```

`n_estimators`

```text
Number of trees in the ensemble.
```

`learning_rate`

```text
Controls how strongly each new tree changes the model.
Smaller learning rate means more gradual learning.
```

## Exam Sentences

```text
Gradient Boosting builds trees sequentially.
Each new tree tries to improve the errors made by the current ensemble.
It can be powerful, but it can also overfit if the model is too complex.
```

## Where In Code

```text
Cells 40-42: Gradient-Boosted Trees model and cross-validation scores.
```

---

# 12. Random Forest vs Gradient Boosting

Random Forest and Gradient Boosting are both tree ensembles.

But they work differently.

Random Forest builds many trees mostly independently and averages their votes.

Gradient Boosting builds trees sequentially, where each tree tries to correct the previous model.

In this notebook, Random Forest performs better.

## Terms From This Section

`independently`

```text
Trees are trained mostly separately, not one after another.
This is how Random Forest works.
```

`averaging`

```text
Combining many tree predictions to make the result more stable.
```

`correct previous errors`

```text
Boosting tries to improve where the current ensemble is weak.
```

## Exam Sentences

```text
Random Forest builds many trees mostly independently and averages them.
Gradient Boosting builds trees sequentially, where each tree tries to correct previous errors.
In this setup, Random Forest performed better.
```

---

# 13. Feature Importance

Tree-based models can report feature importance.

Feature importance estimates which questions helped the model make splits and reduce prediction error.

In this assignment, important features are candidate-test questions that help distinguish parties.

But feature importance is not causal proof.

It only tells what was useful for this model.

## Terms From This Section

`feature importance`

```text
An estimate of which features contributed most to model decisions.
```

`model-specific`

```text
The importance belongs to this model and method.
Another model could rank features differently.
```

`causal proof`

```text
Proof that one thing causes another.
Feature importance does not prove causality.
```

`informative question`

```text
A question whose answers help the model distinguish parties.
```

## Exam Sentences

```text
Feature importance shows which questionnaire answers helped the tree models distinguish parties.
I treat it as model-specific importance, not as causal political importance.
```

## Where In Code

```text
Cells 32-39: feature importance for Decision Tree and Random Forest.
```

---

# 14. Model Comparison And Final Test Evaluation

The notebook compares Decision Tree, Random Forest and Gradient-Boosted Trees using the same cross-validation setup.

The results are sorted by mean macro F1.

Random Forest is selected as the best model.

Then it is trained on the training data and evaluated once on the held-out test set.

Final result:

```text
Test accuracy: about 0.92
Test macro F1: about 0.92
```

The test set is used only at the end.

## Terms From This Section

`model comparison`

```text
Evaluating models with the same data splits and metrics.
```

`mean macro F1`

```text
Average macro F1 across cross-validation folds.
```

`held-out test set`

```text
Data kept separate until final evaluation.
```

`classification report`

```text
Table showing precision, recall and F1-score for each class.
```

`confusion matrix`

```text
Table showing which parties are predicted correctly or confused with each other.
```

## Exam Sentences

```text
I compare all models using the same cross-validation setup on the training data.
I sort results by macro F1 because party classes are imbalanced.
Random Forest is selected as the best model and evaluated once on the untouched test set.
```

## Where In Code

```text
Cells 43-45: cross-validation model comparison.
Cells 46-49: final model and test evaluation.
```

---

# 15. Classification Metrics

Accuracy measures the overall fraction of correct predictions.

Macro F1 gives every party equal weight.

This is important because party classes are imbalanced.

The classification report gives precision, recall and F1-score for each party.

The confusion matrix shows which parties are confused with which other parties.

## Terms From This Section

`accuracy`

```text
Correct predictions divided by all predictions.
```

`precision`

```text
When the model predicts a party, how often it is correct.
```

`recall`

```text
Out of all real members of a party, how many the model finds.
```

`F1-score`

```text
A balance between precision and recall.
```

`macro average`

```text
Average over classes with equal weight for each class.
```

## Exam Sentences

```text
Accuracy measures overall correctness.
Macro F1 is important because every party gets equal weight, which matters with imbalanced classes.
The confusion matrix is useful because it shows which parties are confused with each other.
```

## Where In Code

```text
Cells 27-29: baseline accuracy and macro F1.
Cells 32-45: cross-validation accuracy and macro F1.
Cells 46-49: final test accuracy, macro F1, classification report and confusion matrix.
```

---

# 16. Candidates Associated With Another Party

The notebook has an exploratory interpretation section.

It uses cross-validated Random Forest predictions to find candidates whose answer patterns are weakly associated with their actual party.

This does not prove they politically belong to another party.

It only shows a model-based signal.

## Terms From This Section

`cross_val_predict`

```text
Creates out-of-fold predictions.
Each row is predicted by a model that was not trained on that row.
```

`out-of-fold prediction`

```text
A prediction made for a validation fold during cross-validation.
```

`model-based signal`

```text
Something suggested by the model, not objective truth.
```

`weakly associated`

```text
The model is not confident that the answer pattern matches the actual party.
```

## Exam Sentences

```text
This section is exploratory.
I use cross-validated Random Forest predictions to find candidates whose answer patterns are weakly associated with their actual party.
This does not prove that they belong to another party; it is only a model-based signal.
```

## Where In Code

```text
Cells 50-52: candidates potentially associated with another party.
```

---

# 17. Limitations

The model predicts party affiliation from questionnaire answers quite well.

But the result should not be overinterpreted.

Main limitations:

```text
answers are self-reported
party is not perfect ground truth for ideology
classes are imbalanced
tree models can overfit
feature importance is model-specific, not causal
high accuracy may reflect party discipline and similar answer patterns
```

The model predicts party labels, not true political identity.

## Terms From This Section

`self-reported answers`

```text
Candidates gave the answers themselves.
They may not perfectly reflect later political behavior.
```

`ground truth`

```text
The label treated as correct.
Here, party is used as the label, but it is not perfect ideology.
```

`party discipline`

```text
Candidates from the same party may answer similarly because of party positions.
```

`overinterpret`

```text
Claim more than the model can actually prove.
```

## Exam Sentences

```text
The main limitation is that party affiliation is not the same as true political ideology.
Candidates from the same party can disagree, and candidates from different parties can answer similarly.
Feature importance is model-specific and should not be treated as causal proof.
```

## Where In Code

```text
Cells 53-54: conclusion and limitations.
```

---

# 18. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Data loading:
Cells 5-6, where I load candidate datasets and question metadata.

Initial overview:
Cells 7-10, where I use info() and describe().

Data cleaning and feature definition:
Cells 11-14, where I clean age, define metadata and response features, and check response range.

Descriptive analysis:
Cells 15-18, where I inspect party counts and age summaries.

Box plots:
Cells 19-22, where I plot age and confidence score by party.

Classification setup:
Cells 23-26, where I define X, y, remove the tiny independent class, encode labels and split data.

Baseline:
Cells 27-29.

Cross-validation setup:
Cells 30-31.

Decision Tree:
Cells 32-35.

Random Forest:
Cells 36-39.

Gradient-Boosted Trees:
Cells 40-42.

Model comparison:
Cells 43-45.

Final test evaluation:
Cells 46-49.

Exploratory candidate interpretation:
Cells 50-52.

Conclusion and limitations:
Cells 53-54.
```

---

# 19. A2 In 30 Seconds

```text
Assignment 2 uses Danish candidate test data from the 2022 election.
The first part is descriptive analysis: I inspect the data, check party distribution and use box plots for age and confidence score by party.

The second part is supervised multiclass classification.
The target is party affiliation and the input features are the 49 political answer columns.

I remove the independent candidate class because it has only three observations.
Then I use stratified train-test split and cross-validation on the training data.

I compare Decision Tree, Random Forest and Gradient-Boosted Trees using accuracy and macro F1.
Random Forest performs best, with about 0.92 test accuracy and macro F1.
The result should be interpreted as prediction of party labels from answer patterns, not as full truth about political ideology.
```

---

# 20. Emergency Speaking Pattern

If you forget a formal definition, use this pattern:

```text
[Term] means [simple meaning].
In my assignment, I used it for [specific thing].
The reason is [why].
```

Example:

```text
Macro F1 means F1 is calculated for each party and then averaged equally.
In my assignment, I use it because the party classes are imbalanced.
The reason is that smaller parties should not be ignored.
```

---

# 21. Top Words To Memorize

```text
descriptive analysis = describing the data before modelling
box plot = median, quartiles, spread, outliers
confidence score = proportion of strong answers
target = party affiliation
features = 49 answer columns
multiclass classification = more than two classes
baseline = simple reference model
Decision Tree = if-else split model
Gini impurity = class mixture in a node
Random Forest = many trees voting
bootstrap sample = sample with replacement
Gradient Boosting = trees built sequentially
macro F1 = equal-weight average over parties
feature importance = useful features for this model
confusion matrix = which parties are confused
```

Final survival sentence:

```text
The key point in this assignment is that I first describe the candidate data, then compare tree-based classifiers to predict party from answer patterns, while using macro F1 because the party classes are imbalanced.
```

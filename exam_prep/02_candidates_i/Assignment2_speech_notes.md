# Assignment 2: Candidate Test 2022 Analysis Part 1 - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with candidate test data from the Danish 2022 election.
The goal was to analyze candidate responses and build classification models that predict party affiliation from political answer patterns.

I started with data loading, initial inspection and simple cleaning.
Then I performed descriptive analysis and used box plots to visualize age and answer confidence across parties.

For the machine learning part, I used the 49 candidate-test answer columns as features and party affiliation as the target.
I removed the independent candidate class because it only had three observations, which is too small for stable stratified classification.

I used a stratified train-test split.
There is no separate validation set because I compare models using stratified cross-validation on the training data.
The test set is kept untouched and used only once for final evaluation.

I compared three tree-based classification algorithms: Decision Tree, Random Forest and Gradient-Boosted Trees.
The Random Forest performed best, with about 0.92 test accuracy and about 0.92 macro F1.
I still interpret the result carefully because party affiliation is not a perfect ground-truth label for political ideology.
```

## Cells 0-4: Title, Imports, Plot Style And Party Colors

```text
At the beginning I import the libraries needed for data handling, visualization, model training, cross-validation and evaluation.
I also define party colors so that party-based plots are consistent and easier to read.
```

## Cells 5-10: Load Data And Initial Overview

```text
I load the candidate response datasets and question metadata from the local data folder.
The main dataset is all_df, with 867 candidates and 53 columns.
Then I inspect data types, missing values and summary statistics to check that the data loaded correctly.
```

## Cells 11-14: Data Cleaning And Feature Definition

```text
I replace age values equal to 0 with missing values because age 0 is not realistic for election candidates.
Then I separate metadata columns from response columns.
The 49 response columns are the features used later for classification.

I also check duplicates and confirm that response values are on the expected scale from -2 to 2.
```

## Cells 15-18: Descriptive Analysis

```text
I count candidates per party and summarize age by party.
This shows that the party classes are imbalanced, meaning some parties have more candidates than others.

This matters for model evaluation because accuracy alone can be misleading with imbalanced classes.
That is why I later also use macro F1.
```

## Cells 19-22: Visualization With Box Plots

```text
This section covers the box plot requirement from the exam overview.
I use one box plot for age distribution by party and one for confidence score by party.

The confidence score is the proportion of strong answers, meaning answers equal to -2 or 2.
It is descriptive only and is not added as a model feature.
```

## Cells 23-26: Classification Problem Setup

```text
Here I define the supervised classification problem.
The target is party affiliation and the features are only the 49 political response columns.

I remove Løsgænger because it has only three observations.
Then I encode the party labels and create a stratified train-test split.

I do not create a separate validation set because cross-validation on the training data is used for model comparison.
The test set stays untouched until the final evaluation.
```

## Cells 27-29: Baseline Model

```text
Before training real models, I create a baseline model that always predicts the most frequent party.
This gives a simple reference point.

The baseline accuracy is about 0.116 and the macro F1 is about 0.015, so the real models should clearly beat it.
```

## Cells 30-31: Cross-Validation Setup

```text
I define Stratified K-Fold cross-validation with three folds.
Stratification is important because the party classes are imbalanced.

The metrics are accuracy and macro F1.
Macro F1 is important because it gives each party equal weight.
```

## Cells 32-35: 1. Decision Tree

```text
The first model is a Decision Tree.
It classifies candidates by following a sequence of question-based splits.

I limit the tree depth and require a minimum number of samples per leaf to reduce overfitting.
The Decision Tree is interpretable, but it performs worse than the ensemble methods.

I plot only the first two levels of the tree because this is a multiclass problem with many parties.
The full tree would be difficult to read.
```

## Cells 36-39: 2. Random Forest

```text
The second model is a Random Forest.
It trains many decision trees on different bootstrap samples and random feature subsets.

This usually reduces overfitting compared with one tree.
In this notebook, Random Forest performs best in cross-validation.

I also inspect feature importance, but I treat it as model-specific importance, not causal political importance.
```

## Cells 40-42: 3. Gradient-Boosted Trees

```text
The third model is Gradient-Boosted Trees.
Boosting builds trees sequentially, where each new tree tries to improve the previous model's errors.

This can be powerful, but it can also overfit if the trees are too deep or if too many trees are used.
In this setup, it performs better than a single Decision Tree but worse than Random Forest.
```

## Cells 43-45: Cross-Validation Model Comparison

```text
I compare the three models using the same cross-validation setup on the training data.
The results are sorted by mean macro F1.

This model comparison is done before touching the test set.
Random Forest is selected as the best model.
```

## Cells 46-49: Final Model And Test Evaluation

```text
I select the best model from cross-validation, fit it on the training data and evaluate it once on the held-out test set.

The final Random Forest achieves about 0.92 accuracy and about 0.92 macro F1.
The classification report shows precision, recall and F1-score for each party.
The confusion matrix is larger than in binary assignments because this is a multiclass problem with many parties.
```

## Cells 50-52: Candidates Potentially Associated With Another Party

```text
This is an exploratory interpretation section.
I use cross-validated Random Forest predictions to find candidates whose answer patterns are weakly associated with their actual party by the model.

This does not prove that they politically belong to another party.
It is only a model-based signal.
```

## Cells 53-54: Conclusion And Limitations

```text
The notebook shows that party affiliation can be predicted quite well from candidate-test answers.
However, this does not mean the model fully captures political ideology.

The main limitations are self-reported answers, class imbalance, party labels not being perfect ideology labels, possible overfitting and model-specific feature importance.
```

## Final Sentence

```text
The Random Forest is the strongest model in this notebook, but I interpret the result as prediction of party labels from answer patterns, not as a complete explanation of political ideology.
```


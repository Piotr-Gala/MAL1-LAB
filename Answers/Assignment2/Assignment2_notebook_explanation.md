# Assignment 2: Candidate Test 2022 Analysis Part 1 - Oral Exam Notes

## Opening Speech

```text
Good morning. In this assignment I worked with candidate test data from the Danish 2022 election.
The goal was to analyze candidate answers and build classification models that predict party affiliation from the answer patterns.

The dataset contains candidate responses from DR and TV2 candidate tests.
The answer features are numerical values from -2 to 2, where negative and positive values represent disagreement and agreement.

I started with basic data inspection and descriptive analysis.
I checked the dataset shape, missing values, party distribution, age distribution and the range of response values.
Then I used box plots to visualize differences between parties, especially age and the proportion of strong answers.

For the machine learning part, I defined a classification problem: predict the party from the 49 political response features.
I removed the independent candidate class because it only contains three observations, which is too small for stable stratified classification.
Then I used a stratified train-test split and kept the test set untouched until the final evaluation.

I compared three tree-based classification algorithms: a Decision Tree, a Random Forest and Gradient-Boosted Trees.
The models were compared using cross-validation on the training data with accuracy and macro F1.
Finally, I selected the best model from cross-validation and evaluated it once on the held-out test set.

The Random Forest performed best, with about 0.92 test accuracy and about 0.92 macro F1.
I still interpret the result carefully because party affiliation is not a perfect ground-truth label for political ideology.
```

---

## Cells 0-4: Title, Imports, Plot Style And Party Colors

**What to say**

```text
At the beginning I define the topic and import the libraries needed for data analysis, visualization and tree-based classification.
Pandas and NumPy are used for tabular data handling, Matplotlib and Seaborn for plots, and Scikit-learn for splitting, modelling, cross-validation and evaluation.
I also define party colors so the party-based plots are visually consistent and easier to read.
```

**Plain English**

This is only setup. No data has been changed or modelled yet.

**Most important**

- `pandas` - loading and working with tables.
- `numpy` - numerical operations.
- `matplotlib`, `seaborn` - visualizations.
- `train_test_split` - creates the train and test sets.
- `StratifiedKFold` - cross-validation that preserves class proportions.
- `cross_validate` - compares models across folds.
- `cross_val_predict` - creates out-of-fold predictions for exploratory interpretation.
- `DummyClassifier` - baseline model.
- `DecisionTreeClassifier` - single tree model.
- `RandomForestClassifier` - ensemble of many trees.
- `GradientBoostingClassifier` - sequential boosted tree model.
- `classification_report`, `ConfusionMatrixDisplay` - evaluation.

**Question**

```text
Why do you use pandas?
```

**Answer**

```text
I use pandas because the data is tabular, so a DataFrame is convenient for loading, inspecting and selecting columns.
```

**Question**

```text
Why do you define party colors?
```

**Answer**

```text
The colors do not affect the model. They only make party-based visualizations easier to read and consistent across plots.
```

**Question**

```text
Why do you import three different tree-based models?
```

**Answer**

```text
Because the assignment focuses on classification algorithms: decision trees, random forests and gradient-boosted trees.
I compare them instead of relying on one model.
```

---

## Cells 5-6: Load Data

**What to say**

```text
Here I load the candidate response datasets and question metadata from the local data folder.
The main dataset is all_df, which contains all candidates and their answers.
The other files are loaded because they belong to the assignment data, although the classification model mainly uses all_df.
```

**Notebook details**

```text
all_df: 867 rows and 53 columns
DR candidates: 904 rows and 27 columns
TV2 candidates: 962 rows and 28 columns
Elected candidates: 169 rows and 53 columns
```

**Plain English**

`DATA_DIR = Path("data")` makes the file paths explicit, so the notebook can run from the Assignment2 folder.

**Question**

```text
Why do you use Path("data")?
```

**Answer**

```text
I use it to make the data location explicit. The Excel files are stored in the local data folder, so the notebook is less dependent on the current working directory.
```

**Question**

```text
What does one row represent?
```

**Answer**

```text
One row represents one candidate with metadata such as party and age, plus answers to political candidate-test questions.
```

---

## Cells 7-10: Initial Overview

**What to say**

```text
After loading the data, I inspect the DataFrame with info() and describe().
This tells me the number of rows and columns, data types, missing values and basic summary statistics.
```

**Plain English**

This step checks whether the data looks reasonable before cleaning, visualization or modelling.

**Most important**

- the main dataset has 867 candidates,
- it has 53 columns,
- most response columns are integer values,
- metadata columns include candidate name, party, district and age,
- the response columns are the future model features.

**Question**

```text
Why do you check info()?
```

**Answer**

```text
info() shows data types and non-null counts, so it helps me identify missing values and confirm that response features are numerical.
```

**Question**

```text
Why do you use describe()?
```

**Answer**

```text
describe() gives summary statistics and helps detect unusual values before modelling.
```

---

## Cells 11-14: Data Cleaning And Feature Definition

**What to say**

```text
In this section I replace age values equal to 0 with missing values, because age 0 is not realistic for election candidates.
Then I separate metadata columns from response columns.
The response columns are the 49 political answer features used later for classification.
I also check duplicates, missing values and the response range.
```

**Plain English**

Candidate name, party, district and age describe the candidate. The answer columns are the actual input variables for the model.

**Most important**

- `meta_cols` are not model features.
- `feature_cols` are the 49 response columns.
- age is cleaned for honest descriptive analysis.
- age is not used as a classification feature.
- response values are checked to be from `-2` to `2`.
- duplicate rows are checked before modelling.

**Question**

```text
Why do you not use age as a model feature?
```

**Answer**

```text
The classification task is to predict party from political answers.
Using age would change the question and could make the model depend on demographic information instead of answer patterns.
```

**Question**

```text
Why do you check the response range?
```

**Answer**

```text
Because the assignment states that answers should be on a scale from -2 to 2.
Checking the range confirms that the response features match the expected format.
```

**Question**

```text
Is replacing age 0 a leakage problem?
```

**Answer**

```text
No. Age is not used as a model feature here.
This cleaning is only for descriptive analysis and visualization.
```

---

## Cells 15-18: Descriptive Analysis

**What to say**

```text
Here I perform descriptive analysis.
I count candidates per party and summarize age by party.
This shows that the dataset is imbalanced: some parties have many more candidates than others.
```

**Plain English**

Descriptive analysis explains what the data looks like before building models.

**Most important**

- party counts show class imbalance,
- class imbalance matters for classification,
- age summary is descriptive, not part of the model,
- imbalance is why macro F1 is useful later.

**Question**

```text
Why is party imbalance important?
```

**Answer**

```text
Because a model can get high accuracy by performing well on large parties while performing badly on smaller parties.
That is why I also use macro F1, which gives every class equal weight.
```

**Question**

```text
What is descriptive analysis?
```

**Answer**

```text
It is analysis that summarizes and describes the dataset before modelling, for example counts, distributions and basic statistics.
```

---

## Cells 19-22: Visualization With Box Plots

**What to say**

```text
This section covers the visualization part of the assignment.
I use box plots to compare age distributions and confidence-score distributions across parties.
The confidence score is the proportion of strong answers, meaning answers equal to -2 or 2.
```

**Plain English**

A box plot shows the median, spread and possible outliers of a numerical variable for each party.

**Most important**

- age box plot visualizes demographic spread by party,
- confidence-score box plot visualizes answer behavior by party,
- confidence score is descriptive only,
- it is not added as a model feature,
- party colors make the plots easier to read.

**Question**

```text
What does a box plot show?
```

**Answer**

```text
A box plot shows the median, quartiles, spread and possible outliers of a numerical variable.
```

**Question**

```text
What is the confidence score?
```

**Answer**

```text
It is the proportion of questions where a candidate gives a strong answer, either -2 or 2.
It is a simple descriptive measure of how often a candidate takes clear positions.
```

**Question**

```text
Why not use confidence score as a model feature?
```

**Answer**

```text
Because it is derived from the response features and would summarize information that is already present in X.
I use it for interpretation, not to add another engineered feature.
```

---

## Cells 23-26: Classification Problem Setup

**What to say**

```text
Here I switch from descriptive analysis to supervised classification.
The target variable is party affiliation and the input features are the 49 candidate-test answers.
I remove the independent candidate class because it has only three observations, which is too small for stable stratified splitting and classification.
Then I encode the party labels and create a stratified train-test split.
```

**Plain English**

The model learns patterns in political answers and tries to predict the candidate's party.

**Most important**

- `X` contains only response features,
- `y` is party affiliation,
- `LabelEncoder` converts party names to numerical labels,
- `stratify=y_encoded` preserves party proportions,
- the test set is kept untouched until final evaluation,
- there is no separate validation set because cross-validation on the training data is used for model comparison.

**Question**

```text
Why do you remove Løsgænger?
```

**Answer**

```text
Because it has only three candidates.
With such a tiny class, stratified splitting and model evaluation are unstable, so removing it makes the classification task more defensible.
```

**Question**

```text
Why do you use stratification?
```

**Answer**

```text
I use stratification because the party classes are imbalanced.
It helps preserve similar party proportions in the train and test sets.
```

**Question**

```text
Why is the test set kept untouched?
```

**Answer**

```text
Because the test set should simulate new unseen data.
If I use it for model selection, the final evaluation becomes too optimistic.
```

**Question**

```text
Why do you have only train and test sets, but no separate validation set?
```

**Answer**

```text
Because I use cross-validation on the training data for model comparison.
The cross-validation folds play the role of validation: each model is trained on part of the training data and validated on another part.
The test set is still kept untouched and is used only once at the end.
```

**Question**

```text
Would a train-validation-test split also be acceptable?
```

**Answer**

```text
Yes, that would also be acceptable.
Here I chose train-test plus cross-validation because the dataset is not very large and cross-validation uses the training data more efficiently than one fixed validation split.
```

---

## Cells 27-29: Baseline Model

**What to say**

```text
Before training real models, I create a baseline model.
The baseline always predicts the most frequent party.
This gives a minimum reference point that real classifiers should beat.
```

**Plain English**

The baseline asks: how well can we do without learning any useful patterns?

**Notebook details**

```text
Baseline accuracy is about 0.116.
Baseline macro F1 is about 0.015.
```

**Question**

```text
Why do you need a baseline?
```

**Answer**

```text
A baseline gives context.
If a model only performs close to the baseline, then it is not learning useful patterns from the response features.
```

**Question**

```text
Why is the baseline macro F1 so low?
```

**Answer**

```text
Because the baseline predicts only one party.
It ignores all other parties, so its average per-class performance is very poor.
```

---

## Cells 30-31: Classification Algorithms And Cross-Validation Setup

**What to say**

```text
In this section I define the shared cross-validation setup.
I use Stratified K-Fold because the party classes are imbalanced.
I also define the metrics used later: accuracy and macro F1.
```

**Plain English**

This cell only prepares how the models will be evaluated. No model is trained here yet.

**Most important**

- 3-fold stratified CV preserves party proportions in each fold.
- accuracy and macro F1 are used.
- using the same CV and metrics later makes the model comparison fair.

**Question**

```text
Why use cross-validation?
```

**Answer**

```text
Cross-validation gives a more stable estimate than one train-validation split because the model is evaluated on several different folds.
It is used here instead of a separate validation set.
```

**Question**

```text
Why cross-validation only on the training data?
```

**Answer**

```text
Because the test set must remain untouched until the final evaluation.
Cross-validation is used for model comparison, not for final reporting on unseen data.
```

---

## Cells 32-35: 1. Decision Tree

**What to say**

```text
The first classification algorithm is a Decision Tree.
I define the Decision Tree model directly in this section.
It predicts by splitting the data based on response features.
I limit the depth and require a minimum number of samples per leaf to reduce overfitting.
Then I inspect feature importance and plot only the first levels of the tree, because the full multiclass tree would be difficult to read.
```

**Plain English**

A decision tree is like a sequence of if-else rules based on candidate answers.

The tree plot is only a preview. In Titanic there are two classes, so the tree is readable. Here there are many party classes, so a full tree contains long class-count vectors in each node.

**Notebook details**

```text
Decision Tree CV accuracy is about 0.69.
Decision Tree CV macro F1 is about 0.68.
```

**Question**

```text
How does a decision tree classify candidates?
```

**Answer**

```text
It asks a sequence of feature-based questions.
At each split, it separates candidates into groups that are more pure with respect to party labels.
```

**Question**

```text
Why limit max_depth?
```

**Answer**

```text
Without a depth limit, the tree can become too complex and overfit the training data.
Limiting depth makes it simpler and more likely to generalize.
```

**Question**

```text
Why is a decision tree interpretable?
```

**Answer**

```text
Because we can inspect the splits and see which features are used to make predictions.
```

**Question**

```text
Why do you only plot the first levels of the tree?
```

**Answer**

```text
Because this is a multiclass problem with many parties.
The full tree would contain long class-count vectors in every node and would be hard to read.
The preview is enough to show how tree splits work, while feature importance is easier to use for interpretation.
```

---

## Cells 36-39: 2. Random Forest

**What to say**

```text
The second model is a Random Forest.
I define the Random Forest model directly in this section.
It trains many decision trees on different bootstrap samples and random subsets of features.
The final prediction is based on the votes of many trees.
This usually improves generalization compared with a single decision tree.
```

**Plain English**

Instead of trusting one tree, the Random Forest combines many trees.

**Notebook details**

```text
Random Forest CV accuracy is about 0.89.
Random Forest CV macro F1 is about 0.89.
```

**Most important**

- `n_estimators=100` means 100 trees,
- `min_samples_leaf=2` makes trees slightly less extreme,
- `n_jobs=1` makes execution predictable,
- feature importance is inspected,
- feature importance is not causal importance.

**Question**

```text
Why does Random Forest often outperform a single tree?
```

**Answer**

```text
Because a single tree has high variance and can overfit.
Random Forest averages many different trees, which reduces variance and usually improves generalization.
```

**Question**

```text
What is bootstrap sampling?
```

**Answer**

```text
Bootstrap sampling means sampling training rows with replacement.
Each tree sees a slightly different version of the training data.
```

**Question**

```text
Can feature importance be interpreted causally?
```

**Answer**

```text
No. It only tells which features helped this model reduce prediction error.
It does not prove that a question causes party affiliation.
```

---

## Cells 40-42: 3. Gradient-Boosted Trees

**What to say**

```text
The third model is Gradient-Boosted Trees.
I define the Gradient-Boosted Trees model directly in this section.
Unlike Random Forest, boosting builds trees sequentially.
Each new tree tries to improve the errors made by the previous trees.
```

**Plain English**

Boosting is an ensemble method where trees are added one after another, and each tree focuses on improving the current model.

**Notebook details**

```text
Gradient-Boosted Trees CV accuracy is about 0.84.
Gradient-Boosted Trees CV macro F1 is about 0.83.
```

**Question**

```text
What is the difference between Random Forest and Gradient Boosting?
```

**Answer**

```text
Random Forest builds many trees mostly independently and averages them.
Gradient Boosting builds trees sequentially, where each tree tries to correct previous errors.
```

**Question**

```text
Why can boosting overfit?
```

**Answer**

```text
Because if the model uses too many trees or trees that are too deep, it can keep fitting small details and noise in the training data.
```

**Question**

```text
Why use a learning rate?
```

**Answer**

```text
The learning rate controls how strongly each new tree changes the model.
A smaller learning rate makes learning more gradual.
```

---

## Cells 43-45: Cross-Validation Model Comparison

**What to say**

```text
Here I compare all three models using the same cross-validation setup on the training data.
The results are sorted by mean macro F1, because macro F1 handles class imbalance better than accuracy alone.
```

**Plain English**

This section chooses the best model without touching the final test set.

**Notebook details**

```text
Random Forest performs best in cross-validation.
Decision Tree is the weakest model.
Gradient-Boosted Trees are better than a single tree but worse than Random Forest in this setup.
```

**Question**

```text
Why sort by macro F1?
```

**Answer**

```text
Because the party classes are imbalanced.
Macro F1 gives each party equal weight, so smaller parties are not ignored.
```

**Question**

```text
What does the standard deviation show?
```

**Answer**

```text
It shows how much the score changes across folds.
A smaller standard deviation means the model performance is more stable across different splits.
```

---

## Cells 46-49: Final Model And Test Evaluation

**What to say**

```text
After model comparison, I select the best model from cross-validation.
Then I fit it on the training data and evaluate it once on the held-out test set.
The final evaluation includes accuracy, macro F1, a classification report and a confusion matrix.
```

**Plain English**

The test set is used only here, at the end, to estimate how the selected model performs on unseen data.

**Notebook details**

```text
Best model from CV: Random Forest
Test accuracy: about 0.92
Test macro F1: about 0.92
```

**Most important**

- final model selection is based on training CV,
- test set is used once,
- classification report gives precision, recall and F1 for each party,
- confusion matrix shows which parties are confused with each other.

**Question**

```text
What is accuracy?
```

**Answer**

```text
Accuracy is the proportion of all predictions that are correct.
```

**Question**

```text
What is macro F1?
```

**Answer**

```text
Macro F1 calculates F1 for each class separately and then averages them equally.
It is useful when classes are imbalanced.
```

**Question**

```text
Why use a confusion matrix?
```

**Answer**

```text
A confusion matrix shows not only how many predictions are wrong, but which parties are confused with which other parties.
```

---

## Cells 50-52: Candidates Potentially Associated With Another Party

**What to say**

```text
This section is exploratory.
I use cross-validated Random Forest predictions to find candidates whose answer pattern the model associates weakly with their actual party.
This is not treated as final test evaluation and not as political truth.
```

**Plain English**

The table shows candidates where the model is least confident in the true party label.

**Most important**

- this is model-based interpretation,
- cross-validation avoids predicting candidates with a model trained directly on the same row,
- it does not prove that a candidate belongs to another party,
- it only shows answer patterns that are unusual for the actual party.

**Question**

```text
Why use cross_val_predict here?
```

**Answer**

```text
Because it creates out-of-fold predictions.
Each prediction is made by a model that was not trained on that specific row, which is more honest for exploratory analysis.
```

**Question**

```text
Can we say these candidates are truly closer to another party?
```

**Answer**

```text
No. We can only say that the model associates their answer pattern more weakly with their actual party.
This is a signal for interpretation, not ground truth.
```

---

## Cells 53-54: Conclusion And Limitations

**What to say**

```text
In the conclusion I summarize the full workflow.
The notebook starts with descriptive analysis and box plots, then moves to tree-based classification.
The Random Forest performs best in this setup, but the result must be interpreted with limitations.
```

**Plain English**

The model predicts party labels from answers well, but that does not mean it fully understands political ideology.

**Most important limitations**

- responses are self-reported,
- party is not a perfect ground-truth label for ideology,
- classes are imbalanced,
- tree models can overfit,
- feature importance is model-specific and not causal,
- high accuracy may partly reflect party discipline and similar answer patterns within parties.

**Question**

```text
What is the main limitation of this classification task?
```

**Answer**

```text
The target label is party affiliation, but party affiliation is not the same as true political ideology.
Candidates from the same party can disagree, and candidates from different parties can answer similarly.
```

**Question**

```text
Why is high accuracy not enough?
```

**Answer**

```text
Because the classes are imbalanced.
A model can perform well overall while still performing worse for smaller parties, so macro F1 and the classification report are also needed.
```

**Question**

```text
What would you improve if you had more time?
```

**Answer**

```text
I would tune hyperparameters more systematically and maybe compare more validation strategies.
But I would still keep the test set untouched until the final evaluation.
```

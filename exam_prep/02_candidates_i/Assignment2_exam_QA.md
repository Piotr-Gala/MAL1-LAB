# Assignment 2: Candidates I - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment2.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This assignment combines exploratory data analysis and supervised classification. The goal is to describe candidate data and then predict a candidate's party from questionnaire answers.

**How used in the assignment:** The notebook loads candidate test datasets, performs descriptive analysis, visualizes distributions with box plots, then trains tree-based classifiers: decision tree, random forest, and gradient-boosted trees.

## 1. Descriptive Analysis

**Question:** What is descriptive analysis?

**Theory answer:** Descriptive analysis summarizes and explores the dataset before modelling. It includes checking shape, data types, missing values, duplicates, value ranges, class counts, and basic statistics.

**How used in the assignment:** The notebook uses `info()`, `describe()`, missing-value checks, duplicate checks, party counts, age summaries, and response range checks before training models.

## 2. Box Plot

**Question:** What does a box plot show?

**Theory answer:** A box plot shows the median, quartiles, spread, and possible outliers of a numerical variable. It is useful for comparing distributions between groups.

**How used in the assignment:** The notebook creates box plots for age and confidence score by party. This helps compare candidate demographics and answer strength across parties.

## 3. Supervised Classification

**Question:** Why is this a classification problem?

**Theory answer:** Classification predicts a discrete class label. In supervised classification, the training data contains input features and known class labels.

**How used in the assignment:** The input features are candidate answers to political questions. The label is party. The model learns patterns in answers and predicts party membership.

## 4. Multiclass Classification

**Question:** Is this binary or multiclass classification?

**Theory answer:** It is multiclass classification because there are more than two possible party labels. Some algorithms handle multiclass directly; others use strategies such as one-vs-rest or one-vs-one.

**How used in the assignment:** The notebook predicts one party out of several parties. The tree-based models can handle multiclass classification directly in Scikit-learn.

## 5. Train-Test Split And Validation

**Question:** Why split the data before evaluating models?

**Theory answer:** Splitting data gives an unseen test set for final evaluation. Validation or cross-validation can be used for model comparison and hyperparameter tuning without touching the test set.

**How used in the assignment:** The notebook sets up a classification train/test workflow and uses cross-validation for comparing models before final evaluation.

## 6. Baseline Model

**Question:** Why use a baseline?

**Theory answer:** A baseline is a simple reference model. It tells us whether a more complex model is actually learning useful patterns or only looking good by chance.

**How used in the assignment:** The notebook includes a baseline before tree-based models. The classifiers should beat this simple reference to be meaningful.

## 7. Decision Tree

**Question:** How does a decision tree work?

**Theory answer:** A decision tree repeatedly splits the feature space using decision rules. Each internal node asks a question about a feature, and each leaf predicts a class. Splits are chosen to improve class purity, often using Gini impurity.

**How used in the assignment:** The notebook trains a decision tree to classify party from questionnaire answers. It can also inspect feature importance and decision logic.

## 8. Gini Impurity

**Question:** What is Gini impurity?

**Theory answer:** Gini impurity measures how mixed the classes are in a node. A node is pure if all samples belong to one class. Decision trees prefer splits that reduce impurity.

**How used in the assignment:** The decision tree uses impurity-based splitting to find political questions that separate parties well.

## 9. Overfitting In Trees

**Question:** Why can decision trees overfit?

**Theory answer:** A deep tree can memorize training examples by making very specific splits. This lowers training error but can hurt performance on unseen data.

**How used in the assignment:** The notebook uses model comparison and evaluation to check whether the tree generalizes. Hyperparameters such as depth and leaf size are typical ways to control overfitting.

## 10. Random Forest

**Question:** What is a random forest?

**Theory answer:** A random forest is an ensemble of decision trees. It uses bootstrapping/bagging and random feature subsets so trees make different errors. Final predictions are made by voting.

**How used in the assignment:** The notebook trains a random forest as a stronger alternative to a single tree. It can improve generalization and reduce variance.

## 11. Randomness In Random Forests

**Question:** Why is randomness useful in a random forest?

**Theory answer:** If all trees are very similar, averaging them gives little benefit. Random data samples and random feature subsets make trees diverse, which makes the ensemble more stable.

**How used in the assignment:** The model uses many trees trained with random variation. The final party prediction is based on the ensemble rather than one fragile tree.

## 12. Gradient-Boosted Trees

**Question:** What is gradient boosting?

**Theory answer:** Gradient boosting builds trees sequentially. Each new tree tries to correct errors made by previous trees. It can be powerful, but it is sensitive to hyperparameters and can overfit.

**How used in the assignment:** The notebook trains gradient-boosted trees and compares them against decision tree and random forest models.

## 13. Feature Importance

**Question:** What does feature importance mean in tree models?

**Theory answer:** Feature importance estimates which features contributed most to splitting decisions and reducing impurity. It is useful for interpretation but should not be treated as causal proof.

**How used in the assignment:** Important questionnaire answers indicate which political questions helped distinguish parties in the model.

## 14. Classification Metrics

**Question:** Which metrics matter here?

**Theory answer:** Accuracy gives the overall fraction of correct predictions. Precision, recall, F1, and the confusion matrix give more detail, especially when classes are imbalanced or errors differ by class.

**How used in the assignment:** Since party classes can have different sizes, confusion-matrix style interpretation is useful for seeing which parties are confused with each other.

## 15. Limitations

**Question:** What are the limitations?

**Theory answer:** Prediction does not prove political identity or causality. Models can be affected by imbalanced parties, missing answers, correlated questions, and the exact candidate-test design.

**How used in the assignment:** The notebook interprets results as patterns in questionnaire answers, not as objective truth about candidates.

## Fast Last-Minute Answers

- **Main task:** multiclass classification of party.
- **EDA topic:** descriptive analysis and box plots.
- **Models:** decision tree, random forest, gradient-boosted trees.
- **Tree split idea:** reduce impurity, often Gini impurity.
- **Best one-sentence defense:** I first describe the candidate data, then compare tree-based classifiers to see how well questionnaire answers can predict party membership and which questions are most informative.

# Assignment 3: Mushroom Foraging - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment3.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This is a supervised binary classification problem. The goal is to predict whether a mushroom is edible or poisonous from its features.

**How used in the assignment:** The notebook loads `secondary_data.csv`, prepares categorical and numerical features, trains logistic regression, and evaluates it with validation/test data, a confusion matrix, and classification metrics.

## 1. Data Preparation

**Question:** What data preparation steps are used?

**Theory answer:** Data preparation includes exploration, cleaning, feature selection, handling missing values, encoding categorical data, scaling numerical data, and splitting the data correctly.

**How used in the assignment:** The notebook checks shape, data types, missing values, target distribution, duplicates, and response variables. It drops columns with too many missing values, imputes missing values, scales numerical features, and one-hot encodes categorical features.

## 2. Missing Values

**Question:** How can missing values be handled?

**Theory answer:** Missing values can be removed, imputed, or sometimes kept as a separate category. Numerical values are often imputed with mean or median. Categorical values are often imputed with the most frequent value.

**How used in the assignment:** The notebook removes columns with more than 80% missing values based on training data, then imputes remaining missing values inside the preprocessing pipeline.

## 3. One-Hot Encoding

**Question:** Why do you one-hot encode categorical variables?

**Theory answer:** Many ML algorithms need numerical input. One-hot encoding converts categories into binary columns without creating a false ordinal relationship between category names.

**How used in the assignment:** Mushroom features are mostly categorical, so the notebook uses one-hot encoding before logistic regression.

## 4. Scaling

**Question:** Why scale features for logistic regression?

**Theory answer:** Scaling helps optimization because features have comparable ranges. It is especially useful for gradient-based algorithms and regularized models.

**How used in the assignment:** The notebook scales numerical features in the preprocessing pipeline before fitting logistic regression.

## 5. Split Before Preprocessing

**Question:** Why split before fitting preprocessing?

**Theory answer:** Preprocessing can learn information from data, such as imputation values, scaling parameters, or selected columns. If it is fitted on all data, validation/test information leaks into training.

**How used in the assignment:** The notebook splits into train, validation, and test before fitting preprocessing. The training set decides missing-value handling and scaler/encoder parameters.

## 6. Stratification

**Question:** What is stratification?

**Theory answer:** Stratification preserves class proportions in train, validation, and test splits. It is useful when classes are imbalanced or when we want representative splits.

**How used in the assignment:** The notebook uses stratified splitting so edible/poisonous proportions stay similar across splits.

## 7. Logistic Regression

**Question:** What is logistic regression?

**Theory answer:** Logistic regression is a classification algorithm that returns a probability. It applies a sigmoid function to a linear combination of features, producing values between 0 and 1.

**How used in the assignment:** The model predicts the probability that a mushroom belongs to one class. The final class is chosen using a decision threshold, usually 0.5.

## 8. Sigmoid Function

**Question:** Why does logistic regression use sigmoid?

**Theory answer:** The sigmoid function maps any real number to a value between 0 and 1. That makes it suitable for probability-like outputs in binary classification.

**How used in the assignment:** The logistic regression model turns weighted mushroom features into a probability of class membership.

## 9. Log Loss

**Question:** What loss function is used in logistic regression?

**Theory answer:** Logistic regression typically uses log loss. It penalizes confident wrong predictions strongly and encourages predicted probabilities to match true labels.

**How used in the assignment:** Scikit-learn handles optimization internally, but theoretically it is fitting coefficients to minimize classification loss.

## 10. Validation Methods

**Question:** Why use train, validation, and test sets?

**Theory answer:** The training set fits the model. The validation set supports model selection or threshold decisions. The test set is used only at the end for final unbiased evaluation.

**How used in the assignment:** The notebook uses validation results before final test evaluation. This follows the exam requirement that test data should be reserved for the final check.

## 11. Confusion Matrix

**Question:** What does a confusion matrix show?

**Theory answer:** A confusion matrix counts true positives, true negatives, false positives, and false negatives. It shows what kinds of classification errors the model makes.

**How used in the assignment:** For mushroom classification, false negatives can be especially dangerous if poisonous mushrooms are predicted as edible.

## 12. Accuracy, Precision, Recall, F1

**Question:** Explain classification metrics.

**Theory answer:** Accuracy is all correct predictions divided by all predictions. Precision asks: when the model predicts positive, how often is it right? Recall asks: out of actual positives, how many did it find? F1 is the harmonic mean of precision and recall.

**How used in the assignment:** The notebook reports classification metrics to evaluate more than just overall correctness. Recall is important when missing a dangerous class has high cost.

## 13. Precision-Recall Trade-Off

**Question:** What is the precision-recall trade-off?

**Theory answer:** Changing the classification threshold can increase recall but reduce precision, or increase precision but reduce recall. The best choice depends on the cost of different errors.

**How used in the assignment:** In mushroom foraging, the cost of predicting a poisonous mushroom as edible is high, so the threshold and recall for the dangerous class matter.

## 14. Overfitting

**Question:** How do you know if a model overfits?

**Theory answer:** A model overfits if it performs much better on training data than on validation/test data. It has learned noise or details that do not generalize.

**How used in the assignment:** The notebook checks validation and test performance instead of relying only on training performance.

## 15. Limitations

**Question:** What are the limitations?

**Theory answer:** Logistic regression is linear in the feature space after encoding. It may not capture every complex interaction unless features make them visible. Data quality, missing values, duplicates, and class distribution also affect reliability.

**How used in the assignment:** The notebook presents logistic regression as interpretable and exam-relevant, but not necessarily the best possible mushroom classifier.

## Fast Last-Minute Answers

- **Main task:** binary classification: edible vs poisonous.
- **Main model:** logistic regression.
- **Preprocessing:** missing values, scaling, one-hot encoding.
- **Most important safety point:** avoid data leakage by splitting before preprocessing.
- **Best one-sentence defense:** I prepare the mushroom data in a leakage-safe way, train logistic regression as a probability-based binary classifier, and evaluate the dangerous error types with a confusion matrix and classification metrics.

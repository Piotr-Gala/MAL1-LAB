# Assignment 4: Detecting Exoplanets - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment4.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This is a supervised binary classification problem. The goal is to classify Kepler objects as exoplanet candidates or false positives using measured astronomical features.

**How used in the assignment:** The notebook loads `exoplanet_dataset.csv`, prepares features, removes leakage-prone target-proxy columns, trains logistic regression and SVM models, and evaluates them with classification metrics and confusion matrices.

## 1. Data Preparation

**Question:** What data preparation is done?

**Theory answer:** Data preparation means inspecting the data, cleaning unusable columns, handling missing values/outliers, encoding the target, selecting features, and preparing train/validation/test splits.

**How used in the assignment:** The notebook checks data shape, types, missing values, summary statistics, renames technical columns, removes empty or identifier columns, and encodes the target.

## 2. Data Analysis

**Question:** What is the difference between data preparation and data analysis?

**Theory answer:** Data preparation makes data usable for modelling. Data analysis investigates structure, distributions, relationships, missingness, outliers, and possible problems.

**How used in the assignment:** The notebook analyzes missing-value percentages, outliers, correlations, and feature relationships before final modelling.

## 3. Missing Values

**Question:** How are missing values handled?

**Theory answer:** Missing values can be removed or imputed. The decision depends on how much data is missing and whether the feature is useful. Imputation must be fitted only on training data.

**How used in the assignment:** The notebook removes columns with 100% missing values and uses median imputation after splitting for remaining numerical missing values.

## 4. Outliers

**Question:** What are outliers and how can they be handled?

**Theory answer:** Outliers are extreme values compared with the rest of the data. They can be kept, removed, capped, transformed, or imputed depending on domain meaning and model sensitivity.

**How used in the assignment:** The notebook performs IQR-based outlier analysis. It does not blindly delete all outliers, because astronomical measurements may naturally contain extreme values.

## 5. Target Leakage

**Question:** What is a target-proxy feature?

**Theory answer:** A target-proxy feature is a feature that directly or indirectly contains the answer. Using it makes model performance unrealistically high because the model learns the label source instead of real patterns.

**How used in the assignment:** The notebook removes columns that are too directly related to the final Kepler disposition decision, so the classifier does not cheat.

## 6. Train-Validation-Test Split

**Question:** Why use train, validation, and test sets?

**Theory answer:** Training fits model parameters. Validation supports model comparison and hyperparameter choices. Test data is held back for final evaluation only.

**How used in the assignment:** The notebook uses separate splits and keeps preprocessing fitted on training data only.

## 7. Leakage-Safe Preprocessing

**Question:** What does leakage-safe preprocessing mean?

**Theory answer:** Any preprocessing step that learns from data must be fitted only on training data. Then the learned transformation is applied to validation/test data.

**How used in the assignment:** Median imputation and scaling are fitted after splitting, not on the full dataset.

## 8. Correlation And Multicollinearity

**Question:** What is multicollinearity?

**Theory answer:** Multicollinearity means features are strongly correlated with each other. It can make linear model coefficients unstable and harder to interpret.

**How used in the assignment:** The notebook checks correlation between features and discusses how correlated astronomical measurements can affect interpretation.

## 9. Logistic Regression

**Question:** Why use logistic regression?

**Theory answer:** Logistic regression is a probability-based classification algorithm. It uses a linear decision boundary in feature space and maps scores through a sigmoid function.

**How used in the assignment:** Logistic regression provides a simple interpretable baseline for exoplanet classification.

## 10. SVM

**Question:** How does SVM work?

**Theory answer:** SVM tries to find a separating hyperplane with the largest margin between classes. The closest points to the boundary are support vectors. Soft-margin SVM allows some margin violations.

**How used in the assignment:** The notebook trains an SVM classifier and compares it with logistic regression on the same prepared data.

## 11. Margin, Support Vectors, C

**Question:** What does the SVM hyperparameter `C` do?

**Theory answer:** `C` controls the trade-off between a wide margin and classification errors. Low `C` allows more violations and stronger regularization. High `C` tries harder to classify training points correctly and can overfit.

**How used in the assignment:** The SVM model depends on scaled features and the chosen regularization strength. This is discussed as part of model limitations and comparison.

## 12. Kernel Functions

**Question:** What is an SVM kernel?

**Theory answer:** A kernel lets SVM model nonlinear boundaries by computing similarity as if data were mapped into a higher-dimensional space. Common kernels include linear, polynomial, and RBF.

**How used in the assignment:** The assignment's SVM section is exam-relevant because exoplanet data may not be perfectly linearly separable. If a nonlinear kernel is used or discussed, it explains how SVM can handle more complex boundaries.

## 13. Scaling For SVM

**Question:** Why is scaling important for SVM?

**Theory answer:** SVM uses distances and margins. If features have very different scales, large-scale features dominate the hyperplane.

**How used in the assignment:** The notebook scales features before SVM, which is necessary for a fair model.

## 14. Confusion Matrix

**Question:** What does the confusion matrix tell you here?

**Theory answer:** It shows true positives, true negatives, false positives, and false negatives. For exoplanets, it tells whether the model confuses candidates with false positives.

**How used in the assignment:** The notebook uses confusion matrices to compare logistic regression and SVM beyond a single accuracy score.

## 15. Classification Metrics

**Question:** Which metrics are used and why?

**Theory answer:** Accuracy measures overall correctness. Precision measures reliability of positive predictions. Recall measures how many actual positives are found. F1 balances precision and recall.

**How used in the assignment:** The notebook reports classification metrics for both models, making it possible to compare error trade-offs.

## 16. Limitations

**Question:** What are the limitations?

**Theory answer:** Classification depends on dataset quality, missing values, target definitions, class balance, preprocessing decisions, and possible target leakage. SVM and logistic regression also depend strongly on scaling and feature representation.

**How used in the assignment:** The notebook explicitly discusses limitations and avoids claiming that the model discovers planets independently from the scientific pipeline.

## Fast Last-Minute Answers

- **Main task:** binary classification: candidate vs false positive.
- **Models:** logistic regression and SVM.
- **Critical issue:** target-proxy feature removal.
- **SVM keyword:** margin, support vectors, soft margin, `C`, kernel.
- **Best one-sentence defense:** I clean the Kepler dataset, remove leakage-prone columns, apply preprocessing after splitting, and compare logistic regression with SVM using confusion matrices and classification metrics.

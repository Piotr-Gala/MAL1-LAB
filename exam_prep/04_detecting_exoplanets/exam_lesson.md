# Assignment 4: Detecting Exoplanets - ADHD-Friendly Oral Exam Lesson

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

Assignment 4 is about detecting exoplanets using the Kepler dataset.

The goal is to classify Kepler objects as planet candidates or false positives.

This is a supervised binary classification problem.

The notebook loads `exoplanet_dataset.csv`, cleans the data, removes leakage-prone target-proxy features, applies preprocessing after splitting, and compares Logistic Regression with SVM.

The main result is that SVM performs better on the final test set, with about 85.5% accuracy and F1-score.

## Terms From This Section

`exoplanet`

```text
A planet outside our solar system.
```

`Kepler object`

```text
An object observed by the Kepler mission and evaluated as a possible planet signal.
```

`planet candidate`

```text
An object that may be an exoplanet.
In the target encoding, candidate is class 1.
```

`false positive`

```text
An object that looks like a planet signal but is not considered a planet candidate.
In the target encoding, false positive is class 0.
```

`supervised binary classification`

```text
The model learns from labelled examples and predicts one of two classes.
Here: false positive or candidate.
```

## Exam Sentences

```text
This is a supervised binary classification problem.
The goal is to classify Kepler objects as planet candidates or false positives using measured astronomical features.
The notebook compares Logistic Regression and SVM after leakage-safe preprocessing.
```

---

# 1. Loading And Initial Data Preparation

The notebook starts by loading the Kepler exoplanet dataset from `exoplanet_dataset.csv`.

The dataset has:

```text
9564 rows
49 columns
```

One row represents one Kepler object of interest.

The notebook checks the first rows, data types, missing values and summary statistics.

It also renames technical Kepler column names to more readable names.

## Terms From This Section

`row`

```text
One observation.
Here: one Kepler object of interest.
```

`column`

```text
One variable or measurement.
Here: astronomical properties, flags, identifiers or labels.
```

`data preparation`

```text
Making the dataset usable for modelling.
Examples: cleaning columns, encoding target, preparing X and y.
```

`summary statistics`

```text
Basic numerical summaries like mean, minimum, maximum and quartiles.
```

`renaming columns`

```text
Changing technical column names into clearer names for interpretation.
```

## Exam Sentences

```text
I started by loading and inspecting the Kepler dataset.
I checked the shape, data types, missing values and summary statistics.
I also renamed technical columns to make the notebook easier to interpret.
```

## Where In Code

```text
Cells 4-5: load dataset and inspect first rows.
Cells 6-8: initial overview and column renaming.
```

---

# 2. Target And Classes

The final target is `KeplerDispositionStatus`.

The notebook uses a binary version of the target:

```text
0 = FALSE POSITIVE
1 = CANDIDATE
```

The target is almost balanced:

```text
FALSE POSITIVE: 4847
CANDIDATE: 4717
```

This means accuracy is more meaningful than in a heavily imbalanced dataset, but precision, recall and F1-score are still important.

## Terms From This Section

`target`

```text
The thing the model should predict.
Here: KeplerDispositionStatus.
```

`class`

```text
The category assigned to an object.
Here: false positive or candidate.
```

`binary target`

```text
A target with two possible values.
```

`balanced dataset`

```text
Classes have similar numbers of examples.
Here the classes are almost balanced.
```

`accuracy`

```text
The percentage of correct predictions overall.
```

## Exam Sentences

```text
The final target is KeplerDispositionStatus.
It is a binary target with false positives encoded as 0 and candidates encoded as 1.
The classes are almost balanced, but I still use precision, recall and F1-score for a fuller evaluation.
```

## Where In Code

```text
Cells 9-13: missing values, target encoding and target distribution.
```

---

# 3. Removing Empty And Identifier Columns

The notebook removes two columns with 100% missing values.

It also removes identifier columns such as object IDs and names.

Removed examples:

```text
EquilibriumTemperatureUpperUnc, K
EquilibriumTemperatureLowerUnc, K
KepID
KOIName
KeplerName
TCEDeliver
```

Identifier columns are removed because they do not describe physical properties and can encourage memorization.

## Terms From This Section

`100% missing`

```text
The column has no useful observed values.
It cannot help the model.
```

`identifier column`

```text
A column used to identify a row, such as an ID or name.
It usually should not be used as a predictive feature.
```

`memorization`

```text
The model may learn object identities instead of real patterns.
```

`physical property`

```text
A real measured characteristic of the object.
Examples: radius, orbital period, flux.
```

## Exam Sentences

```text
I removed columns with 100% missing values because they contain no useful information.
I removed identifier columns because they do not describe physical properties and may cause the model to memorize objects.
```

## Where In Code

```text
Cells 9-13: removing empty columns and identifier columns.
```

---

# 4. Missing Values And Outliers

The notebook analyzes missing values and outliers before modelling.

Dropping every row with missing values would remove:

```text
1761 rows
```

So the notebook uses imputation instead of dropping all incomplete rows.

Outliers are analyzed with the IQR method, but they are not blindly removed.

Extreme astronomical measurements may represent real physical cases.

## Terms From This Section

`missing value`

```text
A value that is absent or unknown.
```

`dropna()`

```text
A method that removes rows with missing values.
In this notebook it is checked but not used for final modelling.
```

`imputation`

```text
Filling missing values with a replacement value.
Here, median imputation is used later.
```

`outlier`

```text
An extreme value compared with the rest of the data.
```

`IQR method`

```text
A common method for detecting outliers using the interquartile range.
```

## Exam Sentences

```text
I analyzed missing values and outliers, but I did not remove all incomplete rows or extreme values.
Dropping all rows with missing values would remove 1761 observations.
I kept outliers because extreme astronomical measurements may be valid physical cases.
```

## Where In Code

```text
Cells 9-11: missing value percentages.
Cells 14-18: IQR outlier analysis and dropna check.
Cell 28: median imputation after split.
```

---

# 5. Target-Proxy Feature Removal

This is one of the most important parts of Assignment 4.

Before modelling, the notebook removes target-related columns and explicit false-positive flag columns.

Removed target-related columns:

```text
DispositionScore
KeplerDispositionStatus
ArchiveDispositionStatus
```

Removed false-positive flag columns:

```text
NotTransit-LikeFalsePositiveFlag
koi_fpflag_ss
CentroidOffsetFalsePositiveFlag
EphemerisMatchIndicatesContaminationFalsePositiveFlag
```

These columns are very close to the labelling process.

Keeping them would give the model an unrealistic shortcut.

Removing them lowers the score, but makes the result more honest and easier to defend.

## Terms From This Section

`target-proxy feature`

```text
A feature that directly or indirectly contains the answer.
It lets the model cheat.
```

`leakage-prone`

```text
Risky because it may leak information about the target into the features.
```

`false-positive flag`

```text
A column indicating a reason why an object may be a false positive.
These flags are close to the target decision.
```

`shortcut`

```text
The model learns an easy label-related signal instead of real physical patterns.
```

`conservative result`

```text
A lower but more defensible score.
```

## Exam Sentences

```text
I removed the explicit false-positive flag columns before modelling.
These columns are very close to the target labelling process, so keeping them would create a target-proxy shortcut.
Removing them makes the experiment more conservative and easier to defend.
```

## Where In Code

```text
Cell 20: removing target-related columns and explicit false-positive flag columns from X.
```

---

# 6. X, y And Train-Validation-Test Split

The notebook prepares `X` and `y`.

`y` is the target:

```text
KeplerDispositionStatus
```

`X` contains the remaining features after removing identifiers, target-related columns and false-positive flags.

Then the data is split into train, validation and test sets using stratification.

Notebook detail:

```text
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

## Terms From This Section

`X`

```text
The feature matrix.
Here: astronomical features used as input.
```

`y`

```text
The target vector.
Here: false positive or candidate.
```

`training set`

```text
Data used to fit model parameters.
```

`validation set`

```text
Data used for model selection and hyperparameter tuning.
```

`test set`

```text
Data held back for final evaluation only.
```

`stratification`

```text
Keeping class proportions similar in train, validation and test.
```

## Exam Sentences

```text
I use a train-validation-test split with stratification.
The training set is used for fitting, the validation set for model selection, and the test set only for final evaluation.
```

## Where In Code

```text
Cells 19-22: prepare X and y, remove proxy features, and split data.
```

---

# 7. Leakage-Safe Preprocessing

After splitting, preprocessing is fitted only on training data.

The notebook performs:

```text
correlation filtering
median imputation
standard scaling
```

The important rule is:

```text
Fit on train, transform validation and test.
```

This avoids data leakage.

## Terms From This Section

`leakage-safe preprocessing`

```text
Any preprocessing that learns from data is fitted only on training data.
Then it is applied to validation/test data.
```

`fit`

```text
Learn parameters from data.
For example, an imputer learns medians and a scaler learns means/stds.
```

`transform`

```text
Apply an already learned preprocessing step.
```

`data leakage`

```text
When validation or test information influences training.
This makes evaluation too optimistic.
```

`median imputation`

```text
Replacing missing values with the median learned from training data.
```

`standard scaling`

```text
Scaling features using mean and standard deviation.
```

## Exam Sentences

```text
The split is done before correlation filtering, imputation and scaling.
The imputer and scaler are fitted only on the training data.
Validation and test data are only transformed, so they do not influence preprocessing.
```

## Where In Code

```text
Cells 21-28: split first, then correlation filtering, imputation and scaling.
Cell 41: final retraining on train plus validation.
```

---

# 8. Correlation And Multicollinearity

The notebook calculates a correlation matrix only on the training features.

Then it removes two highly correlated columns from all splits:

```text
PlanetaryRadiusLowerUnc, Earthradii
InsolationFluxLowerUnc, Earthflux
```

This reduces redundancy and multicollinearity.

Doing this only on training data avoids leakage.

## Terms From This Section

`correlation`

```text
A measure of how strongly two variables move together.
```

`correlation matrix`

```text
A table of correlations between features.
```

`multicollinearity`

```text
When features are strongly correlated with each other.
It can make linear model coefficients unstable and harder to interpret.
```

`redundancy`

```text
Two features carry very similar information.
```

`train-only feature selection`

```text
Choosing features using only training data.
Validation/test data should not influence the choice.
```

## Exam Sentences

```text
I calculated correlations only on the training data to avoid leakage.
Then I removed two highly correlated features from all splits to reduce redundancy and multicollinearity.
```

## Where In Code

```text
Cells 23-26: train-only correlation analysis and highly correlated feature removal.
```

---

# 9. Logistic Regression

Logistic Regression is the first model.

It is a probability-based classification model.

It produces a score from a linear combination of features and maps it through a sigmoid function.

The notebook tests several `C` values and selects the best one using validation F1-score.

Result:

```text
Best C: 10
Validation F1-score: about 0.803
Final test accuracy: about 0.821
Final test F1-score: about 0.820
```

## Terms From This Section

`Logistic Regression`

```text
A classification model that estimates class probabilities.
```

`linear combination`

```text
Features multiplied by coefficients and added together.
```

`sigmoid function`

```text
Maps any real number to a value between 0 and 1.
Useful for probability-like outputs.
```

`C`

```text
Inverse regularization strength.
Smaller C means stronger regularization.
Larger C means weaker regularization.
```

`validation F1-score`

```text
F1-score calculated on the validation set, used for model selection.
```

## Exam Sentences

```text
Logistic Regression provides a simple interpretable baseline for exoplanet classification.
I tuned Logistic Regression by testing several C values and selecting the one with the best validation F1-score.
The best C was 10, and the final test F1-score was about 0.820.
```

## Where In Code

```text
Cells 29-34: Logistic Regression tuning and validation.
Cell 41: final retraining.
Cells 42-43: final test evaluation.
```

---

# 10. Support Vector Machine

SVM is the second model.

SVM tries to find a separating hyperplane with the largest possible margin between classes.

The closest points to the boundary are called support vectors.

The notebook tests different `C` values and two kernels:

```text
linear
rbf
```

The best model is:

```text
C = 10
kernel = rbf
```

Final result:

```text
Final test accuracy: about 0.855
Final test F1-score: about 0.855
```

SVM performs better than Logistic Regression in this notebook.

## Terms From This Section

`SVM`

```text
Support Vector Machine.
A classifier that tries to separate classes with a maximum-margin boundary.
```

`hyperplane`

```text
The decision boundary that separates classes.
In 2D it is a line; in higher dimensions it is a hyperplane.
```

`margin`

```text
Distance between the decision boundary and the closest training points.
SVM tries to make this margin large.
```

`support vectors`

```text
The closest points to the decision boundary.
They strongly influence the SVM boundary.
```

`soft margin`

```text
Allows some mistakes or margin violations to get a better general boundary.
```

## Exam Sentences

```text
SVM tries to find a separating hyperplane with the largest margin between classes.
The closest points to the boundary are support vectors.
The best SVM used an RBF kernel with C equal to 10 and performed best on the final test set.
```

## Where In Code

```text
Cells 35-39: SVM tuning and validation.
Cell 41: final retraining.
Cells 42-43: final test evaluation.
```

---

# 11. C In SVM

The SVM hyperparameter `C` controls the trade-off between margin width and classification errors.

Low `C`:

```text
allows more violations
stronger regularization
wider margin
```

High `C`:

```text
tries harder to classify training points correctly
weaker regularization
can overfit
```

In Assignment 4, the best SVM uses:

```text
C = 10
```

## Terms From This Section

`margin violation`

```text
A point is on the wrong side of the margin or boundary.
Soft-margin SVM allows some of these.
```

`regularization`

```text
A way to control model complexity and reduce overfitting.
```

`overfitting`

```text
The model fits training data too closely and performs worse on unseen data.
```

`trade-off`

```text
A balance between two goals.
Here: wide margin vs fewer training errors.
```

## Exam Sentences

```text
In SVM, C controls the trade-off between a wide margin and classification errors.
Low C allows more violations and gives stronger regularization.
High C tries harder to classify training points correctly and can overfit.
```

---

# 12. Kernel Functions

The notebook compares a linear kernel and an RBF kernel for SVM.

A linear kernel creates a linear decision boundary.

An RBF kernel can model nonlinear relationships.

The best SVM uses the RBF kernel.

This suggests that the class boundary may be nonlinear after removing the explicit false-positive flags.

## Terms From This Section

`kernel`

```text
A function that lets SVM model more complex boundaries.
```

`linear kernel`

```text
Creates a linear decision boundary.
```

`RBF kernel`

```text
Radial Basis Function kernel.
Can model nonlinear boundaries by measuring similarity in a more flexible way.
```

`nonlinear boundary`

```text
A decision boundary that is not a straight line/hyperplane.
```

## Exam Sentences

```text
A linear kernel creates a linear decision boundary, while an RBF kernel can model nonlinear relationships.
The best model used the RBF kernel, which suggests that the class boundary may not be purely linear.
```

## Where In Code

```text
Cells 35-39: SVM models with linear and RBF kernels.
```

---

# 13. Scaling For Logistic Regression And SVM

The notebook scales features before modelling.

Scaling is especially important for SVM because SVM uses distances and margins.

If features have very different numerical scales, large-scale features can dominate the hyperplane.

Scaling also helps Logistic Regression optimization.

## Terms From This Section

`scaling`

```text
Putting features on comparable numeric ranges.
```

`feature scale`

```text
The numeric range of a feature.
Example: one feature may range from 0 to 1, another from 0 to 100000.
```

`dominate`

```text
Have too much influence only because of large numerical values.
```

`optimization`

```text
The process of finding model parameters that minimize loss.
```

## Exam Sentences

```text
Scaling is important for SVM because SVM uses distances and margins.
If features have very different scales, large-scale features can dominate the decision boundary.
Scaling also helps Logistic Regression optimization.
```

## Where In Code

```text
Cell 28: StandardScaler fitted on training data and applied to validation/test data.
```

---

# 14. Confusion Matrix

The confusion matrix shows error types, not only total performance.

Classes:

```text
0 = FALSE POSITIVE
1 = CANDIDATE
```

For candidate as the positive class:

```text
                 Predicted false positive   Predicted candidate
Actual false positive    TN                 FP
Actual candidate         FN                 TP
```

This helps compare Logistic Regression and SVM beyond accuracy.

## Terms From This Section

`confusion matrix`

```text
A table showing correct and incorrect predictions by class.
```

`TP`

```text
True positive.
Actual candidate, predicted candidate.
```

`TN`

```text
True negative.
Actual false positive, predicted false positive.
```

`FP`

```text
False positive.
Actual false positive, predicted candidate.
```

`FN`

```text
False negative.
Actual candidate, predicted false positive.
```

## Exam Sentences

```text
The confusion matrix shows which types of classification errors the model makes.
For this assignment, it shows whether the model confuses candidates with false positives.
```

## Where In Code

```text
Cell 33: Logistic Regression validation confusion matrix.
Cell 39: SVM validation confusion matrix.
Cell 43: final test confusion matrices.
```

---

# 15. Classification Metrics

The notebook uses classification metrics to compare models.

Accuracy measures overall correctness.

Precision measures how reliable candidate predictions are.

Recall measures how many actual candidates are found.

F1-score balances precision and recall.

Final test results:

```text
Logistic Regression:
Accuracy: about 0.821
F1-score: about 0.820

SVM:
Accuracy: about 0.855
F1-score: about 0.855
```

## Terms From This Section

`accuracy`

```text
Correct predictions divided by all predictions.
```

`precision`

```text
When the model predicts candidate, how often it is correct.
```

`recall`

```text
How many true candidates the model finds.
```

`F1-score`

```text
A balance between precision and recall.
```

`model comparison`

```text
Comparing models using the same data splits and metrics.
```

## Exam Sentences

```text
Accuracy gives overall correctness, but precision and recall show different types of errors.
F1-score is useful because it balances precision and recall.
SVM performed better than Logistic Regression on the final test set.
```

## Where In Code

```text
Cells 30, 32, 36, 38 and 42: classification metrics for model comparison.
```

---

# 16. Final Model And Limitations

After tuning, both models are retrained on train plus validation data.

Then both are evaluated on the untouched test set.

SVM is selected because it performs better on the final test set.

The most important limitation is target-proxy risk.

The explicit false-positive flags were removed, which reduces the shortcut risk.

However, this does not prove that all indirect proxy information is gone.

Some remaining measurement features may still be related to the labelling process.

## Terms From This Section

`train plus validation`

```text
After choosing hyperparameters, the final model can be trained on more data.
```

`untouched test set`

```text
Test data that was not used for fitting, preprocessing decisions or tuning.
```

`limitation`

```text
A weakness or boundary of the solution.
```

`indirect proxy`

```text
A remaining feature that may still be related to the label, even if it is not an explicit target column.
```

`honest result`

```text
A result that may be lower but is more defensible because it avoids obvious leakage.
```

## Exam Sentences

```text
I selected SVM because it achieved better final test performance than Logistic Regression.
The explicit false-positive flags were removed, so the scores are lower but more realistic.
The main limitation is that removing explicit flags does not prove that all indirect proxy information is gone.
```

## Where In Code

```text
Cells 40-45: final model comparison, test evaluation and limitations.
```

---

# 17. If Asked What Was Fixed

The original version allowed the model to use explicit false-positive flag columns.

Those columns were removed from `X` before modelling.

As a result, performance dropped from the old flag-based scores to more realistic final scores.

This is not a bad thing.

It means the model is no longer using the most obvious shortcut.

## Terms From This Section

`flag-based score`

```text
A score helped by explicit false-positive flags.
It may look very high but is less defensible.
```

`realistic score`

```text
A score after removing obvious shortcuts.
Lower, but more honest.
```

## Exam Sentences

```text
The original version allowed the model to use explicit false-positive flag columns.
I removed those columns before modelling.
The performance dropped, but the new result is more realistic and easier to defend.
```

---

# 18. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Data loading:
Cells 4-5, where I load exoplanet_dataset.csv and inspect first rows.

Initial overview and renaming:
Cells 6-8.

Missing values and target encoding:
Cells 9-13.

Outlier analysis and dropna check:
Cells 14-18.

X and y:
Cells 19-20, where I select KeplerDispositionStatus as target.

Target-proxy feature removal:
Cell 20, where I remove target-related columns and explicit false-positive flags.

Train-validation-test split:
Cells 21-22.

Correlation filtering:
Cells 23-26, calculated only on training data.

Imputation and scaling:
Cell 28.

Logistic Regression:
Cells 29-34.

SVM:
Cells 35-39.

Final model comparison:
Cells 40-45.

Confusion matrices:
Cells 33, 39 and 43.
```

---

# 19. A4 In 30 Seconds

```text
Assignment 4 is about classifying Kepler objects as planet candidates or false positives.
This is supervised binary classification.

I first load and inspect the Kepler dataset, remove empty and identifier columns, encode the target, and analyze missing values and outliers.

Before modelling, I remove target-related columns and explicit false-positive flags because they are too close to the target labelling process.
Then I split the data into train, validation and test sets.

After the split, I do train-only correlation filtering, median imputation and standard scaling.
Finally, I compare Logistic Regression and SVM.
SVM with an RBF kernel and C equal to 10 performs best, with about 85.5% final test accuracy and F1-score.
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
Target-proxy features are features that contain information too close to the answer.
In my assignment, I removed the false-positive flag columns.
The reason is that otherwise the model could use an unrealistic shortcut.
```

---

# 21. Top Words To Memorize

```text
target = KeplerDispositionStatus
binary classification = false positive vs candidate
target-proxy feature = feature too close to the answer
false-positive flag = explicit shortcut column
leakage-safe preprocessing = fit on train, transform validation/test
correlation = relationship between two variables
multicollinearity = strongly correlated features
Logistic Regression = probability-based classifier
SVM = maximum-margin classifier
margin = distance from boundary to closest points
support vectors = closest points to SVM boundary
kernel = lets SVM model more complex boundaries
RBF = nonlinear SVM kernel
C = regularization / margin-error trade-off
F1-score = balance between precision and recall
```

Final survival sentence:

```text
The key point in this assignment is that I removed leakage-prone target-proxy features, applied preprocessing only after splitting, and compared Logistic Regression with SVM using fair validation and final test evaluation.
```

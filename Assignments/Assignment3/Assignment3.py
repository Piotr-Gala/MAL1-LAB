# %% [markdown]
# # 3. Mushroom foraging

# %% [markdown]
# The [mushroom dataset](https://www.kaggle.com/datasets/dhinaharp/mushroom-dataset) contains data about approximately 60000 mushrooms, and your task is to classify them as either edible or poisonous. You can read about the features [here](https://www.kaggle.com/datasets/uciml/mushroom-classification) and import the data using:

# %%
import pandas as pd
pd.set_option('display.max_columns', 1000)
df = pd.read_csv('secondary_data.csv', delimiter = ';')
df.head()

# %% [markdown]
# It's up to you how you approach this data, but at a minimum, your analysis should include:
# 
# * Informed **data preparation**.
# * Use the **logistic regression**.
# * Three different **validation methodologies** used to tune hyperparameters, discussing the pros and cons of each.
# * **Confusion matrices** for your models, and associated comments.
#  * Evaluate your models using the following metrics and provide associated comments for each:
#    - **ROC curve**
#    - **precision-recall curve**
#    - **F1 score**
#    - **accuracy**
#    - **recall**
#    - **precision**
# * A discussion of which **performance metric** is most relevant for the evaluation of your models.
# 
# Please remember to provide associated comments for each metric; it is not enough to just provide the metric values.

# %% [markdown]
# # 3. Mushroom Foraging
# 
# This notebook analyzes the mushroom dataset and builds a logistic regression model to classify mushrooms as edible or poisonous.
# 
# The analysis includes: 
# - data preparation, 
# - exploratory inspection, 
# - logistic regression, 
# - three validation methodologies for hyperparameter tuning, 
# - confusion matrices and classification metrics, 
# - discussion of strengths and weaknesses of the approaches.

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
    LeaveOneOut
)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
    auc,
    roc_auc_score, 
    precision_recall_curve, 
    classification_report
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.max_columns", 100)


# %% [markdown]
# # Load data

# %%
df = pd.read_csv("secondary_data.csv", delimiter=";")
df.head()

df.shape

df.info()

df.describe(include="all").T

# %% [markdown]
# # Initial inspection

# %%
df.dtypes

df.isna().sum().sort_values(ascending=False)

df["class"].value_counts()
df["class"].value_counts(normalize=True)

# %% [markdown]
# # Data cleaning and preparation

# %%
# Drop duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

df = df.drop_duplicates()

# Ckeck missing values ratio
missing_ratio = df.isna().mean().sort_values(ascending=False)
missing_ratio

# Drop columns with more than 80% missing values
cols_to_drop = missing_ratio[missing_ratio > 0.80].index.tolist()
print(f"Columns to drop (more than 80% missing): {cols_to_drop}")

df = df.drop(columns=cols_to_drop)

X = df.drop(columns=["class"])
y = df["class"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_features, categorical_features

# %% [markdown]
# #### Interpretation
# 
# The dataset contains over 61,000 observations with mostly categorical features and a few numerical ones.
# 
# Several columns have a high proportion of missing values (above 80%) and were removed to reduce noise and improve model reliability.
# 
# Duplicate rows were dropped to avoid bias in the training process.
# 
# The target variable is slightly imbalanced, meaning accuracy alone may not be a sufficient evaluation metric.

# %% [markdown]
# # Preprocessing + model

# %%
# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features) 
])

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

# Train / Val / Test split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# %% [markdown]
# #### Interpretation
# 
# A preprocessing pipeline was created to handle numerical and categorical features separately.  
# Missing values were imputed (median for numerical, most frequent for categorical), and categorical variables were encoded using one-hot encoding. Numerical features were scaled to improve model performance.
# 
# The preprocessing steps were combined with logistic regression into a single pipeline to prevent data leakage and ensure consistent transformations.
# 
# The dataset was split into training, validation, and test sets using stratification to preserve class distribution. The validation set is used for model tuning, while the test set is reserved for final evaluation.

# %% [markdown]
# # Validation method 1 - single validation split

# %%
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

# Metrics
acc = accuracy_score(y_val, y_val_pred)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_val, y_val_pred))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.show()

# %% [markdown]
# #### Interpretation
# 
# The model achieves around 0.81 accuracy, indicating overall good performance.
# 
# However, the confusion matrix shows that a significant number of poisonous mushrooms are misclassified as edible (false negatives). This is a critical issue, as such errors could lead to dangerous real-world consequences.
# 
# Therefore, recall for the poisonous class is more important than overall accuracy in this problem.

# %%
recall_poisonous = recall_score(y_val, y_val_pred, pos_label="p")
print(f"Recall (poisonous): {recall_poisonous:.3f}")

# %% [markdown]
# The recall for the poisonous class is approximately 0.81, meaning that around 81% of dangerous mushrooms are correctly identified, while about 19% are still misclassified as edible.

# %% [markdown]
# # Validation method 2 - K-Fold cross-validation

# %%
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"] # model training engine
}

scorer = make_scorer(f1_score, pos_label='p')
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring=scorer,
    n_jobs=-1
)

grid.fit(X_train_val, y_train_val)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# %%
kf = KFold(n_splits=5, shuffle = True, random_state=42)

cv_scores = cross_val_score( 
    grid.best_estimator_,
    X_train_val,
    y_train_val,
    cv = kf, 
    scoring=make_scorer(f1_score, pos_label='p')
)

print("Cross-validation F1 scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())

# %%
best_model = grid.best_estimator_

# %% [markdown]
# #### Interpretation
# 
# K-fold cross-validation provides a more stable estimate of model performance compared to a single validation split.
# 
# The mean F1 score is around 0.81 with a very low standard deviation (~0.004), indicating that the model performs consistently across different data splits.
# 
# This suggests that the model is stable and not highly sensitive to how the data is partitioned.

# %% [markdown]
# # Validation method 3 Nested Cross-Validation

# %%
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

nested_scores = cross_val_score(
    GridSearchCV(model, param_grid, cv=5, scoring=scorer),
    X_train_val,
    y_train_val,
    cv=outer_cv
)

print("Nested CV F1:", nested_scores.mean())

# %% [markdown]
# #### Interpretation
# 
# Nested cross-validation provides a more reliable estimate of model performance, as hyperparameter tuning is performed independently within each training fold.
# 
# The mean F1 score is around 0.83, slightly higher than in standard cross-validation, indicating good generalization performance.
# 
# However, this method is computationally expensive, as it involves running multiple grid searches, which significantly increases training time.

# %% [markdown]
# # Validation method 4 - Leave One Out cross-validation

# %%
# sample for computational reasons
X_sample = X_train_val.sample(2000, random_state=42)
y_sample = y_train_val.loc[X_sample.index]

loo = LeaveOneOut()

loo_scores = cross_val_score(
    model, 
    X_sample,
    y_sample,
    cv = loo,
    scoring="f1_weighted",
    n_jobs=-1
)

print("LOO mean F1:", loo_scores.mean())

# %% [markdown]
# #### Interpretation
# 
# Leave-One-Out cross-validation uses almost all data for training in each iteration, which makes it nearly unbiased.
# 
# However, it is computationally very expensive and not practical for large datasets, which is why only a subset of the data was used.
# 
# In this case, it does not provide significant advantages over K-Fold cross-validation.

# %% [markdown]
# # Final model on held-out test set

# %%
model.fit(X_train_val, y_train_val)

y_test_pred = model.predict(X_test)

# %% [markdown]
# # Metrics + Confusion Matrix

# %%
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, pos_label="p"))
print("Recall:", recall_score(y_test, y_test_pred, pos_label="p"))
print("F1 Score:", f1_score(y_test, y_test_pred, pos_label="p"))
print("ROC AUC:", roc_auc_score(y_test, y_scores))


# %%
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# %%
# test if probabilities are not reversed
model.classes_

# %%
y_scores = model.predict_proba(X_test)[:, 1] # works when model.classes_ == ['e', 'p']

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label="p")
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %%
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label="p")

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# %% [markdown]
# #### Final Interpretation
# 
# The final model achieves around 0.81 accuracy and an F1 score of approximately 0.83 for the poisonous class, indicating solid overall performance.
# 
# The confusion matrix shows that most predictions are correct; however, a non-negligible number of poisonous mushrooms are misclassified as edible (false negatives). This is a critical issue, as such errors could lead to serious real-world consequences.
# 
# The ROC curve (AUC ≈ 0.88) indicates good discriminative ability, meaning the model can effectively distinguish between edible and poisonous mushrooms. The precision-recall curve further confirms a reasonable balance between precision and recall.
# 
# In this problem, recall for the poisonous class is the most important metric, as minimizing false negatives is crucial for safety. While the model performs well, it is not sufficiently reliable for real-world deployment without further improvements.

# %% [markdown]
# ## Conclusion
# 
# Three validation approaches were compared: a single validation split, K-Fold cross-validation, and nested cross-validation.
# 
# The single split method provided a quick estimate but was less reliable due to its dependence on a specific data partition. K-Fold cross-validation offered more stable and consistent results, while nested cross-validation provided the most reliable estimate by separating hyperparameter tuning from model evaluation.
# 
# The final model achieved solid performance on the test set, with good accuracy and F1 score. However, the presence of false negatives (poisonous mushrooms classified as edible) highlights an important limitation.
# 
# Overall, recall for the poisonous class is the most critical metric in this problem. While the model performs well, further improvements would be needed before considering real-world deployment.



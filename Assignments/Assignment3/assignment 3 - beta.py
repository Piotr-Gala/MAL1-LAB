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
    precision_recall_curve
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
# Duplicate rows were checked and removed to avoid redundancy in the dataset.
# 
# Next, the proportion of missing values was calculated for each feature.  
# Columns with more than 80% missing values were removed, as they provide limited reliable information and could negatively impact model performance.
# 
# This approach ensures that feature selection is based on a systematic criterion rather than arbitrary decisions.
# 
# After cleaning, the dataset was split into numerical and categorical features, which will be handled differently in the preprocessing pipeline.

# %% [markdown]
# # Preprocessing + model

# %% [markdown]
# ### Preprocessing for modeling

# %%
# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocesing for categorical features
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessors
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features) 
])

# %% [markdown]
# #### Interpretation
# 
# Numerical and categorical features require different preprocessing steps.  
# Numerical variables are imputed using the median and scaled, while categorical variables are imputed with the most frequent value and encoded using one-hot encoding.
# 
# This transformation ensures that all features are in a suitable format for logistic regression.

# %% [markdown]
# ### Model (logistic regression)

# %%
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000))
])

# %% [markdown]
# #### Interpretation
# 
# Logistic regression was selected as the classification model.<br>
# It is a simple and interpretable algorithm suitable for binary classification problems.
# 
# The model is combined with preprocessing in a pipeline to prevent data leakage and ensure consistent transformations during training and evaluation.

# %% [markdown]
# ## Train / Val / Test split

# %%
# Split into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify = y,
    random_state=42
)

# Split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25, # 0.25 x 0.8 = 0.2
    stratify = y_train_val,
    random_state=42
)

# %% [markdown]
# #### Interpretation
# 
# The dataset was split into training, validation, and test sets. <br>
# The test set is kept completely separate for final evaluation, while the validation set is used for model selection and hyperparameter tuning.
# 
# This approach mimics a real-world scenario where model decisions are made without accessing the test data.

# %% [markdown]
# # Validation method 1 - single validation split

# %%
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation F1:", f1_score(y_val, y_val_pred, pos_label='e'))

# %% [markdown]
# #### Interpretation
# 
# The model was trained on the training set and evaluated on a separate validation set.  
# This provides a simple estimate of performance, but the result depends on a single split of the data and may therefore be unstable.

# %% [markdown]
# # Validation method 2 - K-Fold cross-validation

# %%
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"] # model training engine
}

scorer = make_scorer(f1_score, pos_label='e')
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
    scoring = "f1_weighted"
)

print("Cross-validation F1 scores:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std F1:", cv_scores.std())

# %%
model = grid.best_estimator_

# %% [markdown]
# #### Interpretation
# 
# K-fold cross-validation was used to evaluate model performance across multiple splits.  
# This method provides a more reliable estimate compared to a single validation split, as it reduces dependency on a specific partition of the dataset.

# %% [markdown]
# # Validation method 3 - nasted cross-validation

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
# Leave-One-Out cross-validation evaluates the model using nearly all available data for training in each iteration.  
# Although it provides an almost unbiased estimate, it is computationally expensive, so a subset of the data was used.
# 
# This method is mainly of theoretical interest rather than practical use for large datasets.

# %% [markdown]
# # Final model on held-out test set

# %%
model.fit(X_train_val, y_train_val)

y_test_pred = model.predict(X_test)

# %% [markdown]
# # Metrics + Confusion Matrix

# %%
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, pos_label="e"))
print("Recall:", recall_score(y_test, y_test_pred, pos_label="e"))
print("F1 Score:", f1_score(y_test, y_test_pred, pos_label="e"))
print("ROC AUC:", roc_auc_score(y_test, y_scores))


# %%
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# %%
y_scores = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label="e")
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
precision, recall, _ = precision_recall_curve(y_test, y_scores, pos_label="e")

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# %% [markdown]
# #### Interpretation
# 
# The final model was evaluated on a held-out test set.  
# The confusion matrix shows the distribution of correct and incorrect predictions.
# 
# Special attention should be given to false negatives (poisonous mushrooms classified as edible), as this type of error is particularly dangerous in real-world scenarios.
# 
# Therefore, recall for the poisonous class is a critical metric in this problem.

# %% [markdown]
# # Conclusion



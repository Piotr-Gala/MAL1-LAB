# %% [markdown]
# # Detecting Exoplanets 

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 2. Load data

# %%
exoplanet_df = pd.read_csv("exoplanet_dataset.csv")
print(exoplanet_df.shape)
exoplanet_df.head()


# %% [markdown]
# ## 3. Initial overview

# %%
exoplanet_df.info()

exoplanet_df.describe(include="all").T

# %%
# For an easier comprehension, we will rename the columns into their description.

exoplanet_df = exoplanet_df.rename(columns={'kepid':'KepID',
'kepoi_name':'KOIName',
'kepler_name':'KeplerName',
'koi_disposition':'ExoplanetArchiveDisposition',
'koi_pdisposition':'DispositionUsingKeplerData',
'koi_score':'DispositionScore',
'koi_fpflag_nt':'NotTransit-LikeFalsePositiveFlag',
'koi_fpflag_ss':'koi_fpflag_ss',
'koi_fpflag_co':'CentroidOffsetFalsePositiveFlag',
'koi_fpflag_ec':'EphemerisMatchIndicatesContaminationFalsePositiveFlag',
'koi_period':'OrbitalPeriod, days',
'koi_period_err1':'OrbitalPeriodUpperUnc, days',
'koi_period_err2':'OrbitalPeriodLowerUnc, days',
'koi_time0bk':'TransitEpoch, BKJD',
'koi_time0bk_err1':'TransitEpochUpperUnc, BKJD',
'koi_time0bk_err2':'TransitEpochLowerUnc, BKJD',
'koi_impact':'ImpactParamete',
'koi_impact_err1':'ImpactParameterUpperUnc',
'koi_impact_err2':'ImpactParameterLowerUnc',
'koi_duration':'TransitDuration, hrs',
'koi_duration_err1':'TransitDurationUpperUnc, hrs',
'koi_duration_err2':'TransitDurationLowerUnc, hrs',
'koi_depth':'TransitDepth, ppm',
'koi_insol':'InsolationFlux, Earthflux',
'koi_insol_err1':'InsolationFluxUpperUnc, Earthflux',
'koi_insol_err2':'InsolationFluxLowerUnc, Earthflux',
'koi_model_snr':'TransitSignal-to-Noise',
'koi_tce_plnt_num':'TCEPlanetNumber',
'koi_tce_delivname':'TCEDeliver',
'koi_steff':'StellarEffectiveTemperature, K',
'koi_steff_err1':'StellarEffectiveTemperatureUpperUnc, K',
'koi_steff_err2':'StellarEffectiveTemperatureLowerUnc, K',
'koi_depth_err1':'TransitDepthUpperUnc, ppm',
'koi_depth_err2':'TransitDepthLowerUnc, ppm',
'koi_prad':'PlanetaryRadius, Earthradii',
'koi_prad_err1':'PlanetaryRadiusUpperUnc, Earthradii',
'koi_prad_err2':'PlanetaryRadiusLowerUnc, Earthradii',
'koi_teq':'EquilibriumTemperature, K',
'koi_teq_err1':'EquilibriumTemperatureUpperUnc, K',
'koi_teq_err2':'EquilibriumTemperatureLowerUnc, K',
'koi_slogg':'StellarSurfaceGravity, log10(cm/s^2)',
'koi_slogg_err1':'StellarSurfaceGravityUpperUnc, log10(cm/s^2)',
'koi_slogg_err2':'StellarSurfaceGravityLowerUnc, log10(cm/s^2)',
'koi_srad':'StellarRadius, Solarradii',
'koi_srad_err1':'StellarRadiusUpperUnc, Solarradii',
'koi_srad_err2':'StellarRadiusLowerUnc, Solarradii',
'ra':'RA, decimaldegrees',
'dec':'Dec, decimaldegrees',
'koi_kepmag':'Kepler-band, mag'
})

# %% [markdown]
# ## 4. Missing values analysis
# 
# We first compute the percentage of missing values for each column.

# %%
missing_percentage = (exoplanet_df.isnull().mean() * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({"missing_percentage": missing_percentage})
missing_df

# %%
# columns with 100% missing values
cols_100_missing = ['EquilibriumTemperatureUpperUnc, K', 'EquilibriumTemperatureLowerUnc, K']

# irrelevant identifiers / name columns
cols_irrelevant = ['KepID', "KOIName", "KeplerName", "TCEDeliver"]

exoplanet_clean = exoplanet_df.drop(columns=cols_100_missing + cols_irrelevant)

print(exoplanet_clean.shape)
exoplanet_clean.head()

# %%
# encode disposition columns:

status_map = {
    "FALSE POSITIVE": 0,
    "CANDIDATE": 1,
    "CONFIRMED": 2
}

exoplanet_clean["KeplerDispositionStatus"] = exoplanet_clean["DispositionUsingKeplerData"].map(status_map)
exoplanet_clean["ArchiveDispositionStatus"] = exoplanet_clean["ExoplanetArchiveDisposition"].map(status_map)

exoplanet_clean = exoplanet_clean.drop(columns=["DispositionUsingKeplerData", "ExoplanetArchiveDisposition"])

exoplanet_clean[["KeplerDispositionStatus", "ArchiveDispositionStatus"]].head()

# %%
# target check:

print(exoplanet_clean["KeplerDispositionStatus"].value_counts(dropna=False))
print(exoplanet_clean["ArchiveDispositionStatus"].value_counts(dropna=False))

# %% [markdown]
# ## 6. Outlier analysis using IQR
# 
# First we count outliers and then decide if we want to remove them.

# %%
numeric_cols = exoplanet_clean.select_dtypes(include=[np.number]).columns.tolist()

outlier_summary = []
for col in numeric_cols:
    series = exoplanet_clean[col].dropna()
    if len(series) == 0:
        continue
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((series < lower) | (series > upper)).sum()
    outlier_summary.append({
        "feature": col,
        "outliers": outliers,
        "total": len(series),
        "percentage": (outliers / len(series)) * 100
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values(by="percentage", ascending=False)
outlier_df

# %% [markdown]
# #### Interpretations
# 
# The IQR-based outlier analysis identified many extreme values, especially in binary flag columns and several scientific measurement columns. For binary variables, these are not true outliers but a consequence of the method being less suitable for 0/1 features. For the continuous astronomical features, extreme values may reflect real observations rather than errors. Therefore, I decided not to remove outliers at this stage. Instead, I kept them and later considered transformations for strongly skewed variables.

# %%
print("Original shape:", exoplanet_clean.shape)
print("Shape after dropna:", exoplanet_clean.dropna().shape)
print("Rows removed:", exoplanet_clean.shape[0] - exoplanet_clean.dropna().shape[0])

# %%
# separate target-related columns before imputation
target_cols = ["KeplerDispositionStatus", "ArchiveDispositionStatus"]

feature_cols = [col for col in exoplanet_clean.columns if col not in target_cols]

# numeric feature columns
numeric_feature_cols = exoplanet_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

# impute numeric features with median
imputer = SimpleImputer(strategy="median")
exoplanet_clean[numeric_feature_cols] = imputer.fit_transform(exoplanet_clean[numeric_feature_cols])

# check missing values again
missing_after_imputation = exoplanet_clean.isnull().sum().sort_values(ascending=False)
print(missing_after_imputation.head(10))
print("Remaining missing values:", exoplanet_clean.isnull().sum().sum())

# %% [markdown]
# ## 11. Correlation analysis

# %%
# columns that will not be used as model features
exclude_cols = ["DispositionScore", "KeplerDispositionStatus", "ArchiveDispositionStatus"]

feature_corr_df = exoplanet_clean.drop(columns=exclude_cols, errors="ignore")

correlation_matrix = feature_corr_df.corr()

plt.figure(figsize=(18,12))
sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Matrix - Feature Candidates")
plt.show()

# %%
# Stronger correlations

corr_pairs = correlation_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]
corr_pairs = corr_pairs.sort_values(ascending=False)

# remove duplicate pairs
corr_pairs = corr_pairs.reset_index()
corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
corr_pairs["pair_key"] = corr_pairs.apply(
    lambda row: tuple(sorted([row["feature_1"], row["feature_2"]])), axis=1
)
corr_pairs = corr_pairs.drop_duplicates(subset="pair_key").drop(columns="pair_key")

# show strongest correlations

high_corr_pairs = corr_pairs[corr_pairs["correlation"] > 0.95]
high_corr_pairs.head(20)

# %%
# drop those with high correlation to avoid multicollinearity issues in modeling

cols_high_corr_to_drop = [
    "PlanetaryRadiusLowerUnc, Earthradii",
    "InsolationFluxLowerUnc, Earthflux"
]

exoplanet_model = exoplanet_clean.drop(columns=cols_high_corr_to_drop)

print(exoplanet_model.shape)

# %% [markdown]
# ### Prepare X & y

# %%
y = exoplanet_model["KeplerDispositionStatus"]

X = exoplanet_model.drop(columns=["DispositionScore", "KeplerDispositionStatus", "ArchiveDispositionStatus"], errors="ignore")

print("X shape:", X.shape)
print("y shape:", y.shape)
print(y.value_counts())

# %% [markdown]
# ## Train / validation / test split

# %%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation target distribution:")
print(y_val.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True))

# %% [markdown]
# ## Scaling

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

final_scaler = StandardScaler()
X_train_val_scaled = final_scaler.fit_transform(X_train_val)

# %% [markdown]
# ## Logistic Regression tuning

# %%
log_results = []

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    log_model = LogisticRegression(C=C, max_iter=5000, random_state=42)
    log_model.fit(X_train_scaled, y_train)

    y_train_pred = log_model.predict(X_train_scaled)
    y_val_pred = log_model.predict(X_val_scaled)

    log_results.append({
        "C": C,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "val_precision": precision_score(y_val, y_val_pred, average="weighted", zero_division=0),
        "val_recall": recall_score(y_val, y_val_pred, average="weighted", zero_division=0),
        "val_f1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
    })

    log_results_df = pd.DataFrame(log_results).sort_values(by="val_f1", ascending=False)
log_results_df

# %%
best_log_C = log_results_df.iloc[0]["C"]
print("Best Logistic Regression C:", best_log_C)

# %%
best_log_model = LogisticRegression(C=best_log_C, max_iter=5000, random_state=42)
best_log_model.fit(X_train_scaled, y_train)

y_train_pred_log = best_log_model.predict(X_train_scaled)
y_val_pred_log = best_log_model.predict(X_val_scaled)

print("Logistic Regression - Train metrics")
print("Accuracy:", accuracy_score(y_train, y_train_pred_log))
print("Precision:", precision_score(y_train, y_train_pred_log, zero_division=0))
print("Recall:", recall_score(y_train, y_train_pred_log, zero_division=0))
print("F1-score:", f1_score(y_train, y_train_pred_log, zero_division=0))

print("\nLogistic Regression - Validation metrics")
print("Accuracy:", accuracy_score(y_val, y_val_pred_log))
print("Precision:", precision_score(y_val, y_val_pred_log, zero_division=0))
print("Recall:", recall_score(y_val, y_val_pred_log, zero_division=0))
print("F1-score:", f1_score(y_val, y_val_pred_log, zero_division=0))

# %%
# Confusion matrix for validation set
cm_log = confusion_matrix(y_val, y_val_pred_log)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm_log,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)
plt.title("Logistic Regression - Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% [markdown]
# #### Interpretation
# 
# Logistic Regression achieved very high validation performance. However, some features in the dataset are false-positive flags that are likely strongly related to how the target label was assigned. Therefore, the results should be interpreted with caution, as the model may partially rely on proxy information rather than only on physical measurements.

# %% [markdown]
# ### SVM tuning

# %%
svm_results = []

for C in [0.01, 0.1, 1, 10]:
    for kernel in ["linear", "rbf"]:
        svm_model = SVC(C=C, kernel=kernel, random_state=42)
        svm_model.fit(X_train_scaled, y_train)

        y_train_pred = svm_model.predict(X_train_scaled)
        y_val_pred = svm_model.predict(X_val_scaled)

        svm_results.append({
            "C": C, 
            "kernel": kernel,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0)
        })

svm_results_df = pd.DataFrame(svm_results).sort_values(by="val_f1", ascending=False)
svm_results_df

# %%
best_svm_C = svm_results_df.iloc[0]["C"]
best_svm_kernel = svm_results_df.iloc[0]["kernel"]

print("Best SVM C:", best_svm_C)
print("Best SVM kernel:", best_svm_kernel)

# %%
best_svm_model = SVC(C=best_svm_C, kernel=best_svm_kernel, random_state=42)
best_svm_model.fit(X_train_scaled, y_train)

y_train_pred_svm = best_svm_model.predict(X_train_scaled)
y_val_pred_svm = best_svm_model.predict(X_val_scaled)

print("SVM - Train metrics")
print("Accuracy:", accuracy_score(y_train, y_train_pred_svm))
print("Precision:", precision_score(y_train, y_train_pred_svm, zero_division=0))
print("Recall:", recall_score(y_train, y_train_pred_svm, zero_division=0))
print("F1-Score:", f1_score(y_train, y_train_pred_svm, zero_division=0))

print("\nSVM - Validation metrics")
print("Accuracy:", accuracy_score(y_val, y_val_pred_svm))
print("Precision:", precision_score(y_val, y_val_pred_svm, zero_division=0))
print("Recall:", recall_score(y_val, y_val_pred_svm, zero_division=0))
print("F1-Score:", f1_score(y_val, y_val_pred_svm, zero_division=0))

# %%
cm_svm = confusion_matrix(y_val, y_val_pred_svm)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)
plt.title("SVM - Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% [markdown]
# ## Final model

# %%
# final scaling
final_scaler = StandardScaler()
X_train_val_scaled = final_scaler.fit_transform(X_train_val)
X_test_scaled = final_scaler.transform(X_test)

# final Logistic Regression
final_log_model = LogisticRegression(C=best_log_C, max_iter=5000, random_state=42)
final_log_model.fit(X_train_val_scaled, y_train_val)
y_test_pred_log = final_log_model.predict(X_test_scaled)

# final SVM
final_svm_model = SVC(C=best_svm_C, kernel=best_svm_kernel, random_state=42)
final_svm_model.fit(X_train_val_scaled, y_train_val)
y_test_pred_svm = final_svm_model.predict(X_test_scaled)

# %%
def evaluate_model(y_true, y_pred, model_name):
    print(model_name)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, zero_division=0))
    print()

evaluate_model(y_test, y_test_pred_log, "Logistic Regression - Test set")
evaluate_model(y_test, y_test_pred_svm, "SVM - Test set")

# %%
cm_log_test = confusion_matrix(y_test, y_test_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_log_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)
plt.title("Logistic Regression - Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm_svm_test = confusion_matrix(y_test, y_test_pred_svm)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_svm_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)
plt.title("SVM - Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% [markdown]
# ### Interpretations
# 
# Both models performed extremely well on the validation set, but their performance dropped notably on the test set. This suggests that the validation results were optimistic and that the test set provides a more realistic estimate of generalization performance. Logistic Regression achieved the best overall test performance, with slightly higher accuracy and F1-score than SVM. SVM achieved higher recall and fewer false negatives, but this came at the cost of more false positives and lower precision. Therefore, Logistic Regression was selected as the final model because it provided the best overall balance between precision and recall, while also being simpler and easier to interpret. However, if the main goal were to minimize missed exoplanet candidates, SVM could also be justified due to its higher recall.

# %% [markdown]
# ### Limitations
# 
# One limitation of this analysis is that some preprocessing steps, such as imputation and correlation-based feature filtering, were performed before the final train-validation-test split. In addition, some flag-based features may be strongly related to the target labeling process itself. Therefore, the reported validation performance may have been overly optimistic, and the test set results should be considered the more reliable estimate.



# Assignment 4: Detecting Exoplanets - Oral Exam Notes

## Opening Speech

```text
Good morning. In this assignment I worked with the Kepler exoplanet dataset.
The goal was to build a supervised machine learning pipeline that classifies Kepler objects as planet candidates or false positives.

I started with exploratory data analysis: I checked the dataset shape, column types, missing values, target distribution, outliers and correlations.
Then I cleaned the data by removing completely empty columns and identifier columns, encoded the target labels, and prepared the feature matrix.

For modelling, I split the data into training, validation and test sets using stratification.
After the split, I applied median imputation and standard scaling, fitting both steps only on the training data to avoid data leakage.

Finally, I trained and compared Logistic Regression and SVM models.
Logistic Regression performed slightly better on the final test set, with about 90% accuracy and F1-score.
I also treated the result carefully, because some false-positive flag features may act as proxy information for the target label.
```

---

## Cells 1-3: Imports

**Co powiedziec**

```text
At the beginning I import the libraries needed for the whole machine learning workflow.
Pandas and NumPy are used for data handling, Matplotlib and Seaborn for visualization, and Scikit-learn for preprocessing, modelling and evaluation.
```

**Po ludzku**

To tylko przygotowanie narzedzi. Jeszcze nic sie nie dzieje z danymi.

**Najwazniejsze**

- `pandas` - praca z tabela danych.
- `numpy` - operacje numeryczne.
- `matplotlib`, `seaborn` - wykresy.
- `train_test_split` - podzial danych.
- `SimpleImputer` - uzupelnianie brakow.
- `StandardScaler` - skalowanie cech.
- `LogisticRegression`, `SVC` - modele.
- metrics - ocena modelu.

**Pytanie**

```text
Why do you use pandas?
```

**Odpowiedz**

```text
I use pandas because the dataset is tabular, so a DataFrame is convenient for loading, inspecting and transforming the data.
```

**Pytanie**

```text
Why do you use seaborn?
```

**Odpowiedz**

```text
I use seaborn to create clearer statistical visualizations, for example the correlation heatmap.
```

**Pytanie**

```text
Why do you import evaluation metrics?
```

**Odpowiedz**

```text
Because after training the models I need to measure their performance using accuracy, precision, recall, F1-score and confusion matrices.
```

**Pytanie**

```text
Why do you use StandardScaler?
```

**Odpowiedz**

```text
I use StandardScaler because Logistic Regression and SVM are sensitive to feature scale. Scaling makes features comparable.
```

---

## Cells 4-5: Loading The Dataset

**Co powiedziec**

```text
In this step I load the exoplanet dataset from a CSV file into a pandas DataFrame.
Then I check the shape of the dataset and display the first few rows to make sure the data was loaded correctly.
```

**Konkret z notebooka**

```text
9564 rows and 49 columns
```

**Po ludzku**

`shape` mowi, ile jest wierszy i kolumn.  
`head()` pokazuje pierwsze rekordy.

**Pytanie**

```text
Why do you check the shape of the dataset?
```

**Odpowiedz**

```text
I check the shape to know how many observations and features are available in the dataset.
```

**Pytanie**

```text
Why do you use head()?
```

**Odpowiedz**

```text
I use head() to quickly inspect the first rows and verify that the dataset was loaded correctly.
```

**Pytanie**

```text
What does one row represent?
```

**Odpowiedz**

```text
One row represents one Kepler object of interest, with its measured astronomical properties and disposition labels.
```

---

## Cells 6-8: Initial Overview And Renaming Columns

**Co powiedziec**

```text
After loading the data, I perform an initial overview.
I use info() to check column types and missing values, and describe() to see basic statistics.
Then I rename the columns to make them easier to understand.
```

**Po ludzku**

Sprawdzasz, co jest w tabeli: typy danych, braki, podstawowe statystyki.

**Konkret**

- dataset ma 49 kolumn,
- sa kolumny numeryczne i tekstowe,
- niektore kolumny maja braki,
- `koi_teq_err1` i `koi_teq_err2` sa calkowicie puste.

**Pytanie**

```text
What is the purpose of info()?
```

**Odpowiedz**

```text
info() shows the data types, number of non-null values and memory usage, so it helps identify missing values and categorical columns.
```

**Pytanie**

```text
What is the purpose of describe()?
```

**Odpowiedz**

```text
describe() gives summary statistics, such as mean, standard deviation, minimum, maximum and quartiles.
```

**Pytanie**

```text
Why did you rename the columns?
```

**Odpowiedz**

```text
I renamed the columns to make them more descriptive and easier to interpret during analysis.
```

**Pytanie**

```text
Does renaming columns affect the model?
```

**Odpowiedz**

```text
No, renaming only changes column labels. It does not change the data values.
```

**Uwaga**

W notebooku jest literowka `ImpactParamete`. To tylko blad w nazwie kolumny, nie zmienia danych ani modelu.

---

## Cells 9-13: Missing Values And Encoding Target

**Co powiedziec**

```text
I calculated missing value percentages for all columns.
Then I removed two columns with 100% missing values and several identifier columns.
After that, I encoded the disposition labels numerically. 
The final target based on Kepler data is almost balanced, with false positives and candidates in similar numbers.
```

**Po ludzku**

Najpierw sprawdzasz braki, potem usuwasz kolumny bez sensu, potem zamieniasz tekstowe etykiety na liczby.

**Usuniete kolumny**

- `EquilibriumTemperatureUpperUnc, K`
- `EquilibriumTemperatureLowerUnc, K`
- `KepID`
- `KOIName`
- `KeplerName`
- `TCEDeliver`

**Encoding**

```text
FALSE POSITIVE -> 0
CANDIDATE -> 1
CONFIRMED -> 2
```

**Konkret targetu**

```text
KeplerDispositionStatus:
0 -> 4847
1 -> 4717
```

To jest prawie zbalansowany binary classification problem.

**Pytanie**

```text
Why analyze missing values first?
```

**Odpowiedz**

```text
Because missing values affect preprocessing decisions. Some columns may need to be removed, while others can be imputed.
```

**Pytanie**

```text
Why remove columns with 100% missing values?
```

**Odpowiedz**

```text
Because they contain no information and cannot help the model.
```

**Pytanie**

```text
Why remove ID columns?
```

**Odpowiedz**

```text
Because identifiers do not describe physical properties and may cause the model to memorize objects instead of learning patterns.
```

**Pytanie**

```text
Why encode labels?
```

**Odpowiedz**

```text
Because machine learning models need numerical labels instead of text categories.
```

**Pytanie**

```text
Is the target balanced?
```

**Odpowiedz**

```text
Yes, the Kepler disposition target is almost balanced: about 4847 false positives and 4717 candidates.
```

---

## Cells 14-18: Outliers And Missing Values Before Split

**Co powiedziec**

```text
I analyzed outliers using the IQR method, but I decided not to remove them because extreme astronomical values may be valid observations.
I also checked that dropping all rows with missing values would remove 1761 rows, so imputation is a better choice.
However, I postponed imputation until after the train-validation-test split to avoid data leakage.
```

**IQR**

```text
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower bound = Q1 - 1.5 * IQR
upper bound = Q3 + 1.5 * IQR
```

**Po ludzku**

Outliery zostaja, bo w astronomii ekstremalne wartosci moga byc prawdziwe, a nie bledne.

**Dropna check**

```text
Original shape: (9564, 43)
Shape after dropna: (7803, 43)
Rows removed: 1761
```

**Pytanie**

```text
What is an outlier?
```

**Odpowiedz**

```text
An outlier is a value that lies far away from the typical range of a feature.
```

**Pytanie**

```text
How does the IQR method work?
```

**Odpowiedz**
```text
It uses the first and third quartiles. Values below Q1 minus 1.5 IQR or above Q3 plus 1.5 IQR are treated as outliers.
```

**Pytanie**

```text
Why is IQR not ideal for binary flags?
```

**Odpowiedz**

```text
Because binary flags only have values 0 and 1, so the IQR method may mark valid flag values as outliers.
```

**Pytanie**

```text
Why did you not remove outliers?
```

**Odpowiedz**

```text
Because extreme values in astronomical data may be valid physical observations, not errors. Removing them could remove useful information.
```

**Pytanie**

```text
Why not use dropna()?
```

**Odpowiedz**

```text
Because it would remove 1761 rows, which is a significant amount of data.
```

**Pytanie**

```text
What is data leakage?
```

**Odpowiedz**

```text
Data leakage happens when information from validation or test data is used during training or preprocessing.
```

---

## Cells 19-28: Features, Split, Train-Only Correlation, Imputation And Scaling

**Co powiedziec**

```text
I prepared X and y by selecting KeplerDispositionStatus as the target and removing target-related columns from the features.
Then I split the data into train, validation and test sets using stratification.
After the split, I analyzed correlations only on the training features and removed two highly correlated columns from all splits.
Finally, I applied median imputation and standard scaling, fitting both preprocessing steps only on the training data to avoid data leakage.
```

**Po ludzku**

Najpierw robisz `X` i `y`, potem dzielisz dane, potem patrzysz na korelacje tylko na `X_train`. Dopiero po tym usuwasz wybrane kolumny ze wszystkich splitow i robisz imputacje oraz scaling.

**X i y**

```text
X = features
y = target
```

**Target**

```text
KeplerDispositionStatus
```

**Usuniete z X**

- `DispositionScore`
- `KeplerDispositionStatus`
- `ArchiveDispositionStatus`

**Split**

```text
X_train: 6120 rows
X_val: 1531 rows
X_test: 1913 rows
```

**Po co stratify**

Zachowuje podobny rozklad klas w train, validation i test.

**Correlation analysis**

Korelacja jest liczona tylko na `X_train`, nie na calym datasecie. To jest wazne, bo feature selection nie powinno korzystac z informacji z validation ani test setu.

**Usuniete przez korelacje**

- `PlanetaryRadiusLowerUnc, Earthradii`
- `InsolationFluxLowerUnc, Earthflux`

**Teoria korelacji**

Korelacja Pearsona:

```text
+1 = strong positive linear relation
 0 = no linear relation
-1 = strong negative linear relation
```

**Imputation**

Braki sa uzupelniane mediana. Mediana jest lepsza od sredniej przy outlierach, bo jest odporniejsza na ekstremalne wartosci.

**Scaling**

```text
x_scaled = (x - mean) / standard_deviation
```

Scaling jest potrzebny, bo Logistic Regression i SVM sa wrazliwe na skale cech.

**Pytanie**

```text
What are X and y?
```

**Odpowiedz**

```text
X is the feature matrix and y is the target variable.
```

**Pytanie**

```text
Why remove ArchiveDispositionStatus?
```

**Odpowiedz**

```text
Because it is another disposition label and could leak information about the target.
```

**Pytanie**

```text
Why remove DispositionScore?
```

**Odpowiedz**

```text
It is likely too directly related to the disposition label and could leak target information.
```

**Pytanie**

```text
Why use train, validation and test sets?
```

**Odpowiedz**

```text
The training set is used to fit the model, the validation set to tune hyperparameters, and the test set for final unbiased evaluation.
```

**Pytanie**

```text
Why use stratified split?
```

**Odpowiedz**

```text
To keep the class proportions similar in all subsets.
```

**Pytanie**

```text
What is correlation?
```

**Odpowiedz**

```text
Correlation measures the strength and direction of a linear relationship between two variables.
```

**Pytanie**

```text
Why did you use absolute correlation?
```

**Odpowiedz**

```text
Because both strong positive and strong negative correlations can indicate redundant information.
```

**Pytanie**

```text
Why remove highly correlated features?
```

**Odpowiedz**

```text
To reduce redundancy and multicollinearity, especially for Logistic Regression.
```

**Pytanie**

```text
What is multicollinearity?
```

**Odpowiedz**

```text
Multicollinearity occurs when two or more features are strongly correlated and carry similar information.
```

**Pytanie**

```text
Was this step leakage-safe?
```

**Odpowiedz**

```text
Yes. Correlation-based feature selection is fitted on the training set only and the same selected columns are removed from validation and test sets.
```

**Pytanie**

```text
Why use median imputation?
```

**Odpowiedz**

```text
Median imputation is robust to outliers and allows us to keep rows with missing values.
```

**Pytanie**

```text
Why fit imputer only on training data?
```

**Odpowiedz**

```text
To avoid using information from validation or test data during preprocessing.
```

**Pytanie**

```text
What does StandardScaler do?
```

**Odpowiedz**

```text
It subtracts the mean and divides by the standard deviation for each feature.
```

---

## Metrics: FP, FN, Precision, Recall, F1

**Klasy**

```text
0 = FALSE POSITIVE
1 = CANDIDATE
```

**Confusion matrix**

```text
                 Predicted 0        Predicted 1
Actual 0         TN                 FP
Actual 1         FN                 TP
```

**TP**

```text
Actual: CANDIDATE
Predicted: CANDIDATE
```

**TN**

```text
Actual: FALSE POSITIVE
Predicted: FALSE POSITIVE
```

**FP**

```text
Actual: FALSE POSITIVE
Predicted: CANDIDATE
```

Falsy alarm.

**FN**

```text
Actual: CANDIDATE
Predicted: FALSE POSITIVE
```

Model przegapil potencjalna planete.

**Accuracy**

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Overall correctness.

**Precision**

```text
precision = TP / (TP + FP)
```

Jak model mowi "candidate", to jak czesto ma racje?

**Recall**

```text
recall = TP / (TP + FN)
```

Ile prawdziwych candidates model znalazl?

**F1**

```text
F1 = 2 * precision * recall / (precision + recall)
```

Balans precision i recall.

**Do powiedzenia**

```text
In this task, precision tells me how reliable the predicted planet candidates are.
Recall tells me how many actual candidates the model managed to detect.
F1-score balances these two aspects.
A false positive means that a false-positive object was predicted as a candidate, while a false negative means that a real candidate was missed.
```

**Intuicja**

```text
Precision asks: when the model says "candidate", can I trust it?
Recall asks: did the model find most real candidates?
Accuracy asks: how often is the model correct overall?
F1 asks: is there a good balance between precision and recall?
```

**Mini przyklad**

```text
TP = 90
TN = 80
FP = 10
FN = 20

accuracy = (90 + 80) / 200 = 0.85
precision = 90 / (90 + 10) = 0.90
recall = 90 / (90 + 20) = 0.818
F1 is about 0.857
```

**Najwazniejsze**

```text
Precision cares about FP.
Recall cares about FN.
F1 balances both.
```

---

## Cells 29-34: Logistic Regression

**Co powiedziec**

```text
I trained Logistic Regression with several C values and selected the one with the best validation F1-score.
The best C was 0.1.
The model achieved very high validation performance, with F1 around 0.994.
I also checked the confusion matrix to see the types of errors.
However, I interpret the result carefully because some false-positive flags may be strongly related to the target.
```

**Po ludzku**

Regresja logistyczna to model klasyfikacyjny, mimo nazwy "regression".

**C**

```text
small C = stronger regularization
large C = weaker regularization
```

**Wynik**

```text
Best Logistic Regression C: 0.1
Validation F1: about 0.994
```

**Pytanie**

```text
Why is Logistic Regression used for classification?
```

**Odpowiedz**

```text
Because it estimates class probabilities and then assigns observations to classes based on a decision boundary.
```

**Pytanie**

```text
What does C control?
```

**Odpowiedz**

```text
C controls regularization strength. Smaller C means stronger regularization.
```

**Pytanie**

```text
Why tune C?
```

**Odpowiedz**

```text
Because different regularization strengths can affect generalization performance.
```

**Pytanie**

```text
Why use F1-score?
```

**Odpowiedz**

```text
F1-score balances precision and recall, so it is useful when we care about both false positives and false negatives.
```

**Pytanie**

```text
What is a confusion matrix?
```

**Odpowiedz**

```text
A confusion matrix shows how many samples were classified correctly and incorrectly for each class.
```

**Pytanie**

```text
What is overfitting?
```

**Odpowiedz**

```text
Overfitting happens when a model learns the training data too closely and performs poorly on unseen data.
```

**Pytanie**

```text
Why should the high validation score be interpreted carefully?
```

**Odpowiedz**

```text
Because some false-positive flags may act as proxy information for the target label.
```

**Wazna uwaga**

False-positive flags moga byc proxy targetu, czyli model moze miec bardzo mocna podpowiedz.

---

## Cells 35-39: SVM

**Co powiedziec**

```text
I trained and tuned an SVM model using different C values and two kernels: linear and RBF.
The best model used a linear kernel with C equal to 0.01.
It achieved a validation F1-score around 0.993, which is very high but slightly lower than Logistic Regression.
I also used a confusion matrix to inspect the classification errors.
```

**Po ludzku**

SVM szuka granicy decyzyjnej, ktora oddziela klasy z jak najwiekszym marginesem.

**Kernel**

- `linear` - liniowa granica decyzyjna.
- `rbf` - nieliniowa, bardziej elastyczna granica.

**Wynik**

```text
Best SVM C: 0.01
Best SVM kernel: linear
Validation F1: about 0.993
```

**Pytanie**

```text
What is SVM?
```

**Odpowiedz**

```text
SVM is a classification algorithm that tries to find a decision boundary with the maximum margin between classes.
```

**Pytanie**

```text
What is a margin?
```

**Odpowiedz**

```text
The margin is the distance between the decision boundary and the closest training points.
```

**Pytanie**

```text
What are support vectors?
```

**Odpowiedz**

```text
Support vectors are the closest points to the decision boundary, and they determine the position of that boundary.
```

**Pytanie**

```text
What is the difference between linear and RBF kernel?
```

**Odpowiedz**

```text
A linear kernel creates a linear decision boundary, while an RBF kernel can model nonlinear relationships.
```

**Pytanie**

```text
Why did the linear kernel perform best?
```

**Odpowiedz**

```text
It suggests that the classes may be separated well using a relatively linear boundary in the scaled feature space.
```

**Pytanie**

```text
What does C do in SVM?
```

**Odpowiedz**

```text
C controls the trade-off between a wider margin and fewer training errors. Smaller C means stronger regularization.
```

---

## Cells 40-45: Final Model, Test Evaluation, Limitations

**Co powiedziec**

```text
After tuning, I retrained both models on the combined training and validation data and evaluated them on the untouched test set.
Logistic Regression achieved about 90.2% accuracy and F1-score, while SVM achieved about 89.9%.
I selected Logistic Regression because it was slightly better and easier to explain.
However, I mention one important limitation: some false-positive flags may act as proxy variables for the target.
```

**Po ludzku**

Po wyborze hiperparametrow trenujesz finalnie na `train + validation`, a test zostaje tylko do koncowej oceny.

**Finalne wyniki**

```text
Logistic Regression:
Accuracy: 0.902
F1-score: 0.902

SVM:
Accuracy: 0.899
F1-score: 0.898
```

**Wybrany model**

```text
Logistic Regression
```

Bo jest minimalnie lepszy na tescie i prostszy do wyjasnienia.

**Pytanie**

```text
Why retrain on train plus validation?
```

**Odpowiedz**

```text
Because after hyperparameters are selected, using more training data can improve the final model.
```

**Pytanie**

```text
Why is the test set used only at the end?
```

**Odpowiedz**

```text
Because it should represent unseen data and provide an unbiased estimate of final performance.
```

**Pytanie**

```text
Which model was selected and why?
```

**Odpowiedz**

```text
I selected Logistic Regression because it achieved slightly better test performance and is simpler to interpret.
```

**Pytanie**

```text
Why did validation performance differ from test performance?
```

**Odpowiedz**

```text
The validation result was probably optimistic. The test set gives a more realistic estimate of generalization.
```

**Pytanie**

```text
Is 90% accuracy good?
```

**Odpowiedz**

```text
Yes, it is a strong result, but it should be interpreted carefully because of possible proxy features and dataset-specific limitations.
```

**Main limitation**

```text
Some false-positive flags may act as proxy information for the target.
```

**Finalna obrona**

```text
The model performs well, but I treat the result carefully because the dataset contains features that may be strongly related to the labelling process itself.
```

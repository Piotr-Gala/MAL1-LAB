# Assignment Intros For Oral Exam

## Assignment 1: Car Prices

```text
Good afternoon. In this assignment I worked with a dataset of electric vehicle prices.
The main target variable is Price in Danish Kroner, so the first part is a supervised regression problem.

I started by loading the data, checking missing values and separating the input features from the target.
Then I split the data into training and test sets before fitting models or scalers, to avoid data leakage.

First, I implemented linear regression manually using NumPy and linear algebra.
Then I used Scikit-learn LinearRegression and evaluated the model with MSE, RMSE and R2.

After that, I standardized the data and compared Ridge, Lasso and Elastic Net regularization.
Finally, I changed the problem into binary classification by using the median price to label cars as Cheap or Expensive, and I trained a kNN classifier.
```

## Assignment 2: Candidates I

```text
Good afternoon. In this assignment I worked with Danish candidate test data from the 2022 election.
The goal was to predict party affiliation from candidates' political answer patterns, so this is supervised multiclass classification.

I started by loading and inspecting the data, cleaning simple issues like invalid age values, and doing descriptive analysis.
I also used box plots to compare age and answer confidence across parties.

For modelling, I used the 49 candidate-test answer columns as features and party affiliation as the target.
I removed the independent candidate class because it had only three observations, which was too small for stable classification.

Then I used a stratified train-test split and compared Decision Tree, Random Forest and Gradient-Boosted Trees using cross-validation on the training data.
Random Forest performed best, but I interpret the result carefully because party labels are not a perfect measure of political ideology.
```

## Assignment 3: Mushroom Foraging

```text
Good afternoon. In this assignment I worked with a mushroom dataset.
The goal was to classify mushrooms as edible or poisonous, so this is supervised binary classification.

I started with exploratory analysis: checking data types, missing values, target distribution and duplicate rows.
Then I split the data into training, validation and test sets using stratification before fitting any preprocessing.

For preprocessing, I used imputation and scaling for numerical features, and imputation plus one-hot encoding for categorical features.
These steps were placed in a pipeline together with Logistic Regression, so preprocessing is fitted only on training data.

I compared validation methods: a single validation split, stratified cross-validation and nested cross-validation.
The final model performed reasonably well, but I focus on recall for the poisonous class because false negatives are dangerous in this problem.
```

## Assignment 4: Detecting Exoplanets

```text
Good afternoon. In this assignment I worked with the Kepler exoplanet dataset.
The goal was to classify objects as planet candidates or false positives, so this is supervised binary classification.

I started by inspecting the data, checking missing values, outliers, target distribution and correlations.
Then I removed empty columns, identifier columns and target-related columns from the feature matrix.

This was important because some columns are very close to the target labelling process.
Removing them makes the experiment more conservative and reduces target leakage risk.

After splitting the data into train, validation and test sets, I applied median imputation and standard scaling fitted only on the training data.
Finally, I compared Logistic Regression and SVM models.
The SVM with an RBF kernel performed best on the final test set.
```

## Assignment 5: Sentiment Analysis

```text
Good afternoon. In this assignment I worked with IMDb movie reviews.
The goal was to classify each review as positive or negative, so this is supervised binary text classification.

Because the input data is text, I first converted reviews into numerical features using Bag-of-Words with CountVectorizer.
The most important workflow rule is that I split the raw text before fitting the vectorizer.
This avoids data leakage because the vocabulary is learned only from training data.

Then I trained a simple neural network with one hidden layer using Keras.
The output layer uses sigmoid activation because this is binary classification, and the loss function is binary crossentropy.

I used a validation set to choose hyperparameters and kept the test set untouched until final evaluation.
The final model reached about 87.9 percent test accuracy, but Bag-of-Words is limited because it ignores word order, context, negation and sarcasm.
```

## Assignment 6: Candidates II

```text
Good afternoon. In this assignment I worked again with Danish candidate test data from the 2022 election.
This time the goal was exploratory analysis using unsupervised learning, so there is no target variable.

The data contains answers to political questions on a scale from -2 to 2.
I started by loading the candidate datasets and question metadata, cleaning simple metadata issues, and mapping technical question IDs back to readable question text.

Then I standardized the 49 response features and used PCA to reduce them to two main dimensions.
I interpreted the PCA axes using loadings, meaning the questions that contribute most strongly to each principal component.

After that, I compared clustering methods: K-Means, hierarchical clustering and DBSCAN.
The main result is that candidates form broad ideological blocs rather than one clean cluster per party.
Finally, I analyzed elected candidates by placing them in the same PCA space and measuring agreement using distances.
```

## Universal Backup Sentence

```text
The key thing in this notebook is that I understand what is fitted on training data, what is evaluated later, and why this avoids leakage or overfitting.
```

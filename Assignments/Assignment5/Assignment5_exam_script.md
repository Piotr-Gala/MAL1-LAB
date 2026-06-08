# Assignment 5 exam script - Sentiment Analysis

## 0. How to use this file

This is a speaking script for the oral exam.

Use it in this order:

1. Start with the short overview.
2. Go through the notebook cell by cell.
3. For each part, explain: what I do, why I do it, and what the result means.
4. At the end, be ready for the likely questions.

Do not read it word-for-word like a robot. Use it as a checklist.

---

## 1. Opening - what this assignment is about

Say this first:

> This assignment is about sentiment analysis on IMDb movie reviews.  
> The task is binary classification: given a text review, the model predicts whether the sentiment is positive or negative.  
> Since machine learning models cannot directly use raw text, I first convert reviews into numerical features using Bag of Words.  
> Then I train a neural network classifier, tune its hyperparameters using a validation set, and finally evaluate it on a separate test set.

In simple words:

- input: movie review text,
- output: positive or negative sentiment,
- type of task: supervised binary classification,
- model: neural network, specifically `MLPClassifier`,
- text encoding: Bag of Words with `CountVectorizer`.

---

## 2. Loading the data

Notebook part:

```python
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)
Y = (labels=='positive').astype(np.int_)
```

Say:

> I load the reviews and labels from text files.  
> The labels are originally strings: positive or negative.  
> I convert them into numbers, where positive becomes 1 and negative becomes 0.  
> This is needed because the classifier works with numerical target values.

If asked why:

> The expression `labels == 'positive'` gives boolean values, and `astype(np.int_)` converts them to 0 and 1.

Key terms:

- `1` means positive,
- `0` means negative,
- this is supervised learning because labels are known.

---

## 3. Train / validation / test split

Notebook part:

```python
X_train_val_text, X_test_text, y_train_val, y_test = train_test_split(...)
X_train_text, X_val_text, y_train, y_val = train_test_split(...)
```

Results:

```text
Train set shape: (16000, 10000)
Validation set shape: (4000, 10000)
Test set shape: (5000, 10000)
Vocabulary size: 10000
```

Say:

> I split the data into training, validation and test sets.  
> The training set is used to fit the model.  
> The validation set is used to compare hyperparameter settings.  
> The test set is kept untouched until the final evaluation, so it gives a more honest estimate of generalization.

Important:

> I use `stratify`, so the class balance between positive and negative reviews is preserved in each split.

Why this matters:

> Without stratification, one split could accidentally contain more positive or more negative reviews, which could make evaluation less reliable.

If asked why not tune on the test set:

> If I used the test set to choose hyperparameters, then the test set would no longer be independent. It would leak into the model selection process.

---

## 4. Bag of Words with CountVectorizer

Notebook part:

```python
vectorizer = CountVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)
```

Say:

> Bag of Words converts each review into a vector of word counts.  
> Each column corresponds to one word from the vocabulary.  
> The value in that column says how many times the word appears in the review.  
> The order of words is ignored.

About `max_features=10000`:

> I keep only the 10,000 most frequent words.  
> This reduces dimensionality, makes training faster, and removes some rare noisy words.

Trade-off:

- advantage: faster and simpler model,
- disadvantage: rare but useful words can be lost,
- disadvantage: word order and context are ignored.

Very important:

> I use `fit_transform` only on the training data.  
> For validation and test data, I only use `transform`.  
> This prevents data leakage, because the vocabulary is learned only from the training data.

---

## 5. Exploring the representation

Notebook part:

```python
word = 'movie'
word_index = vectorizer.vocabulary_[word]
```

Result:

```text
The word 'movie' is represented by feature index 5848.
Count of 'movie' in the first training review: 5
A whole review is a sparse 10,000-dimensional vector with 253 non-zero entries.
```

Say:

> A single word is represented as one feature index in the vocabulary.  
> For example, the word `movie` has one column in the Bag of Words matrix.  
> A full review is represented as a 10,000-dimensional vector, where each value is a word count.

Explain sparse vector:

> The vector is sparse because most reviews only contain a small part of the full vocabulary.  
> So most values are zero.

If asked what the model loses:

> Bag of Words loses word order and grammar.  
> For example, negation can be difficult: `good` and `not good` may look too similar if the model only counts words.

---

## 6. Neural network model

Notebook part:

```python
model = MLPClassifier(
    solver='adam',
    hidden_layer_sizes=settings['hidden_layer_sizes'],
    activation=settings['activation'],
    alpha=settings['alpha'],
    batch_size=settings['batch_size'],
    learning_rate_init=settings['learning_rate_init'],
    max_iter=20,
    random_state=0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=3
)
```

Say:

> I use `MLPClassifier`, which is a feed-forward neural network.  
> The input layer has 10,000 features, one for each Bag of Words feature.  
> The assignment asks for a single hidden layer, so I use settings like `(64,)`, which means one hidden layer with 64 neurons.  
> The output is a binary sentiment prediction.

Important:

- `(32,)` means one hidden layer with 32 neurons,
- `(64,)` means one hidden layer with 64 neurons,
- `(128,)` means one hidden layer with 128 neurons,
- `(64, 32)` would mean two hidden layers, but that is not what this assignment asks for.

---

## 7. Activation functions

Tested activations:

```python
'relu'
'tanh'
```

Say:

> Activation functions introduce non-linearity into the neural network.  
> Without activation functions, the network would behave like a linear model, even with multiple layers.

About ReLU:

> ReLU returns zero for negative inputs and keeps positive inputs.  
> It is commonly used because it is simple and often trains efficiently.

About tanh:

> Tanh maps values to the range from -1 to 1.  
> In my validation results, tanh with 64 hidden neurons performed best.

---

## 8. Hyperparameter tuning

Notebook part:

```python
candidate_settings = [
    {'hidden_layer_sizes': (32,), 'activation': 'relu', ...},
    {'hidden_layer_sizes': (64,), 'activation': 'relu', ...},
    {'hidden_layer_sizes': (64,), 'activation': 'tanh', ...},
    ...
]
```

Say:

> I compare several hyperparameter settings and choose the one with the best validation accuracy.  
> I tune the number of hidden neurons, the activation function and the regularization strength `alpha`.  
> I keep the test set separate and do not use it for choosing hyperparameters.

Results:

```text
Best validation accuracy: 0.8972
Best settings: hidden_layer_sizes=(64,), activation='tanh', alpha=0.0001
```

Say:

> The best validation result was about 89.7% accuracy, using one hidden layer with 64 neurons and tanh activation.

About `alpha`:

> In `MLPClassifier`, `alpha` controls L2 regularization.  
> Regularization penalizes large weights and helps reduce overfitting.

About `learning_rate_init`:

> The learning rate controls how large the weight updates are during training.  
> If it is too high, training can become unstable.  
> If it is too low, training can be very slow.

About `batch_size`:

> The batch size controls how many training examples are used before updating the model weights.

---

## 9. Overfitting and early stopping

Observed result:

```text
train_accuracy around 0.95
validation_accuracy around 0.89
```

Say:

> The training accuracy is higher than validation accuracy, so there is some overfitting.  
> But the gap is not extreme.  
> I use early stopping and L2 regularization to reduce overfitting.

About early stopping:

> Early stopping stops training when the validation score stops improving.  
> This prevents the model from continuing to memorize the training data.

In this notebook:

```python
early_stopping=True
validation_fraction=0.1
n_iter_no_change=3
```

Say:

> The model internally keeps part of the training data for early stopping.  
> If the score does not improve for 3 iterations, training stops.

---

## 10. Final model and test evaluation

Notebook part:

```python
vectorizer_final = CountVectorizer(max_features=10000)
X_train_val = vectorizer_final.fit_transform(X_train_val_text)
X_test = vectorizer_final.transform(X_test_text)
```

Say:

> After choosing the best hyperparameters, I retrain the model on the combined training and validation data.  
> This gives the final model more data to learn from.  
> Then I evaluate it once on the test set.

Result:

```text
Test accuracy: 0.8830
Confusion matrix:
[[2236  264]
 [ 321 2179]]
```

Say:

> The final test accuracy is 88.3%.  
> This means the model correctly classifies about 88 out of 100 reviews.

Explain confusion matrix:

```text
[[2236  264]
 [ 321 2179]]
```

Say:

> The first row is negative reviews.  
> 2236 negative reviews were correctly classified as negative.  
> 264 negative reviews were incorrectly classified as positive.  
> The second row is positive reviews.  
> 321 positive reviews were incorrectly classified as negative.  
> 2179 positive reviews were correctly classified as positive.

Terms:

- true negatives: 2236,
- false positives: 264,
- false negatives: 321,
- true positives: 2179.

About metrics:

> Precision, recall and F1-score are all around 0.88, so the model performs similarly on both classes.

Why accuracy is acceptable here:

> Accuracy is useful here because the test set is balanced: 2500 negative and 2500 positive reviews.  
> If the classes were imbalanced, I would focus more on precision, recall and F1-score.

---

## 11. Custom sentences

Notebook part:

```python
X_custom = vectorizer_final.transform(my_sentences)
custom_predictions = final_model.predict(X_custom)
custom_probabilities = final_model.predict_proba(X_custom)[:, 1]
```

Say:

> Finally, I test the classifier on sentences I wrote myself.  
> I transform them using the same final vectorizer and then use the trained model to predict sentiment.  
> I also print the probability of positive sentiment.

Important:

> I do not fit a new vectorizer on these sentences.  
> I must use the same vectorizer that was fitted on the training data, otherwise the feature columns would not match the model.

---

## 12. Short final summary

Say this at the end:

> To summarize, this assignment uses Bag of Words to convert text reviews into numerical vectors, then trains a neural network for binary sentiment classification.  
> I split the data into train, validation and test sets to avoid overfitting and to evaluate generalization fairly.  
> The best validation model used one hidden layer with 64 neurons and tanh activation.  
> The final model achieved 88.3% accuracy on the test set, with balanced precision and recall for both positive and negative reviews.

---

## 13. Likely exam questions and answers

### What is Bag of Words?

> Bag of Words is a text representation where each document is represented as word counts.  
> Each feature corresponds to one word, and the value is how often that word appears.

### What does Bag of Words ignore?

> It ignores word order, grammar and deeper context.  
> This can make phrases like `not good` difficult, because the model mostly sees separate word counts.

### Why use `max_features=10000`?

> To limit the vocabulary to the 10,000 most frequent words.  
> This reduces dimensionality and makes training faster, but may remove rare useful words.

### Why use train, validation and test sets?

> Train is for fitting the model.  
> Validation is for choosing hyperparameters.  
> Test is for final independent evaluation.

### Why not use the test set for tuning?

> Because then the test set would influence model selection.  
> It would no longer be a fair estimate of generalization.

### What is data leakage?

> Data leakage happens when information from validation or test data influences training.  
> For example, fitting the vectorizer before splitting the data would leak vocabulary information from the test set.

### What is a neural network?

> A neural network is a model made of layers of neurons.  
> Each neuron computes a weighted sum of inputs, adds a bias, applies an activation function, and passes the result forward.

### What does one hidden layer mean?

> It means there is one layer between the input features and the output prediction.  
> In this notebook, `(64,)` means one hidden layer with 64 neurons.

### Why do we need activation functions?

> Activation functions add non-linearity.  
> Without them, the network would only behave like a linear model.

### What is ReLU?

> ReLU outputs zero for negative inputs and keeps positive inputs.  
> It is simple and often works well in neural networks.

### What is tanh?

> Tanh maps values to the range from -1 to 1.  
> In this assignment, tanh gave the best validation accuracy.

### What is overfitting?

> Overfitting happens when the model performs very well on training data but worse on unseen data.  
> It means the model learned patterns too specific to the training set.

### How do you reduce overfitting here?

> I use validation-based hyperparameter tuning, early stopping and L2 regularization through `alpha`.

### What does `alpha` do?

> `alpha` controls L2 regularization in `MLPClassifier`.  
> A higher alpha penalizes large weights more strongly.

### What is early stopping?

> Early stopping stops training when validation performance stops improving.  
> It helps prevent the model from memorizing the training data.

### What is the confusion matrix result?

> The model correctly classified 2236 negative reviews and 2179 positive reviews.  
> It misclassified 264 negative reviews as positive and 321 positive reviews as negative.

### Is accuracy enough?

> In this case, accuracy is acceptable because the test set is balanced.  
> But I still check precision, recall and F1-score to make sure both classes perform similarly.

### What are the weaknesses of this solution?

> Bag of Words ignores word order and context.  
> The model may struggle with sarcasm, negation and complex language.  
> More advanced approaches could use TF-IDF, word embeddings or transformer-based models, but that would be beyond this assignment.

---

## 14. Emergency short version

If you have little time, say only this:

> This assignment is binary sentiment classification on IMDb reviews.  
> I convert text to numerical features using Bag of Words with the 10,000 most frequent words.  
> I split the data into train, validation and test sets.  
> I train a one-hidden-layer neural network using `MLPClassifier`.  
> I tune hyperparameters like hidden layer size, activation function and regularization using validation accuracy.  
> The best model uses 64 hidden neurons and tanh activation.  
> Finally, I retrain on train plus validation data and test once on the test set.  
> The final test accuracy is 88.3%, and precision/recall are balanced around 0.88.


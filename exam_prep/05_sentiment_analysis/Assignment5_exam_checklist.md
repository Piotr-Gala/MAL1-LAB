# Assignment 5: Sentiment Analysis - Exam Checklist Map

This file maps the official exam overview topics to the Assignment 5 notebook.

Sources:

- `Materials/exam/Exam_information_and_assignments_overview.pdf`
- `Materials/exam/Exam_theory_topics.pdf`
- `Materials/presentations/09_Neural_Networks_filled_in.pdf`
- Notebook: `MAL1-LAB/Answers/Assignment5/Assignment5.ipynb`
- Notes: `MAL1-LAB/Answers/Assignment5/Assignment5_notebook_explanation.md`
- Speech notes: `MAL1-LAB/Answers/Assignment5/Assignment5_speech_notes.md`

---

## Official Assignment-Specific Checklist From Exam Overview

Official topic for Assignment 5: **Sentiment Analysis**

- Sentiment analysis
- Bag of words encoding
- Neural networks
- Hyperparameters for NN
- Activation functions

General supporting topics from the theory checklist:

- Supervised learning
- Binary classification
- Train / validation / test split
- Data leakage
- Bag-of-Words for string data
- Loss function
- Overfitting and regularization
- Accuracy, precision, recall, F1-score
- Confusion matrix

---

## Official Topic To Notebook Map

| Official exam topic | Where in notebook | How it is covered |
| --- | --- | --- |
| Sentiment analysis | Cells 3-7, 29-32 | IMDb reviews are classified as positive or negative sentiment. |
| Bag of words encoding | Cells 10-13 | `CountVectorizer(max_features=10000)` converts reviews into word-count vectors. |
| Neural networks | Cells 16-18, 22-28 | Keras `Sequential` feed-forward neural networks are trained. |
| Hyperparameters for NN | Cells 22-24 | Hidden units, activation, L2 regularization and optimizer are compared. |
| Activation functions | Cells 16-18, 22-24 | ReLU and tanh are tested in hidden layers; sigmoid is used in the output layer. |
| Train / validation / test split | Cells 8-9 | Raw text is split before fitting the vectorizer. |
| Data leakage | Cells 8-11, 25-29 | Vectorizer is fitted only on training data, test set is used only at the end. |
| Classification metrics | Cells 29-30 | Accuracy, precision, recall, F1-score and confusion matrix are reported. |
| Overfitting / regularization | Cells 19-24, 33 | Learning curves, early stopping and L2 candidates are discussed. |

---

## 1. Sentiment Analysis

**Status:** covered

**Where in notebook**

- Cells 3-7: load reviews and sentiment labels
- Cells 29-32: final predictions and custom sentence classification

**What happens**

The notebook predicts whether a movie review is negative or positive.

The target is encoded as:

```text
0 = negative
1 = positive
```

**Why it matters**

This is a supervised binary classification problem. The model learns from reviews with known labels and predicts sentiment for unseen reviews.

**What to say**

```text
This assignment is sentiment analysis on IMDb movie reviews.
The model receives review text and predicts whether the sentiment is positive or negative.
```

**Possible question**

```text
Why is this supervised learning?
```

**Answer**

```text
Because the training data contains known labels: each review is already marked as positive or negative.
```

---

## 2. Bag-of-Words Encoding

**Status:** covered

**Where in notebook**

- Cells 10-13

**What happens**

The notebook uses:

```python
CountVectorizer(max_features=10000)
```

This creates a 10,000-column vocabulary of frequent words.

Each review becomes a vector where:

- each column is one word,
- each value is the count of that word in the review.

**Why it matters**

Machine learning models cannot use raw text directly. Bag-of-Words turns text into numerical features.

**What to say**

```text
Bag-of-Words represents text as word counts.
It ignores word order and only keeps information about which words appear and how often.
```

**Possible question**

```text
What does max_features=10000 do?
```

**Answer**

```text
It keeps only the 10000 most frequent words.
This reduces dimensionality and training time, but rare useful words may be removed.
```

**Possible question**

```text
What is the weakness of Bag-of-Words?
```

**Answer**

```text
It ignores word order, grammar and context.
This can make negation and sarcasm difficult.
```

---

## 3. Split Before Preprocessing

**Status:** covered

**Where in notebook**

- Cells 8-11

**What happens**

The notebook first splits raw text into train, validation and test sets.

Only after the split does it fit the vectorizer:

```text
fit_transform on training text
transform on validation text
transform on test text
```

**Why it matters**

This prevents data leakage. The vocabulary is learned without seeing validation or test reviews.

**What to say**

```text
I split before preprocessing.
The vectorizer is fitted only on training data, so validation and test information does not leak into the training process.
```

**Possible question**

```text
What would be leakage here?
```

**Answer**

```text
If I fitted CountVectorizer on all reviews before the split, the vocabulary would be influenced by validation and test text.
That would leak information from unseen data.
```

---

## 4. Train / Validation / Test Roles

**Status:** covered

**Where in notebook**

- Cells 8-9
- Cells 22-24
- Cells 25-30

**What happens**

The split sizes are:

```text
Train:      16000
Validation: 4000
Test:        5000
```

The roles are:

- train: fit preprocessing and model,
- validation: choose hyperparameters,
- test: final evaluation.

**Why it matters**

The test set must not influence model selection. Otherwise, the test score becomes biased.

**What to say**

```text
Validation is used for choosing hyperparameters.
The test set is used only once at the end to estimate generalization.
```

**Possible question**

```text
Why not tune on the test set?
```

**Answer**

```text
Because then the test set would influence model selection and would no longer be an independent evaluation set.
```

---

## 5. Neural Network Architecture

**Status:** covered

**Where in notebook**

- Cells 16-18: baseline model
- Cells 22-28: tuned and final models

**What happens**

The model is a feed-forward neural network with one hidden layer.

Baseline architecture:

```text
Input: 10000 Bag-of-Words features
Hidden: 32 neurons, ReLU
Output: 1 neuron, sigmoid
```

**Why it matters**

The assignment asks for a neural network classifier. A one-hidden-layer feed-forward network is simple and defensible.

**What to say**

```text
The input layer receives 10000 word-count features.
The hidden layer learns combinations of words.
The output layer has one sigmoid neuron because this is binary classification.
```

**Possible question**

```text
What does one hidden layer mean?
```

**Answer**

```text
It means there is one layer of neurons between the input features and the output prediction.
```

---

## 6. Activation Functions

**Status:** covered

**Where in notebook**

- Cells 16-18
- Cells 22-24

**What happens**

The notebook uses:

- ReLU in hidden layers,
- tanh as an alternative hidden activation during tuning,
- sigmoid in the output layer.

**Why it matters**

Activation functions add non-linearity. Without them, the network would behave like a linear model.

**What to say**

```text
ReLU and tanh are hidden-layer activation functions.
They let the network learn nonlinear patterns.
Sigmoid is used in the output layer because it gives a probability for the positive class.
```

**Possible question**

```text
Why not use softmax here?
```

**Answer**

```text
Softmax is common for multiclass classification.
Here we have binary classification, so one sigmoid output neuron is enough.
```

---

## 7. Loss Function And Optimizer

**Status:** covered

**Where in notebook**

- Cells 16-18
- Cells 22-28

**What happens**

The models use:

- `binary_crossentropy` loss,
- `SGD` for the baseline,
- `Adam` for tuned candidate models.

**Why it matters**

Binary crossentropy is appropriate for 0/1 targets and sigmoid output.

The optimizer controls how weights are updated during training.

**What to say**

```text
The model uses binary crossentropy because the target is binary.
The optimizer updates the weights to reduce the loss.
SGD is simple, while Adam often converges faster because it adapts the learning rate.
```

**Possible question**

```text
What is an epoch?
```

**Answer**

```text
One epoch means the model has gone through the full training set once.
```

---

## 8. Hyperparameters For Neural Networks

**Status:** covered

**Where in notebook**

- Cells 22-24

**What happens**

The notebook compares:

```text
32 units, ReLU, no L2, Adam
64 units, ReLU, L2 = 0.001, Adam
64 units, tanh, L2 = 0.001, Adam
```

Best validation result:

```text
hidden_units: 32
activation: relu
l2_strength: 0.0
optimizer: adam
validation accuracy: 0.8907
```

**Why it matters**

Hyperparameters are not learned directly by the model. They are chosen by comparing validation performance.

**What to say**

```text
I tune only a small number of relevant hyperparameters.
The best model is selected using validation accuracy, not test accuracy.
```

**Possible question**

```text
Why not test many more combinations?
```

**Answer**

```text
More combinations could improve the result, but this assignment needs a simple and explainable workflow.
Testing a few relevant settings is enough to show hyperparameter tuning.
```

---

## 9. Overfitting, Regularization And Early Stopping

**Status:** covered

**Where in notebook**

- Cells 19-24
- Cell 33

**What happens**

The notebook checks training and validation performance.

Training accuracy is higher than validation accuracy, so there is some overfitting.

The notebook uses:

- validation monitoring,
- early stopping,
- L2 regularization candidates.

**Why it matters**

Neural networks can memorize training data. Regularization and early stopping help improve generalization.

**What to say**

```text
There is some overfitting because training accuracy is higher than validation accuracy.
I use early stopping and test L2 regularization to reduce this risk.
```

**Possible question**

```text
What does early stopping do?
```

**Answer**

```text
It stops training when validation loss stops improving.
This prevents the model from continuing to fit the training data too closely.
```

**Possible question**

```text
What does L2 regularization do?
```

**Answer**

```text
It penalizes large weights.
This can make the model less complex and reduce overfitting.
```

---

## 10. Metrics And Confusion Matrix

**Status:** covered

**Where in notebook**

- Cells 29-30

**What happens**

The final test result is:

```text
Test accuracy: 0.8788
```

The notebook reports:

- precision,
- recall,
- F1-score,
- confusion matrix.

**Why it matters**

Accuracy alone can hide class-specific problems. Precision, recall and F1-score show whether both classes perform similarly.

In this dataset, the test set is balanced:

```text
2500 negative
2500 positive
```

So accuracy is acceptable, but the other metrics are still useful.

**What to say**

```text
The final model reaches about 87.9% test accuracy.
Precision, recall and F1-score are around 0.88 for both classes, so the model performs similarly on negative and positive reviews.
```

**Possible question**

```text
When is accuracy misleading?
```

**Answer**

```text
Accuracy can be misleading when classes are imbalanced.
For example, if 95% of samples are negative, a model can get high accuracy by predicting negative almost always.
```

---

## 11. Custom Sentences

**Status:** covered

**Where in notebook**

- Cells 31-32

**What happens**

The notebook predicts sentiment for manually written sentences.

It uses:

```text
vectorizer_final.transform(my_sentences)
```

**Why it matters**

The same vectorizer must be used so the feature columns match the model input.

**What to say**

```text
For new sentences, I transform them with the same final vectorizer.
I do not fit a new vectorizer because that would create a different vocabulary mapping.
```

---

## 12. Limitations

**Status:** covered

**Where in notebook**

- Cell 33

**Main limitations**

- Bag-of-Words ignores word order.
- It ignores context and grammar.
- Negation and sarcasm can be difficult.
- Dense conversion uses more memory.
- The model is simpler than modern NLP models.

**What to say**

```text
The main limitation is the text representation.
Bag-of-Words is simple and useful, but it loses word order and deeper meaning.
```

**Possible question**

```text
How could you improve this solution?
```

**Answer**

```text
I could use TF-IDF, word embeddings, recurrent neural networks or transformers.
But for this assignment, Bag-of-Words and a simple neural network are enough and easier to explain.
```

---

## Quick Oral Checklist

- Start by saying this is supervised binary sentiment classification.
- Mention input is raw text and target is positive/negative.
- Say labels are encoded as `0` and `1`.
- Mention the dataset is balanced: 12500 negative and 12500 positive.
- Say the split happens before fitting `CountVectorizer`.
- Explain `fit_transform` on train and `transform` on validation/test.
- Define Bag-of-Words as word-count vectors.
- Mention `max_features=10000`.
- Explain sparse vector: most values are zero.
- Say Bag-of-Words ignores word order and context.
- Explain Keras input needs dense `float32` arrays here.
- Describe architecture: 10000 inputs, one hidden layer, one sigmoid output.
- Explain sigmoid for binary probability.
- Explain binary crossentropy for 0/1 target.
- Mention validation tuning, not test tuning.
- State best validation model: 32 ReLU, Adam, validation accuracy about 0.8907.
- Mention early stopping and overfitting.
- Explain final retraining on train + validation.
- State final test accuracy: about 0.8788.
- Mention precision, recall and F1-score around 0.88.
- Explain confusion matrix mistakes: false positives and false negatives.
- For custom sentences, say same final vectorizer is reused.
- End with limitations: negation, sarcasm, context, dense memory use.

---

## One-Minute Summary

```text
This assignment is binary sentiment classification on IMDb reviews.
I convert positive and negative labels into 1 and 0, then split the raw text into train, validation and test sets.
After the split, I use CountVectorizer with 10000 features to create a Bag-of-Words representation.
The vectorizer is fitted only on training text to avoid data leakage.

Then I train a one-hidden-layer Keras neural network.
The output uses sigmoid and binary crossentropy because this is binary classification.
I tune hidden units, activation, L2 regularization and optimizer using validation accuracy.
The best validation model uses 32 ReLU units and Adam.

Finally, I retrain on train plus validation data and evaluate once on the untouched test set.
The final test accuracy is about 87.9%, and precision, recall and F1-score are balanced around 0.88.
The main limitation is that Bag-of-Words ignores word order and context.
```

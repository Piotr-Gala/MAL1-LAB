# Assignment 5: Sentiment Analysis - Notebook Explanation

## Short Overview

This assignment is about sentiment analysis on IMDb movie reviews.

The task is supervised binary classification:

- input: movie review text,
- target: negative or positive sentiment,
- model: one-hidden-layer neural network,
- text representation: Bag-of-Words with `CountVectorizer`.

Because the input is raw text, the main workflow is:

```text
raw reviews -> train/validation/test split -> Bag-of-Words -> neural network -> validation tuning -> final test
```

The most important defensive point is that the split happens before fitting the text vectorizer. This prevents data leakage.

---

## Cells 1-2: Imports

**What happens**

The notebook imports libraries for:

- data handling: `numpy`, `pandas`,
- plotting: `matplotlib`, `seaborn`,
- text preprocessing: `CountVectorizer`,
- splitting and metrics: `train_test_split`, `classification_report`, `confusion_matrix`,
- neural networks: TensorFlow/Keras.

**Why it matters**

The assignment is a full ML workflow: load text data, preprocess it, train a neural network, tune hyperparameters and evaluate the result.

**Key detail**

The notebook uses Keras `Sequential`, which matches the neural network examples from the course materials.

---

## Cells 3-4: Load Data

**What happens**

The notebook loads:

- `reviews.txt` into a `review` column,
- `labels.txt` into a `label` column.

Then it creates a binary target:

```text
positive -> 1
negative -> 0
```

**Why it matters**

Neural networks need numerical target values. The original labels are strings, so they must be converted to numbers.

**What to say**

```text
I load the raw reviews and labels, then convert the sentiment labels into 0 and 1.
This makes the task a supervised binary classification problem.
```

---

## Cells 5-7: Initial Data Check

**What happens**

The notebook checks:

- class counts,
- class proportions,
- example negative and positive reviews.

The dataset has:

```text
12500 negative reviews
12500 positive reviews
```

So the classes are perfectly balanced.

**Why it matters**

Class balance affects metric interpretation. Because the classes are balanced, accuracy is meaningful, but precision, recall and F1-score are still checked later.

**What to say**

```text
I check the target distribution before modelling.
The dataset is balanced, so accuracy is a reasonable metric, but I still inspect precision, recall and F1-score.
```

---

## Cells 8-9: Train / Validation / Test Split

**What happens**

The raw text and labels are split into:

```text
Train:      16000 reviews
Validation: 4000 reviews
Test:        5000 reviews
```

The notebook uses `stratify=y`, so every split keeps the same positive/negative balance.

**Purpose of each split**

- train: fit preprocessing and train model,
- validation: choose hyperparameters,
- test: final independent evaluation.

**Why it matters**

The test set must stay untouched until the end. If the test set is used for tuning, the final score becomes too optimistic.

**What to say**

```text
I split the raw text before any vectorizer is fitted.
Training data is used to fit the model, validation data is used for model selection, and test data is saved for final evaluation.
```

---

## Cells 10-11: Bag-of-Words Representation

**What happens**

The notebook uses:

```python
CountVectorizer(max_features=10000)
```

This converts reviews into word-count vectors.

The vectorizer is fitted only on the training text:

```text
fit_transform on train
transform on validation
transform on test
```

The resulting shapes are:

```text
Train:      (16000, 10000)
Validation:  (4000, 10000)
Test:        (5000, 10000)
```

**Why it matters**

Each review becomes a 10,000-dimensional vector. Each column represents one word from the vocabulary, and each value is the count of that word in the review.

**Leakage point**

The vocabulary is learned only from the training data. Validation and test data are only transformed using that vocabulary.

**What to say**

```text
Bag-of-Words turns every review into word counts.
I fit the vectorizer only on training text and only transform validation and test text.
This prevents data leakage from the validation or test sets.
```

---

## Cells 12-13: Explore The Representation

**What happens**

The notebook checks how a word and a full review are represented.

Example:

- the word `movie` has one feature index,
- the first training review has 252 non-zero word-count entries,
- the full vector still has 10,000 dimensions.

**Why it matters**

Bag-of-Words vectors are sparse: most values are zero because one review only contains a small part of the full vocabulary.

**Limitation**

Bag-of-Words ignores word order and context.

**What to say**

```text
A single word corresponds to one column in the vocabulary.
A full review is a sparse vector of word counts.
The limitation is that word order is lost, so phrases like not good can be difficult.
```

---

## Cells 14-15: Prepare Data For Keras

**What happens**

The sparse Bag-of-Words matrices are converted into dense `float32` arrays.

The labels are also converted to `float32` arrays.

**Why it matters**

This simple Keras feed-forward model expects dense numeric arrays.

**Trade-off**

Dense arrays are simpler for Keras, but they use more memory than sparse matrices.

**What to say**

```text
CountVectorizer gives sparse matrices.
For this Keras model I convert them to dense float32 arrays.
The trade-off is higher memory usage, but the code stays simple and close to the course examples.
```

---

## Cells 16-18: Baseline Neural Network

**What happens**

The baseline model is a one-hidden-layer feed-forward neural network:

```text
Input: 10000 features
Hidden layer: 32 neurons, ReLU
Output layer: 1 neuron, sigmoid
```

The model uses:

- optimizer: `SGD`,
- loss: `binary_crossentropy`,
- metric: `accuracy`.

Baseline result:

```text
Train accuracy:      0.8645
Validation accuracy: 0.8485
```

**Why sigmoid**

Sigmoid returns a value between 0 and 1, which can be interpreted as the probability of positive sentiment.

**Why binary crossentropy**

The target has two classes encoded as 0 and 1.

**What to say**

```text
The baseline is a simple feed-forward neural network with one hidden layer.
The output layer uses sigmoid because this is binary classification.
The loss is binary crossentropy because the target is 0 or 1.
```

---

## Cells 19-21: Learning Curves And Baseline Evaluation

**What happens**

The notebook plots training and validation accuracy/loss across epochs.

Then it evaluates the baseline on train and validation sets.

**Why it matters**

Learning curves help detect underfitting or overfitting.

**Interpretation**

Training accuracy is higher than validation accuracy, but the gap is not huge. This suggests some overfitting, but not extreme.

**What to say**

```text
I compare training and validation curves to see whether the model generalizes.
The training score is higher than validation score, so there is some overfitting.
```

---

## Cells 22-24: Hyperparameter Tuning

**What happens**

The notebook tests three candidate neural networks:

```text
32 units, ReLU, no L2, Adam
64 units, ReLU, L2 = 0.001, Adam
64 units, tanh, L2 = 0.001, Adam
```

The tuned hyperparameters are:

- number of hidden units,
- activation function,
- L2 regularization strength,
- optimizer.

The best validation model is:

```text
hidden_units: 32
activation: relu
l2_strength: 0.0
optimizer: adam
validation accuracy: 0.8907
```

**Why it matters**

Hyperparameters are selected using validation accuracy, not test accuracy.

**Early stopping**

During tuning, the notebook uses `EarlyStopping`:

```text
monitor: val_loss
patience: 2
restore_best_weights: True
```

This stops training when validation loss stops improving and helps reduce overfitting.

**What to say**

```text
I tune only a few relevant hyperparameters and choose the best model using validation accuracy.
The test set is not used during tuning.
Early stopping prevents the model from training too long after validation loss stops improving.
```

---

## Cells 25-28: Final Model

**What happens**

After choosing hyperparameters, the notebook:

1. refits a final `CountVectorizer` on train + validation text,
2. transforms the test text,
3. builds the final Keras model with the best settings,
4. trains it on train + validation data.

The final model is trained for:

```text
3 epochs
```

**Why it matters**

Validation data has already served its purpose for model selection. After that, train + validation can be combined to train the final model on more data.

The test set is still only transformed and evaluated at the end.

**What to say**

```text
After model selection, I retrain the final model on train plus validation data.
This gives the final model more data while keeping the test set untouched.
```

---

## Cells 29-30: Final Test Evaluation

**What happens**

The final model is evaluated on the test set.

Results:

```text
Test loss:     0.3518
Test accuracy: 0.8788
```

The classification report shows precision, recall and F1-score around `0.88` for both classes.

**Why it matters**

The test set gives the final estimate of generalization to unseen reviews.

**Confusion matrix**

The confusion matrix shows the types of mistakes:

- true negatives: negative reviews predicted as negative,
- false positives: negative reviews predicted as positive,
- false negatives: positive reviews predicted as negative,
- true positives: positive reviews predicted as positive.

**What to say**

```text
The final test accuracy is about 87.9%.
Because the test set is balanced, accuracy is meaningful.
I also check precision, recall and F1-score to make sure both classes perform similarly.
```

---

## Cells 31-32: Custom Sentences

**What happens**

The notebook classifies a few manually written sentences.

The same final vectorizer is used:

```text
vectorizer_final.transform(my_sentences)
```

The model outputs probabilities and class predictions.

**Why it matters**

The model expects the same vocabulary columns as during training. A new vectorizer would create a different feature mapping.

**What to say**

```text
For new sentences I use the same final vectorizer.
I do not fit a new vectorizer, because the model expects the same feature columns as during training.
```

---

## Cell 33: Limitations

**Main limitations**

- Bag-of-Words ignores word order.
- It ignores context and grammar.
- It can struggle with negation and sarcasm.
- Dense conversion uses more memory.
- The model is simple compared to modern NLP methods.

**Possible improvements**

- TF-IDF,
- word embeddings,
- recurrent neural networks,
- transformers.

**Why not use them here**

The assignment is about Bag-of-Words and neural networks. A simple model is easier to explain and defend.

**What to say**

```text
The main weakness is that Bag-of-Words ignores word order and context.
More advanced methods could improve performance, but this solution stays close to the course requirements.
```

---

## Final Defense

```text
This notebook follows a leakage-safe supervised learning workflow.
The raw text is split before preprocessing, the vectorizer is fitted only on training data, validation is used for model selection, and the test set is used only once at the end.
The final model is simple, course-aligned and defensible.
```

# Assignment 5: Sentiment Analysis - Cell Speech

## Opening Speech

```text
Good morning. In this assignment I worked with IMDb movie reviews.
The goal was to classify each review as positive or negative, so this is a supervised binary classification task.

Because the input is text, I first convert the reviews into numerical features using Bag-of-Words with CountVectorizer.
Then I train a one-hidden-layer neural network using Keras.

The most important part of the workflow is that I split the raw text before fitting the vectorizer.
This avoids data leakage, because the vocabulary is learned only from training data.

I use a validation set to choose hyperparameters and keep the test set untouched until the final evaluation.
The final model achieves about 87.9% test accuracy, with balanced precision and recall for both classes.
```

## Cells 1-2: Imports

```text
At the beginning I import the libraries for data handling, plotting, text vectorization, model evaluation and neural networks.
Pandas is used for loading the text files, CountVectorizer for Bag-of-Words, and TensorFlow/Keras for the neural network.
```

## Cells 3-4: Load Data

```text
Here I load the reviews and labels from text files.
The original labels are positive and negative strings.
I convert them into binary values: positive becomes 1 and negative becomes 0.
This is needed because the neural network works with numerical target values.
```

## Cells 5-7: Initial Data Check

```text
Before modelling I check the class distribution and inspect example reviews.
The dataset has 12500 negative and 12500 positive reviews, so it is perfectly balanced.
This means accuracy is a reasonable metric, but I still check precision, recall and F1-score later.
```

## Cells 8-9: Train / Validation / Test Split

```text
I split the raw text into training, validation and test sets.
Training data is used to fit the vectorizer and train the model.
Validation data is used to choose hyperparameters.
The test set is kept untouched until the final evaluation.

I also use stratification, so each split keeps the same positive and negative class balance.
```

## Cells 10-11: Bag-of-Words

```text
I use CountVectorizer with max_features equal to 10000.
This creates a Bag-of-Words representation where each column is one word and each value is the count of that word in the review.

I fit the vectorizer only on the training text.
Then I only transform the validation and test text.
This is important because fitting the vectorizer before the split would leak information from validation or test data.
```

## Cells 12-13: Representation Example

```text
Here I show how the representation works.
A single word is represented by one feature index in the vocabulary.
A whole review becomes a 10000-dimensional sparse vector of word counts.

Most values are zero, because one review only uses a small part of the vocabulary.
The limitation is that Bag-of-Words ignores word order and context.
```

## Cells 14-15: Prepare Data For Keras

```text
CountVectorizer returns sparse matrices.
For this simple Keras model, I convert them to dense float32 arrays.
The trade-off is higher memory usage, but it keeps the neural network input simple and close to the course examples.
```

## Cells 16-18: Baseline Neural Network

```text
The baseline model is a feed-forward neural network with one hidden layer.
The input layer has 10000 features from Bag-of-Words.
The hidden layer has 32 neurons with ReLU activation.
The output layer has one neuron with sigmoid activation.

Sigmoid is used because this is binary classification.
The loss function is binary crossentropy because the target is encoded as 0 or 1.
```

## Cells 19-21: Learning Curves And Baseline Score

```text
I plot training and validation accuracy and loss to see how the model learns.
The baseline reaches about 86.5% training accuracy and 84.9% validation accuracy.
The training score is higher than validation score, so there is some overfitting, but it is not extreme.
```

## Cells 22-24: Hyperparameter Tuning

```text
I tune a small set of hyperparameters: hidden units, activation function, L2 regularization and optimizer.
I compare the candidates using validation accuracy, not test accuracy.

The best validation model uses 32 hidden units, ReLU activation, no L2 regularization and Adam optimizer.
It reaches about 89.1% validation accuracy.

I also use early stopping during tuning.
It monitors validation loss and restores the best weights, which helps reduce overfitting.
```

## Cells 25-28: Final Model

```text
After choosing hyperparameters, I train the final model on train plus validation data.
I also refit the final vectorizer on train plus validation text.
This is acceptable because validation has already been used for model selection.

The test set is still not used for fitting anything.
It is only transformed and evaluated at the end.
```

## Cells 29-30: Test Evaluation And Confusion Matrix

```text
The final model gets about 87.9% test accuracy.
Precision, recall and F1-score are all around 0.88 for both classes.

The confusion matrix shows which errors the model makes:
negative reviews predicted as positive are false positives,
and positive reviews predicted as negative are false negatives.
```

## Cells 31-32: Custom Sentences

```text
Finally, I test the classifier on my own sentences.
I transform them using the same final vectorizer and then predict sentiment with the trained model.
I do not fit a new vectorizer because the model expects the same vocabulary columns as during training.
```

## Cell 33: Limitations

```text
The main limitation is that Bag-of-Words ignores word order and context.
This means the model may struggle with negation, sarcasm and phrases where meaning depends on word sequence.

More advanced methods could use TF-IDF, embeddings or transformers, but this assignment stays with a simple course-level neural network workflow.
```

## Final Sentence

```text
Overall, this solution is a leakage-safe text classification pipeline: split first, fit the vectorizer only on training data, tune on validation data, and evaluate once on the untouched test set.
```

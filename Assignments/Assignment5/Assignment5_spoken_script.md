# Assignment 5 spoken script - short exam version

Target: about 6-8 minutes.

---

## Intro

This assignment is about sentiment analysis on IMDb reviews.

It is a binary classification task: the model predicts whether a review is positive or negative.

Because the input is text, I first convert it into numerical features using Bag of Words, and then I train a neural network classifier.

---

## (a) Split and Bag of Words

I split the data into train, validation and test sets.

Train is used to fit the model, validation is used for hyperparameter tuning, and test is only used at the end for final evaluation.

I use stratification, so the positive and negative labels stay balanced in each split.

Then I use `CountVectorizer(max_features=10000)`.

This creates a Bag of Words representation: each review becomes a vector of word counts.

Each column is one word, and each value says how often that word appears in the review.

I fit the vectorizer only on the training data to avoid data leakage.

---

## (b) Representation

A single word is represented by one feature index in the vocabulary.

A whole review is represented as a 10,000-dimensional vector.

Most values are zero, because one review only contains a small part of the vocabulary, so the matrix is sparse.

The limitation is that Bag of Words ignores word order and context, so phrases like `not good` can be harder to understand.

---

## (c) Neural network and tuning

I use `MLPClassifier`, which is a feed-forward neural network.

The input layer has 10,000 features from Bag of Words.

The assignment asks for one hidden layer, so `(64,)` means one hidden layer with 64 neurons.

I test different hidden layer sizes, activation functions and regularization values.

The activation function adds non-linearity. I test ReLU and tanh.

`alpha` is L2 regularization, which helps reduce overfitting.

I choose the best model using validation accuracy, not test accuracy.

The best validation result is about 89.7%, with 64 hidden neurons and tanh activation.

---

## (d) Test evaluation

After choosing the best hyperparameters, I retrain the final model on train plus validation data.

Then I test it once on the test set.

The final test accuracy is 88.3%.

The confusion matrix shows:

- 2236 negative reviews correctly classified,
- 264 negative reviews classified as positive,
- 321 positive reviews classified as negative,
- 2179 positive reviews correctly classified.

Precision, recall and F1-score are all around 0.88, so the model performs similarly on both classes.

Accuracy is okay here because the test set is balanced.

---

## (e) Custom sentences

Finally, I test the model on my own sentences.

I transform them with the same final vectorizer, because the model expects the same features as during training.

Then the model predicts the sentiment and positive probability.

This is a simple sanity check that the model behaves reasonably on new text.

---

## Closing

So overall, the pipeline is: text reviews, Bag of Words encoding, one-hidden-layer neural network, validation-based tuning, and final test evaluation.

The main weakness is that Bag of Words ignores word order and deeper meaning, but for this assignment it still gives a solid result.


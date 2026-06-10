# Assignment 5: Sentiment Analysis - Exam Q/A

Use this file as an oral-exam cheat sheet: question, theory answer, and how it appears in the notebook.

Sources: `Exam_information_and_assignments_overview.pdf`, `Exam_theory_topics.pdf`, `Assignment5.ipynb`, existing assignment notes.

## 0. Opening Answer

**Question:** What is this assignment about?

**Theory answer:** This is supervised binary classification on text. The model learns to classify movie reviews as negative or positive sentiment.

**How used in the assignment:** The notebook loads IMDb-style reviews and labels, converts text to numerical bag-of-words vectors, trains neural networks, tunes hyperparameters, and evaluates classification performance.

## 1. Sentiment Analysis

**Question:** What is sentiment analysis?

**Theory answer:** Sentiment analysis is an NLP task where text is classified by opinion or emotion, commonly positive vs negative.

**How used in the assignment:** Each review is an input text, and the label tells whether the sentiment is negative or positive.

## 2. Handling String Data

**Question:** Why can a neural network not directly use raw text strings?

**Theory answer:** ML models operate on numerical tensors. Text must be converted into numerical features before modelling.

**How used in the assignment:** The notebook uses `CountVectorizer` to convert reviews into vectors of word counts.

## 3. Bag Of Words

**Question:** What is bag-of-words encoding?

**Theory answer:** Bag-of-words represents text by counting word occurrences. It ignores word order and grammar, but gives a simple numerical representation of documents.

**How used in the assignment:** `CountVectorizer(max_features=10000)` creates a fixed vocabulary and converts each review into a vector of word counts.

## 4. Vocabulary Limit

**Question:** Why limit `max_features`?

**Theory answer:** Text can contain many unique words. Limiting vocabulary reduces dimensionality, memory use, noise, and overfitting risk.

**How used in the assignment:** The notebook keeps the 10,000 most frequent features, which is a practical trade-off between information and complexity.

## 5. Split Before Vectorizer

**Question:** Why split before fitting `CountVectorizer`?

**Theory answer:** The vocabulary is learned from data. If the vectorizer sees test data, information from test text leaks into training.

**How used in the assignment:** The notebook splits raw reviews first, then fits the vectorizer only on the training texts and transforms validation/test texts afterward.

## 6. Neural Network Structure

**Question:** What are the layers in a neural network?

**Theory answer:** The input layer receives features. Hidden layers transform features using weights, biases, and activation functions. The output layer produces the prediction.

**How used in the assignment:** The input is the bag-of-words vector. Hidden dense layers learn combinations of words. The output layer has one neuron for binary sentiment prediction.

## 7. Forward Propagation

**Question:** What is forward propagation?

**Theory answer:** Forward propagation passes input through layers. Each neuron computes a weighted sum plus bias, applies an activation function, and passes the result forward.

**How used in the assignment:** During prediction, review vectors flow through dense layers and end in a sentiment probability.

## 8. Backpropagation

**Question:** What is backpropagation?

**Theory answer:** Backpropagation computes gradients of the loss with respect to weights using the chain rule. The optimizer uses those gradients to update weights.

**How used in the assignment:** Keras handles backpropagation internally while training the neural network over multiple epochs.

## 9. Activation Functions

**Question:** Why use activation functions?

**Theory answer:** Activation functions introduce nonlinearity. Without them, multiple layers would collapse into one linear transformation.

**How used in the assignment:** The notebook compares hidden activations such as ReLU and tanh. The output uses sigmoid because the task is binary classification.

## 10. Sigmoid Output

**Question:** Why use sigmoid in the output layer?

**Theory answer:** Sigmoid maps a real-valued score into a value between 0 and 1, which can be interpreted as probability for the positive class.

**How used in the assignment:** The model outputs a probability that a review is positive. A threshold such as 0.5 converts it into a class label.

## 11. Loss Function

**Question:** What loss function is used for binary sentiment classification?

**Theory answer:** Binary cross-entropy is commonly used for binary classification. It penalizes wrong confident predictions strongly.

**How used in the assignment:** The Keras model is trained with a binary classification loss so predicted sentiment probabilities match the labels.

## 12. Optimizer

**Question:** What does the optimizer do?

**Theory answer:** The optimizer updates weights based on gradients to reduce the loss. SGD and Adam are common optimizers.

**How used in the assignment:** The notebook trains models using a Keras optimizer, comparing training and validation behaviour across epochs.

## 13. Epoch, Batch, Iteration

**Question:** Explain epoch, batch, and iteration.

**Theory answer:** An epoch is one full pass over the training data. A batch is a subset processed before one update. An iteration is one weight update from one batch.

**How used in the assignment:** The neural network trains over epochs, with batches used internally by Keras.

## 14. Hyperparameters

**Question:** What neural-network hyperparameters matter?

**Theory answer:** Important hyperparameters include number of layers, number of neurons, activation functions, learning rate, optimizer, batch size, epochs, and regularization strength.

**How used in the assignment:** The notebook compares different hidden units, activations, L2 regularization, and training settings.

## 15. Overfitting And Early Stopping

**Question:** How do neural networks overfit?

**Theory answer:** A neural network overfits when it learns training-specific patterns instead of general sentiment patterns. Training accuracy improves while validation performance stops improving or gets worse.

**How used in the assignment:** The notebook uses validation curves and early stopping/regularization discussion to control overfitting.

## 16. Regularization

**Question:** What regularization can be used in neural networks?

**Theory answer:** Common methods include L1, L2, dropout, and early stopping. They reduce model complexity or stop training before overfitting.

**How used in the assignment:** The notebook compares models with L2 regularization and discusses early stopping.

## 17. Classification Metrics

**Question:** Why evaluate with precision, recall, F1, and confusion matrix?

**Theory answer:** Accuracy alone can hide error types. Precision, recall, F1, and the confusion matrix show whether the model makes more false positives or false negatives.

**How used in the assignment:** The notebook evaluates final sentiment predictions using classification metrics and confusion matrix.

## 18. Limitations

**Question:** What are the limitations of bag-of-words sentiment analysis?

**Theory answer:** Bag-of-words ignores word order, negation structure, sarcasm, context, and meaning beyond word counts.

**How used in the assignment:** The notebook tests custom sentences, which can reveal cases where simple word-count representation is not enough.

## Fast Last-Minute Answers

- **Main task:** binary text classification.
- **Encoding:** bag-of-words with `CountVectorizer`.
- **Model:** feed-forward neural network.
- **Hidden activation:** ReLU/tanh.
- **Output activation:** sigmoid.
- **Best one-sentence defense:** I convert raw reviews into bag-of-words vectors, train neural networks for binary sentiment classification, tune basic hyperparameters, and evaluate the result with validation curves and classification metrics.

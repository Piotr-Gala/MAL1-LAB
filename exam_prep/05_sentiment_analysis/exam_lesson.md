# Assignment 5: Sentiment Analysis - ADHD-Friendly Oral Exam Lesson

## How To Use This File

To jest wersja do nauki ustnej: najpierw masz normalny kawałek lekcji, a dopiero pod spodem wyjaśnienia słów z tej sekcji.

Schemat:

```text
section
lesson text
term explanations
exam sentences
where in code
```

Main exam pattern:

```text
What does it mean?
Where is it in your code?
Why did you do it?
```

---

# 0. Big Picture

Assignment 5 is about sentiment analysis on IMDb movie reviews.

The goal is to classify each review as negative or positive.

This is supervised binary text classification.

The input is raw review text.

The target is sentiment:

```text
0 = negative
1 = positive
```

The main workflow is:

```text
raw reviews -> train/validation/test split -> Bag-of-Words -> neural network -> validation tuning -> final test
```

The most important defensive point is that the raw text is split before fitting the vectorizer.

## Terms From This Section

`sentiment analysis`

```text
An NLP task where text is classified by opinion or emotion.
Here: negative or positive movie review.
```

`supervised`

```text
The model learns from examples with known labels.
Here, each review has a known positive/negative label.
```

`binary classification`

```text
Classification with two possible classes.
Here: negative or positive.
```

`raw text`

```text
Original review strings before numerical preprocessing.
```

`target`

```text
The thing the model should predict.
Here: sentiment label 0 or 1.
```

## Exam Sentences

```text
This assignment is supervised binary sentiment classification.
The model receives movie review text and predicts whether the sentiment is negative or positive.
The main workflow is to split raw text first, convert it into Bag-of-Words vectors, train a neural network, tune on validation data and evaluate once on the test set.
```

---

# 1. Loading Data And Labels

The notebook loads two text files:

```text
reviews.txt
labels.txt
```

The reviews are stored as text.

The labels are originally strings:

```text
positive
negative
```

They are converted to numerical labels:

```text
positive -> 1
negative -> 0
```

This is needed because neural networks work with numerical targets.

## Terms From This Section

`label`

```text
The correct answer for one example.
Here: positive or negative.
```

`numerical target`

```text
A target represented as numbers instead of strings.
Neural networks need numerical values.
```

`positive class`

```text
The class encoded as 1.
Here: positive review.
```

`negative class`

```text
The class encoded as 0.
Here: negative review.
```

## Exam Sentences

```text
I load the raw reviews and labels, then convert the sentiment labels into 0 and 1.
This makes the task a supervised binary classification problem.
```

## Where In Code

```text
Cells 3-4: load reviews.txt and labels.txt, then encode labels as 0 and 1.
```

---

# 2. Initial Data Check And Class Balance

Before modelling, the notebook checks the class distribution.

The dataset has:

```text
12500 negative reviews
12500 positive reviews
```

So the dataset is perfectly balanced.

Because the classes are balanced, accuracy is a reasonable metric.

But the notebook still checks precision, recall and F1-score later.

## Terms From This Section

`class distribution`

```text
How many examples belong to each class.
```

`balanced dataset`

```text
Both classes have similar or equal numbers of examples.
Here: 12500 negative and 12500 positive.
```

`accuracy`

```text
The percentage of correct predictions overall.
```

`precision / recall / F1`

```text
Extra classification metrics that show more detail than accuracy.
```

## Exam Sentences

```text
I check the target distribution before modelling.
The dataset is balanced, with 12500 negative and 12500 positive reviews.
Because the classes are balanced, accuracy is meaningful, but I still inspect precision, recall and F1-score.
```

## Where In Code

```text
Cells 5-7: class counts, class proportions and example reviews.
```

---

# 3. Train / Validation / Test Split

The raw text is split before fitting any vectorizer.

Split sizes:

```text
Train:      16000 reviews
Validation: 4000 reviews
Test:        5000 reviews
```

The roles are:

```text
train -> fit preprocessing and train model
validation -> choose hyperparameters
test -> final independent evaluation
```

The split uses stratification so each split keeps the same positive/negative balance.

## Terms From This Section

`train set`

```text
Data used to fit the vectorizer and train the model.
```

`validation set`

```text
Data used to choose hyperparameters.
```

`test set`

```text
Data used only once at the end for final evaluation.
```

`stratification`

```text
Keeping class proportions similar in each split.
```

`generalization`

```text
How well the model works on new unseen reviews.
```

## Exam Sentences

```text
I split the raw text before any vectorizer is fitted.
Training data is used to fit preprocessing and train the model.
Validation data is used for model selection, and the test set is saved for final evaluation.
```

## Where In Code

```text
Cells 8-9: train, validation and test split with stratification.
```

---

# 4. Bag-of-Words With CountVectorizer

Raw text cannot be used directly by the neural network.

The notebook uses Bag-of-Words encoding with:

```python
CountVectorizer(max_features=10000)
```

This creates a vocabulary of up to 10,000 frequent words.

Each review becomes a vector:

```text
each column = one word
each value = count of that word in the review
```

The vectorizer is fitted only on training text.

Validation and test text are only transformed.

## Terms From This Section

`Bag-of-Words`

```text
Text representation based on word counts.
It counts which words appear and how often.
```

`CountVectorizer`

```text
Scikit-learn tool that converts text into word-count vectors.
```

`vocabulary`

```text
The set of words used as columns in the Bag-of-Words matrix.
```

`max_features=10000`

```text
Keeps only the 10000 most frequent word features.
This reduces dimensionality, memory use and noise.
```

`word-count vector`

```text
A numerical representation of one review.
Each position counts one vocabulary word.
```

## Exam Sentences

```text
Bag-of-Words represents text as word counts.
CountVectorizer creates a fixed vocabulary and converts each review into a vector of word counts.
I use max_features=10000 to reduce dimensionality and keep the representation practical.
```

## Where In Code

```text
Cells 10-11: CountVectorizer(max_features=10000), fit_transform on train and transform on validation/test.
```

---

# 5. Split Before Vectorizer And Data Leakage

The vectorizer learns the vocabulary from data.

That means it must be fitted only on training text.

Correct workflow:

```text
fit_transform on training text
transform on validation text
transform on test text
```

If the vectorizer were fitted on all reviews before splitting, the vocabulary would be influenced by validation and test reviews.

That would be data leakage.

## Terms From This Section

`fit_transform`

```text
Learn the transformation and apply it.
Used on training text.
```

`transform`

```text
Apply an already learned transformation.
Used on validation and test text.
```

`data leakage`

```text
When validation or test information influences training.
This makes evaluation too optimistic.
```

`vocabulary leakage`

```text
The vocabulary is learned from text that should be unseen.
```

## Exam Sentences

```text
I split before preprocessing.
The vectorizer is fitted only on training data, so validation and test information does not leak into the training process.
If I fitted CountVectorizer on all reviews before the split, the vocabulary would be influenced by validation and test text.
```

## Where In Code

```text
Cells 8-11: raw text split first, then vectorizer fitted only on training text.
```

---

# 6. Sparse And Dense Representation

Bag-of-Words vectors are sparse.

Most values are zero because one review uses only a small part of the 10,000-word vocabulary.

The notebook converts sparse matrices into dense `float32` arrays for Keras.

This keeps the neural network input simple.

The trade-off is higher memory usage.

## Terms From This Section

`sparse vector`

```text
A vector where most values are zero.
Bag-of-Words vectors are sparse.
```

`dense array`

```text
A normal array storing all values, including zeros.
Easier for this Keras model, but uses more memory.
```

`float32`

```text
A numerical data type commonly used for neural network inputs.
```

`trade-off`

```text
A benefit with a cost.
Here: simpler Keras input but more memory use.
```

## Exam Sentences

```text
CountVectorizer gives sparse matrices because most word-count values are zero.
For this simple Keras model, I convert them to dense float32 arrays.
The trade-off is higher memory usage, but the code stays simple and close to the course examples.
```

## Where In Code

```text
Cells 12-15: representation example and conversion to dense float32 arrays.
```

---

# 7. Neural Network Architecture

The baseline model is a feed-forward neural network with one hidden layer.

Architecture:

```text
Input: 10000 Bag-of-Words features
Hidden layer: 32 neurons, ReLU
Output layer: 1 neuron, sigmoid
```

The input layer receives word-count features.

The hidden layer learns combinations of words.

The output layer predicts the probability that the review is positive.

## Terms From This Section

`feed-forward neural network`

```text
A neural network where information moves from input to output through layers.
```

`input layer`

```text
Receives the 10000 Bag-of-Words features.
```

`hidden layer`

```text
Layer between input and output.
It learns combinations of features.
```

`neuron`

```text
A unit that computes a weighted sum plus bias and applies an activation function.
```

`output layer`

```text
Produces the final prediction.
Here: one sigmoid neuron.
```

## Exam Sentences

```text
The input layer receives 10000 word-count features.
The hidden layer learns combinations of words.
The output layer has one sigmoid neuron because this is binary classification.
```

## Where In Code

```text
Cells 16-18: baseline Keras Sequential neural network.
Cells 22-28: tuned and final models.
```

---

# 8. Forward Propagation And Backpropagation

Forward propagation means passing input through the network to get a prediction.

Each neuron computes:

```text
weighted sum + bias -> activation function
```

Backpropagation is used during training.

It computes how the model weights should change to reduce the loss.

Keras handles backpropagation internally.

## Terms From This Section

`forward propagation`

```text
Passing data through the network from input to output.
```

`weighted sum`

```text
Inputs multiplied by weights and added together.
```

`bias`

```text
A constant value added by a neuron.
Similar idea to intercept.
```

`backpropagation`

```text
Algorithm that computes gradients of the loss with respect to weights.
Used to train neural networks.
```

`gradient`

```text
Information about how to change weights to reduce loss.
```

## Exam Sentences

```text
Forward propagation passes the review vector through the layers to produce a sentiment probability.
Backpropagation computes gradients of the loss with respect to weights.
Keras handles backpropagation internally during training.
```

---

# 9. Activation Functions

Activation functions introduce nonlinearity.

Without activation functions, multiple layers would collapse into one linear transformation.

The notebook uses:

```text
ReLU in hidden layers
tanh as an alternative hidden activation
sigmoid in the output layer
```

Sigmoid is used in the output layer because this is binary classification.

It gives a value between 0 and 1.

## Terms From This Section

`activation function`

```text
A function applied inside neurons.
It lets the network learn nonlinear patterns.
```

`ReLU`

```text
Common hidden-layer activation.
Outputs zero for negative values and keeps positive values.
```

`tanh`

```text
Activation function that maps values roughly between -1 and 1.
```

`sigmoid`

```text
Activation function that maps values between 0 and 1.
Useful for binary probability output.
```

`nonlinearity`

```text
Allows the network to learn more complex patterns than a simple linear model.
```

## Exam Sentences

```text
Activation functions introduce nonlinearity.
ReLU and tanh are hidden-layer activation functions.
Sigmoid is used in the output layer because it gives a probability for the positive class.
```

## Where In Code

```text
Cells 16-18: ReLU hidden layer and sigmoid output.
Cells 22-24: ReLU and tanh compared during tuning.
```

---

# 10. Loss Function And Optimizer

The model uses binary crossentropy loss.

This is appropriate because the target is binary:

```text
0 = negative
1 = positive
```

The optimizer updates model weights to reduce the loss.

The notebook uses:

```text
SGD for the baseline
Adam for tuned models
```

Adam often converges faster because it adapts learning rates.

## Terms From This Section

`loss function`

```text
Measures how wrong the model predictions are during training.
```

`binary crossentropy`

```text
Loss function for binary classification.
It penalizes confident wrong predictions strongly.
```

`optimizer`

```text
Algorithm that updates weights to reduce the loss.
```

`SGD`

```text
Stochastic Gradient Descent.
A simple optimizer.
```

`Adam`

```text
Adaptive optimizer that often trains faster or more smoothly than plain SGD.
```

## Exam Sentences

```text
The model uses binary crossentropy because the target is binary and the output is sigmoid.
The optimizer updates the weights to reduce the loss.
SGD is simple, while Adam often converges faster because it adapts the learning rate.
```

## Where In Code

```text
Cells 16-18: baseline model compilation.
Cells 22-28: tuned and final model compilation.
```

---

# 11. Epoch, Batch And Iteration

Neural networks train over epochs.

An epoch means one full pass over the training data.

A batch is a smaller subset of training examples processed before one weight update.

An iteration is one update step based on one batch.

Keras handles batches internally during training.

## Terms From This Section

`epoch`

```text
One full pass over the training set.
```

`batch`

```text
A subset of training examples used for one update.
```

`iteration`

```text
One weight update from one batch.
```

`weight update`

```text
Changing model weights to reduce loss.
```

## Exam Sentences

```text
An epoch is one full pass over the training data.
A batch is a subset processed before one update.
An iteration is one weight update from one batch.
```

---

# 12. Learning Curves And Overfitting

The notebook plots training and validation accuracy/loss across epochs.

These learning curves help detect overfitting.

If training accuracy improves but validation performance stops improving or gets worse, the model may be overfitting.

In the baseline model:

```text
Train accuracy: about 0.8645
Validation accuracy: about 0.8485
```

Training accuracy is higher than validation accuracy, so there is some overfitting, but not extreme.

## Terms From This Section

`learning curve`

```text
A plot of training and validation performance over epochs.
```

`overfitting`

```text
The model learns training-specific patterns instead of general patterns.
It performs better on training data than on validation/test data.
```

`validation loss`

```text
Loss measured on validation data.
Used to monitor generalization.
```

`generalization`

```text
How well the model works on unseen data.
```

## Exam Sentences

```text
I compare training and validation curves to see whether the model generalizes.
Training accuracy is higher than validation accuracy, so there is some overfitting.
Learning curves help show whether the model keeps improving or starts memorizing the training data.
```

## Where In Code

```text
Cells 19-21: learning curves and baseline evaluation.
```

---

# 13. Hyperparameter Tuning

The notebook tests a small number of neural network settings.

Compared hyperparameters:

```text
hidden units
activation function
L2 regularization strength
optimizer
```

Tested candidates:

```text
32 units, ReLU, no L2, Adam
64 units, ReLU, L2 = 0.001, Adam
64 units, tanh, L2 = 0.001, Adam
```

Best validation model:

```text
hidden_units: 32
activation: relu
l2_strength: 0.0
optimizer: adam
validation accuracy: 0.8907
```

The test set is not used during tuning.

## Terms From This Section

`hyperparameter`

```text
A setting chosen before training or by validation.
Examples: hidden units, activation, optimizer.
```

`hidden units`

```text
Number of neurons in the hidden layer.
```

`L2 regularization`

```text
Penalty for large weights.
It can reduce overfitting.
```

`model selection`

```text
Choosing the best model configuration using validation performance.
```

## Exam Sentences

```text
I tune only a small number of relevant hyperparameters.
The best model is selected using validation accuracy, not test accuracy.
The best validation model uses 32 ReLU hidden units and Adam.
```

## Where In Code

```text
Cells 22-24: hyperparameter tuning with validation accuracy.
```

---

# 14. Early Stopping And Regularization

The notebook uses early stopping during tuning.

Early stopping monitors validation loss.

If validation loss stops improving, training stops.

The notebook also tests L2 regularization candidates.

Both techniques are used to reduce overfitting risk.

## Terms From This Section

`early stopping`

```text
Stops training when validation loss stops improving.
It helps avoid training too long.
```

`patience`

```text
How many epochs to wait without improvement before stopping.
```

`restore_best_weights`

```text
After stopping, Keras restores the weights from the best validation epoch.
```

`regularization`

```text
Techniques that reduce overfitting.
Examples: L2, dropout, early stopping.
```

## Exam Sentences

```text
Early stopping stops training when validation loss stops improving.
This helps prevent the model from continuing to fit the training data too closely.
L2 regularization penalizes large weights and can reduce overfitting.
```

## Where In Code

```text
Cells 22-24: EarlyStopping and L2 candidates during tuning.
Cell 33: limitations and overfitting discussion.
```

---

# 15. Final Model

After choosing hyperparameters, the notebook trains the final model.

It refits the final vectorizer on train plus validation text.

Then it trains the final neural network on train plus validation data.

This is acceptable because validation data has already been used for model selection.

The test set is still untouched until final evaluation.

The final model uses:

```text
32 hidden units
ReLU
Adam
sigmoid output
binary crossentropy
```

## Terms From This Section

`train plus validation`

```text
After choosing hyperparameters, training on more data can improve the final model.
```

`refit vectorizer`

```text
Fit a new final vectorizer on train + validation text.
The test set is still not used.
```

`untouched test set`

```text
Test data not used for fitting, tuning or vocabulary creation.
```

## Exam Sentences

```text
After model selection, I retrain the final model on train plus validation data.
I also refit the final vectorizer on train plus validation text.
The test set is still used only once at the end.
```

## Where In Code

```text
Cells 25-28: final vectorizer and final neural network training.
```

---

# 16. Final Test Evaluation And Metrics

The final model is evaluated once on the test set.

Final result:

```text
Test accuracy: 0.8788
```

Precision, recall and F1-score are around:

```text
0.88 for both classes
```

The confusion matrix shows mistake types:

```text
negative predicted as negative = true negative
negative predicted as positive = false positive
positive predicted as negative = false negative
positive predicted as positive = true positive
```

Because the test set is balanced, accuracy is meaningful.

But precision, recall and F1-score are still checked to make sure both classes perform similarly.

## Terms From This Section

`precision`

```text
When the model predicts a class, how often it is correct.
```

`recall`

```text
Out of all true examples of a class, how many the model finds.
```

`F1-score`

```text
A balance between precision and recall.
```

`confusion matrix`

```text
A table showing correct and incorrect predictions by class.
```

`false positive`

```text
Negative review predicted as positive.
```

`false negative`

```text
Positive review predicted as negative.
```

## Exam Sentences

```text
The final model reaches about 87.9% test accuracy.
Because the test set is balanced, accuracy is meaningful.
I also check precision, recall and F1-score to make sure both classes perform similarly.
```

## Where In Code

```text
Cells 29-30: final test evaluation, classification report and confusion matrix.
```

---

# 17. Custom Sentences

The notebook also tests custom manually written sentences.

The same final vectorizer is used:

```text
vectorizer_final.transform(my_sentences)
```

A new vectorizer is not fitted.

This is important because the model expects the same vocabulary columns as during training.

If a new vectorizer were fitted, the feature mapping would change.

## Terms From This Section

`custom sentence`

```text
A new manually written sentence used to test the model.
```

`same vectorizer`

```text
The already fitted final vectorizer.
It keeps the same vocabulary mapping.
```

`feature mapping`

```text
Which word corresponds to which column.
The model expects this mapping to stay fixed.
```

## Exam Sentences

```text
For new sentences, I transform them with the same final vectorizer.
I do not fit a new vectorizer because that would create a different vocabulary mapping.
The model expects the same feature columns as during training.
```

## Where In Code

```text
Cells 31-32: custom sentence predictions using vectorizer_final.transform().
```

---

# 18. Limitations

The main limitation is the Bag-of-Words representation.

Bag-of-Words ignores:

```text
word order
context
grammar
negation structure
sarcasm
```

Example problem:

```text
good
not good
```

Bag-of-Words sees the words, but it does not deeply understand the phrase structure.

Dense conversion also uses more memory.

The model is simpler than modern NLP models.

Possible improvements:

```text
TF-IDF
word embeddings
recurrent neural networks
transformers
```

## Terms From This Section

`word order`

```text
The sequence of words in a sentence.
Bag-of-Words ignores it.
```

`context`

```text
Meaning from surrounding words.
```

`negation`

```text
Words like "not" that can reverse meaning.
```

`sarcasm`

```text
Text where literal words and intended meaning differ.
```

`TF-IDF`

```text
A text representation that weights words by importance, not only raw count.
```

`embeddings`

```text
Dense vector representations of words/text that can capture semantic similarity.
```

`transformers`

```text
Modern NLP models that use context and attention.
```

## Exam Sentences

```text
The main limitation is that Bag-of-Words ignores word order and context.
This makes negation and sarcasm difficult.
More advanced methods could use TF-IDF, embeddings or transformers, but this assignment stays with a simple course-level neural network workflow.
```

## Where In Code

```text
Cell 33: limitations.
```

---

# 19. Where Is It In The Code?

Use this if the examiner asks where something appears in the notebook.

```text
Imports:
Cells 1-2.

Load reviews and labels:
Cells 3-4.

Initial data check and class balance:
Cells 5-7.

Train-validation-test split:
Cells 8-9.

Bag-of-Words / CountVectorizer:
Cells 10-11.

Representation example:
Cells 12-13.

Dense float32 conversion:
Cells 14-15.

Baseline neural network:
Cells 16-18.

Learning curves:
Cells 19-21.

Hyperparameter tuning:
Cells 22-24.

Final model:
Cells 25-28.

Final test evaluation:
Cells 29-30.

Custom sentences:
Cells 31-32.

Limitations:
Cell 33.
```

---

# 20. A5 In 30 Seconds

```text
Assignment 5 is binary sentiment classification on IMDb reviews.
The input is raw movie review text, and the target is negative or positive sentiment.

I first convert labels into 0 and 1 and split the raw text into train, validation and test sets.
Then I fit CountVectorizer only on the training text to create a Bag-of-Words representation with 10000 features.

After that, I train a one-hidden-layer Keras neural network.
The hidden layer uses ReLU, and the output layer uses sigmoid because this is binary classification.
The loss is binary crossentropy.

I tune a few hyperparameters using validation accuracy and keep the test set untouched.
The final model reaches about 87.9% test accuracy, with precision, recall and F1-score around 0.88.
The main limitation is that Bag-of-Words ignores word order and context.
```

---

# 21. Emergency Speaking Pattern

If you forget a formal definition, use this pattern:

```text
[Term] means [simple meaning].
In my assignment, I used it for [specific thing].
The reason is [why].
```

Example:

```text
Bag-of-Words means representing text as word counts.
In my assignment, I used CountVectorizer to convert reviews into 10000-dimensional vectors.
The reason is that neural networks need numerical input, not raw strings.
```

---

# 22. Top Words To Memorize

```text
sentiment analysis = classify opinion in text
binary classification = two classes
label = correct answer
target = 0 negative, 1 positive
Bag-of-Words = word-count representation
CountVectorizer = converts text to word-count vectors
vocabulary = words used as columns
max_features = vocabulary size limit
sparse vector = mostly zeros
dense array = all values stored
neural network = layered model
hidden layer = learns combinations of features
activation = nonlinearity
ReLU = hidden activation
sigmoid = binary probability output
binary crossentropy = binary classification loss
optimizer = updates weights
epoch = one full pass over training data
early stopping = stop when validation loss stops improving
overfitting = training good, validation/test worse
```

Final survival sentence:

```text
The key point in this assignment is that I split the raw text before fitting CountVectorizer, train a simple neural network for binary sentiment classification, tune on validation data, and evaluate once on the untouched test set.
```

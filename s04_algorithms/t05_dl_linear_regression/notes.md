# Linear Regression Notes in deep learning
* activation functions like relu / sigmoids enable the model to be non linear.
* relu became popular because the mathematics is faster and the output results in less memory usage
* use a softmax from penultimate to classification layer for multiclass to ensure probs are non negative and sum to 1 for all classes
* For classification, minimize neg-log-likelihood which is the sum y*log(f(x)) over all classes. This is CROSS ENTROPY loss fucntion
* For regression, minimize squared error loss
* dropout was inspired from random forests. Randomly remove a fraction of units in a layer when fitting, which help prevents nodes from becoming to specialized. Keep architecture in tact, and set Relu values for a perenct output to zero.
* 2 class logistic regression to multiclass logistic regression: 1) sigmoid activation -> softmax activation, 2) 0/1 labels -> 0/N labels 3) binary cross entropy -> general cross entropy. In multiclass regression, you need N sets of weights for your model to predict prob of N classes. Rather than using a 0.5 threshold on a sigmoid function, we simply take the argmax of the softmax activation. This gives us the index of the one-hot encoded class labels.
* Cross entropy loss for multiclass logistic regression: 1/n Sum of all i Sum of all k (-yk[i]log(ak[i])). K class labels, n rows of data, a is activation. y is one hot encoded value of the all the labels. Basically, only the max of softmax prediction and it's activation contribute to the loss function. For all other classes where the predicted probability is lower, these values do not contribute to the loss function.
* What's better, narrow (not many activations per layer) and deep (many hidden layers), or wide (many activations) and shallow (few hidden layers)
* in theory, we only need one non-linear activation function in a nerual net to approximate any function. However, with 1 layer and many activations, the model is prone to memorization and can overfit. with wide and narrow models, less activations are needed to learn, but the model can have issues with vanishing/exploding gradients and be more difficult to train. Bottom line -> use both wide and narrow design.
* initialize with random weights


# Regularization
* can apply ridge / lasso regularization as usual for regression and classification problems.
* More common to use dropout regularization
* data augmentation is a great way to regularize
* bi-directional RNNs
* early stopping

# Feature Normalization
* if feature inputs are on different scales, then the partial derivatives in gradient descent will be on different scales. However, all partial derivatives are multiplied by a single learning rate, making it very difficult to choose a single learning rate. With feature normalization, a single learning rate will work for all features, leading to faster convergence, and fewer epochs of training. It also leads to more stable gradients.
* Min-Max Normalization: x - min / (max - min) -- leads to smallest value at 0, biggest value at 1 for each feature
* Z-score standardization x - mean / std. It's better to use in deep learning because the center of the distribution will be at zero which makes gradient descent more stable and behave better.

# CNNs
* convolutions are kernels (3x3, 5x5, etc) that look through an image for edges, colors, etc
* pooling layers take a kernel and summarize it in a smaller kernel to reduce the hidden layer size.
* cnns learn the convolutional filters applied in the images as opposed to traditional cv which defines the filters for you.
* if you have K convolutions (typically 3x3x3 to include RGB), your result in K-2d arrays for activations for each convolution, or ixjxk 3d matrix. this passes through a Relu.
* Max pooling typically takes non-overlapping 2x2 blocks and takes the max value. Resulting in i/2, j/2 dimensions.
* subsequent convolutions apply on the input 3D feature map in all 3 dimensions. Again, the number of convolutions defined at the next layer will define the depth of the 3d output.

# Document classification
* how to convert words to numerics:
1. bag of words: take 10k most common words in training data, and create X with 10k features of sparse data where the value represents counts of a given word.
2. TFIDF - normalizes the word count in a document by how much it shows up in the corpus
3. Bag of n-grams allows more context, but explodes the vector space
4. Sequential models like RNNs are much better at this as they hold more context on the sequence of word inputs.
5. Using word embeddings is a much better / lower memory way to store word information. It is based on looking through millions of documents at the words used around it to build relationships.
* word embeddings (word2vec and glove) are determined by a variant of pca and.

# RNNs
* sequences of vectors representing sequential data are processed to a hiddent layer, and a sequential set of weights are used to apply bias on sequential activation functions to keep "memory". The final activation function for the sequence is affected by all the prior activation functions, and then results in an output classification. Model parameters W, U, and B are the same for every item of the sequence! This helps learn how words relate to each other in a sentence.
* input length must be the same as U expects a certain length. Therefore, we limit each document to the last L word and documents shorter than L are padded with zeros upfront.
* if you want, you can actually fit the word embedding space, but it takes a ton of training to do this.
* RNNs do not work very well, even with glove and word2vec.
* RNNs train quickly, and recent developsments have enabled them to outperform sequential taks.

# LSTMs
The difference between RNNs and LSTMs is that LSTMs use 2 tracks of hidden layer activations rather than just one. One track of hidden layer is focused on recent context, and one maintains longer term context.
* its wild, but RNNs significantly underperform sentiment analysis tasks from linear logistic regression (WHAT) and  2-hidden layer feed forward network. LSTMs are only able to match this performance on a generic sentiment analysis dataset from IMDb data.
* LSTMs take a very long time to train.

# Time Series w RNNs
* one sequence of input vs something like 25k documents
* need output predictions for every sequence of future
* for setup, extract short mini series of input sequences with predefined length L, and a corresponding y target.

# Seq2Seq Models
* are RNNs which take the output from each input embedding to enable things like language translation.

# Classification
Log loss and Negative Log Likelihood are used interchangeably for binary classfiication problems. Cross Entropy loss is a generalization of log loss for multiclass classifcation, and with a softmax ensures all probabilities are nonzero and sum to 1

# Interpolation and Double Descent
Interpolating the Training Data refers to building a model that fits the training data perfectly, ie. achieving zero training error. In other words, the model predicts the training data so accurately that it exactly matches the provided examples. This means overfit for validation data. “Double descent” gets its name from the fact that the test error has a U-shape before the interpolation threshold is reached, and then it descends again (for a while, at least) as an increasingly flexible model is fit.

Double descent has been used by machine learning community to explain the successful practice of using overparameterized models to achieve near zero training error that allows for models to generalize well.

# Optimization Algorithms
* SGD is basic approach
* Add momemntum to dappen oscilations. Basically take SGD and move in the average of last direction. In pytorch, just add momentum=0.9 to SGD optimizer.
* ADAM - Adaptive learning rate with momentum: 1) decrease learning if gradient changes direction, 2) increase learning if gradient stays consistent. This algo is built on RMSProp algo which is a moving average of the squared gradient of each weight. Adam is easier to tune and works well in practice. You can still use learning rate scheduler with ADAM.
* learning rate schedulers: StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Choosing Activation Functions
* Sigmoid Function - not recommended for hidden activations as it will slow SGD convergence. The max gradient value of a sigmoid function is at x=0 y=0.5 and gradient=0.25. So when used for many hidden layers, we are taking fractions of our loss signal and multiplying by many gradients which will push updates toward zero.
* ReLU - derivative below 0 is 0, derivative above zero is 1
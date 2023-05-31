import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the training and validation datasets
train_data = pd.read_csv('data/train.csv', header=None)
validate_data = pd.read_csv('data/validate.csv', header=None)

# Separate the features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values - 1
X_validate = validate_data.iloc[:, 1:].values
y_validate = validate_data.iloc[:, 0].values - 1

# y_train = y_train - 1
# y_validate = y_validate - 1

# Convert labels to one-hot encoding
y_train_encoded = np.eye(10)[y_train]
y_validate_encoded = np.eye(10)[y_validate]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def RELU(x):
    return np.maximum(0, x)


def RELU_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_RELU(x):
    return np.maximum(x, 0.01 * x)


def leaky_RELU_derivative(x):
    return np.where(x > 0, 1, 0.01)


ActivationFunction = sigmoid
ActivationFunctionDerivative = sigmoid_derivative


# Define the neural network model
class NeuralNetwork2:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_prob, regularization):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.regularization = regularization

        self.W1 = np.random.randn(self.input_size, self.hidden_size1)
        self.b1 = np.zeros((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.b2 = np.zeros((1, self.hidden_size2))
        self.W3 = np.random.randn(self.hidden_size2, self.output_size)
        self.b3 = np.zeros((1, self.output_size))

    def forward(self, X, training):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ActivationFunction(self.z1)

        # Apply dropout during training
        if training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=self.a1.shape)
            self.a1 *= dropout_mask / (1 - self.dropout_prob)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = ActivationFunction(self.z2)

        # Apply dropout during training
        if training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=self.a2.shape)
            self.a2 *= dropout_mask / (1 - self.dropout_prob)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, X, y, output, learning_rate):
        delta3 = output - y

        dW3 = np.dot(self.a2.T, delta3) + self.regularization * self.W3
        db3 = np.sum(delta3, axis=0, keepdims=True)

        # Backpropagate dropout mask
        dropout_mask2 = np.random.binomial(1, 1 - self.dropout_prob, size=self.a2.shape)
        delta2 = np.dot(delta3, self.W3.T) * ActivationFunctionDerivative(self.z2) * dropout_mask2 / (
                1 - self.dropout_prob)

        dW2 = np.dot(self.a1.T, delta2) + self.regularization * self.W2
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Backpropagate dropout mask
        dropout_mask1 = np.random.binomial(1, 1 - self.dropout_prob, size=self.a1.shape)

        delta1 = np.dot(delta2, self.W2.T) * ActivationFunctionDerivative(self.z1) * dropout_mask1 / (
                    1 - self.dropout_prob)

        dW1 = np.dot(X.T, delta1) + self.regularization * self.W1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, num_epochs, initial_learning_rate, batch_size, decay_rate):
        learning_rate = initial_learning_rate

        for epoch in tqdm(range(num_epochs), file=sys.stdout):
            # Shuffle the training data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch_start in range(0, X.shape[0], batch_size):
                # Divide the data into mini-batches
                batch_end = batch_start + batch_size
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward pass
                output = self.forward(X_batch, True)
                # print(np.argmax(output, axis=1))

                # Backward pass
                self.backward(X_batch, y_batch, output, learning_rate)

            # Update the learning rate every 5 epochs
            if (epoch + 1) % 5 == 0:
                learning_rate *= decay_rate

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                validate_predictions = self.predict(X_validate)
                validate_accuracy = np.mean(validate_predictions == np.argmax(y_validate_encoded, axis=1)) * 100
                print(f"After {epoch + 1} epochs, Validation Accuracy : {validate_accuracy}%")

                # Calculate training accuracy on a random subset of 1000 different examples
                random_indices = np.random.choice(X.shape[0], size=1000)
                X_train_subset = X[random_indices]
                y_train_subset = y[random_indices]
                train_predictions = self.predict(X_train_subset)
                train_accuracy = np.mean(train_predictions == np.argmax(y_train_subset, axis=1)) * 100
                print(f"After {epoch + 1} epochs, Training Accuracy (Subset of 1000): {train_accuracy}%")

    def predict(self, X):
        output = self.forward(X, False)
        return np.argmax(output, axis=1)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


def _pre_processing(X: np.ndarray, Y: np.ndarray, noise_std: float = 0.1):
    X_reshaped = X.reshape((-1, 3, 32, 32))

    # Flipping Images horizontally
    horizontal_flipped = np.flip(X_reshaped, axis=3)
    horizontal_flipped = horizontal_flipped.reshape(X.shape[0], -1)

    # Flipping Images vertically
    vertically_flipped = np.flip(X_reshaped, axis=2)
    vertically_flipped = vertically_flipped.reshape(X.shape[0], -1)

    X_new = np.concatenate((X, horizontal_flipped, vertically_flipped))
    Y_new = np.concatenate((Y, Y, Y))

    # Resetting Pixels
    pixels_num = int(X.shape[1] * 0.2)
    X_reset = X_new.copy()
    for i in range(X.shape[0]):
        indices_to_reset = np.random.choice(X.shape[1], size=pixels_num, replace=False)
        X_reset[i, indices_to_reset] = 0

    # Adding Gaussian Noise
    noise = np.random.normal(loc=0, scale=noise_std, size=X_new.shape)
    X_noisy = X_new + noise

    X_new = np.concatenate((X_new, X_reset, X_noisy))
    Y_new = np.concatenate((Y_new, Y_new, Y_new))

    return X_new, Y_new


def main():
    np.random.seed(10)
    np.seterr(invalid="raise")
    # Set the hyperparameters
    input_size = X_train.shape[1]
    hidden_size1 = 512
    hidden_size2 = 256
    output_size = 10
    num_epochs = 100
    init_learning_rate = 0.05
    learning_rate_decay = 0.9
    batch_size = 64
    dropout_prob = 0.2
    regularization = 0

    # Create the neural network model
    model = NeuralNetwork2(input_size, hidden_size1, hidden_size2, output_size, dropout_prob, regularization)

    # Train the neural network
    train_x, train_y = _pre_processing(X_train, y_train_encoded)

    model.train(train_x, train_y, num_epochs, init_learning_rate, batch_size, learning_rate_decay)

    # Make predictions on the training set
    train_predictions = model.predict(X_train)
    train_accuracy = np.mean(train_predictions == np.argmax(y_train_encoded, axis=1)) * 100
    print(f"Training Accuracy at the end {train_accuracy}%")

    # Make predictions on the validation set
    validate_predictions = model.predict(X_validate)
    validate_accuracy = np.mean(validate_predictions == np.argmax(y_validate_encoded, axis=1)) * 100
    print(f"Validation Accuracy: at the end {validate_accuracy}%")


if __name__ == "__main__":
    main()

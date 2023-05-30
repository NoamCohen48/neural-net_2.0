import numpy as np
import pandas as pd

# Load the training and validation datasets
train_data = pd.read_csv('data/train.csv')
validate_data = pd.read_csv('data/validate.csv')

# Separate the features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_validate = validate_data.iloc[:, 1:].values
y_validate = validate_data.iloc[:, 0].values

y_train = y_train - 1
y_validate = y_validate - 1

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))

y_train_encoded = np.eye(num_classes)[y_train]
y_validate_encoded = np.eye(num_classes)[y_validate]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def RELU(x):
    return np.maximum(0, x)


def RELU_derivative(x):
    return np.where(x > 0, 1, 0)


ActivationFunction = sigmoid
ActivationFunctionDerivative = sigmoid_derivative


# Define the neural network model
class NeuralNetwork2:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ActivationFunction(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]

        delta2 = output - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * ActivationFunctionDerivative(self.z1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, num_epochs, learning_rate, batch_size):
        for epoch in range(num_epochs):
            # Shuffle the training data
            # indices = np.random.permutation(X.shape[0])
            indices = np.arange(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Divide the data into mini-batches
            num_batches = X.shape[0] // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                output = self.forward(X_batch)
                # print(np.argmax(output, axis=1))

                # Backward pass
                self.backward(X_batch, y_batch, output, learning_rate)

            # Print progress every 100 epochs
            if epoch % 5 == 0:
                validate_predictions = self.predict(X_validate)
                validate_accuracy = np.mean(validate_predictions == np.argmax(y_validate_encoded, axis=1)) * 100
                print("Validation Accuracy: {:.2f}%".format(validate_accuracy))

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


def main():
    np.random.seed(10)
    # Set the hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = num_classes
    num_epochs = 20
    learning_rate = 0.002
    batch_size = 32

    # Create the neural network model
    model = NeuralNetwork2(input_size, hidden_size, output_size)

    # Train the neural network
    model.train(X_train, y_train_encoded, num_epochs, learning_rate, batch_size)

    # Make predictions on the training set
    train_predictions = model.predict(X_train)
    train_accuracy = np.mean(train_predictions == np.argmax(y_train_encoded, axis=1)) * 100
    print("Training Accuracy: {:.2f}%".format(train_accuracy))

    # Make predictions on the validation set
    validate_predictions = model.predict(X_validate)
    validate_accuracy = np.mean(validate_predictions == np.argmax(y_validate_encoded, axis=1)) * 100
    print("Validation Accuracy: {:.2f}%".format(validate_accuracy))


if __name__ == "__main__":
    main()

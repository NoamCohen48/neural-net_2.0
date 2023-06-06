import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the training and validation datasets
train_data = pd.read_csv(f"{sys.argv[1]}/train.csv", header=None)
validate_data = pd.read_csv(f"{sys.argv[1]}/validate.csv", header=None)

# Separate the features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values - 1
X_validate = validate_data.iloc[:, 1:].values
y_validate = validate_data.iloc[:, 0].values - 1

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


def leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha * x)


def leaky_relu_prime(x, alpha=0.01):
    return np.where(x >= 0, 1, alpha)


ActivationFunction = leaky_relu
ActivationFunctionDerivative = leaky_relu_prime


# Define the neural network model
class NeuralNetwork2:
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob

        self.W1 = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.train_accuracy = []
        self.valid_accuracy = []
        self.epochs = []

        plt.ion()
        # Create a figure and axis for the plot
        fig, ax = plt.subplots()
        ax.set_xlabel('X')
        ax.set_ylabel('accuracy')
        ax.set_title('Data Points')

    def forward(self, X, training):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ActivationFunction(self.z1)

        # Apply dropout during training
        if training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=self.a1.shape)
            self.a1 *= dropout_mask / (1 - self.dropout_prob)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]

        delta2 = output - y
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Backpropagation dropout mask
        dropout_mask = np.random.binomial(1, 1 - self.dropout_prob, size=self.a1.shape)

        delta1 = np.dot(delta2, self.W2.T) * ActivationFunctionDerivative(self.z1) * dropout_mask / (
                1 - self.dropout_prob)

        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, num_epochs, initial_learning_rate, batch_size, decay_rate):
        learning_rate = initial_learning_rate

        for epoch in range(num_epochs):
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

                # Calculate training accuracy on a random subset of 1000 different examples
                random_indices = np.random.choice(X.shape[0], size=1000)
                X_train_subset = X[random_indices]
                y_train_subset = y[random_indices]
                train_predictions = self.predict(X_train_subset)
                train_accuracy = np.mean(train_predictions == np.argmax(y_train_subset, axis=1)) * 100

                self.train_accuracy.append(train_accuracy)
                self.valid_accuracy.append(validate_accuracy)
                self.epochs.append(epoch)

                print(f"After {epoch + 1} epochs, Training Accuracy (Subset of 1000): {train_accuracy}%")
                print(f"After {epoch + 1} epochs, Validation Accuracy : {validate_accuracy}%")

        self.plot()
        plt.ioff()
        plt.show()

    def predict(self, X):
        output = self.forward(X, False)
        return np.argmax(output, axis=1)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def plot(self):
        plt.scatter(self.epochs, self.train_accuracy, color='red', s=50)
        plt.plot(self.epochs, self.train_accuracy)
        for x_val, y_val in zip(self.epochs, self.train_accuracy):
            plt.text(x_val, y_val + 1, f'{y_val:.2f}', color='black', ha='center')
        plt.pause(0.1)


def show_image(x):
    plt.imshow(x)
    plt.title("RGB Image")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.colorbar()
    plt.axis("off")  # Remove the axis labels and ticks
    plt.show()


def _pre_processing(X: np.ndarray, Y: np.ndarray, reset_percentage: float = 0.2, noise_std: float = 0.1):
    # min_vals = np.min(X, axis=1)
    # max_vals = np.max(X, axis=1)
    # # Normalize each image individually using min-max normalization
    # X = (X - min_vals[:, np.newaxis]) / (max_vals - min_vals)[:, np.newaxis]

    X_reshaped = X.reshape((-1, 32, 32, 3))
    show_image(X_reshaped[0])

    mean = np.mean(X_reshaped, axis=(0, 1, 2), keepdims=True)
    std = np.std(X_reshaped, axis=(0, 1, 2), keepdims=True)
    X_reshaped = (X_reshaped - mean) / std

    # Flipping Images horizontally
    horizontal_flipped = np.flip(X_reshaped, axis=2)
    horizontal_flipped = horizontal_flipped.reshape(X.shape[0], -1)

    # Flipping Images vertically
    vertically_flipped = np.flip(X_reshaped, axis=1)
    vertically_flipped = vertically_flipped.reshape(X.shape[0], -1)

    # Flipping Images vertically and horizontally
    v_h_flipped = np.flip(X_reshaped, axis=(1, 2))
    v_h_flipped = v_h_flipped.reshape(X.shape[0], -1)

    X_new = np.concatenate((X, horizontal_flipped, vertically_flipped, v_h_flipped))
    Y_new = np.concatenate((Y, Y, Y, Y))

    # Resetting Pixels
    pixels_num = int(X.shape[1] * reset_percentage)
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


def load_model():
    with open("model.pickle", 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    print("loaded from pickle")
    return model


def main():
    np.random.seed(10)
    np.seterr(all="raise")
    # Set the hyperparameters
    input_size = X_train.shape[1]
    hidden_size = 256
    output_size = 10
    num_epochs = 100
    init_learning_rate = 0.05
    learning_rate_decay = 0.9
    batch_size = 32
    dropout_prob = 0.2

    # Create the neural network model
    model = NeuralNetwork2(input_size, hidden_size, output_size, dropout_prob)

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

    # Save predictions on the Test set
    test_predictions = model.predict(X_validate)
    with open("test-results.txt", "w") as f:
        for pred in test_predictions:
            f.write(f"{pred + 1}\n")
    print("saved test results")

    with open("model.pickle", 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    print("saved pickle")


if __name__ == "__main__":
    main()

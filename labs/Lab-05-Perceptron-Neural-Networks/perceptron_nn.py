import numpy as np

# Activation Function
def sigmoid(x):
    "Sigmoid activation function: f(x) = 1 / (1 + e^(-x)) It squashes values to be between 0 and 1."
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    "Derivative of the sigmoid function. This is needed for backpropagation. The derivative is: f'(x) = f(x) * (1 - f(x))"
    return x * (1 - x)


# Set Parameters and Initialize Network

input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 1000
n_samples = 500

# Initialize weights and biases with small random values

# Weights for Input -> Hidden Layer
W1 = np.random.randn(input_size, hidden_size)

# Bias for Hidden Layer
b1 = np.zeros((1, hidden_size))

# Weights for Hidden -> Output Layer
W2 = np.random.randn(hidden_size, output_size)

# Bias for Output Layer
b2 = np.zeros((1, output_size))

#Generate Dataset
# Create 500 random (x, y) points, scaled between -1 and 1
X_train = np.random.rand(n_samples, 2) * 2 - 1

# Create labels (y): 1 if inside a circle, 0 if outside

radius = 0.7
# Checks if the point is inside the circle
y_train = (np.sqrt(X_train[:, 0] ** 2 + X_train[:, 1] ** 2) < radius)
# Convert boolean (True/False) to integer (1/0) and make it a column vector
y_train = y_train.reshape(-1, 1).astype(int)

# Training the Neural Network

print("Starting training...")

for i in range(epochs):

    # "prediction" step.

    # 1. Input to Hidden Layer
    Z1 = np.dot(X_train, W1) + b1

    # 2. Hidden Layer Activation (using sigmoid)
    A1 = sigmoid(Z1)

    # 3. Hidden to Output Layer
    Z2 = np.dot(A1, W2) + b2

    # 4. Final Output Activation (using sigmoid)
    A2 = sigmoid(Z2)

    # 4b. Calculate the Loss
    # It measures how "wrong" our predictions (A2) are compared to the truth (y_train).
    # We add a small value (1e-9) to prevent log(0) errors.
    loss = -np.mean(y_train * np.log(A2 + 1e-9) + (1 - y_train) * np.log(1 - A2 + 1e-9))

    if (i + 1) % 100 == 0:
        # Print the loss every 100 epochs to see progress
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss:.4f}")

    # 4c. Backward Propagation
    # This is the "learning" step. We calculate how much each weight and bias contributed to the error, working backward.

    # 1. Error at the Output Layer
    d_Z2 = A2 - y_train

    # 2. Gradients for W2 and b2
    # How much W2 contributed to the error
    d_W2 = np.dot(A1.T, d_Z2) / n_samples
    # How much b2 contributed to the error
    d_b2 = np.sum(d_Z2, axis=0, keepdims=True) / n_samples

    # 3. Error at the Hidden Layer
    # Propagate the error back to the hidden layer
    d_A1 = np.dot(d_Z2, W2.T)
    # Apply the sigmoid derivative
    d_Z1 = d_A1 * sigmoid_derivative(A1)

    # 4. Gradients for W1 and b1
    # How much W1 contributed to the error
    d_W1 = np.dot(X_train.T, d_Z1) / n_samples
    # How much b1 contributed to the error
    d_b1 = np.sum(d_Z1, axis=0, keepdims=True) / n_samples

    # 4d. Update Parameters
    # Adjust the weights and biases to reduce the error.
    # We move them in the *opposite* direction of their gradient.

    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

print("Training finished!")

# Test the Network ---

print("Testing network on new data...")

# Generate a new test dataset
n_test_samples = 100
X_test = np.random.rand(n_test_samples, 2) * 2 - 1
y_test = (np.sqrt(X_test[:, 0] ** 2 + X_test[:, 1] ** 2) < radius)
y_test = y_test.reshape(-1, 1).astype(int)

# 5a. Make Predictions
# We use our *trained* weights (W1, b1, W2, b2)
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2

# Final probabilities
A2_test = sigmoid(Z2_test)

# Convert probabilities to binary class (1)
# If A2_test > 0.5, it's class 1, otherwise class 0
predictions = (A2_test > 0.5).astype(int)

# Report Accuracy
# Accuracy = (Correct Predictions / Total Predictions) * 100
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy on test dataset: {accuracy:.2f}%")

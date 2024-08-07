from microbit import *
import random

# Generate sample data
random.seed(42)
X = [i for i in range(100)]
Y = [2 * x + 1 + random.uniform(-1, 1) for x in X]

# Convert lists to floats for calculation purposes
X = [float(x) for x in X]
Y = [float(y) for y in Y]

# Function to split the data into train and test sets
def train_test_split(X, Y, test_size=0.3):
    total_size = len(X)
    test_size = int(total_size * test_size)
    indices = list(range(total_size))
    
    # Manually select indices for test set
    X_test = [X[i] for i in indices[:test_size]]
    Y_test = [Y[i] for i in indices[:test_size]]
    
    # Remaining indices for train set
    X_train = [X[i] for i in indices[test_size:]]
    Y_train = [Y[i] for i in indices[test_size:]]
    
    return X_train, X_test, Y_train, Y_test

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Function to calculate predictions
def predict(X, b0, b1):
    return [b0 + b1 * x for x in X]

# Function to calculate Mean Squared Error
def mean_squared_error(Y_true, Y_pred):
    return sum((Y_true[i] - Y_pred[i]) ** 2 for i in range(len(Y_true))) / len(Y_true)

# Function to calculate percentage error
def percentage_error(Y_true, Y_pred):
    errors = [abs(Y_true[i] - Y_pred[i]) / Y_true[i] * 100 for i in range(len(Y_true)) if Y_true[i] != 0]
    return sum(errors) / len(errors) if errors else 0

# Gradient Descent Algorithm
def gradient_descent(X, Y, b0, b1, learning_rate_b0, learning_rate_b1, epochs):
    n = len(X)
    for epoch in range(epochs):
        Y_pred = predict(X, b0, b1)
        
        # Compute gradients
        b0_gradient = -(2/n) * sum(Y[i] - Y_pred[i] for i in range(n))
        b1_gradient = -(2/n) * sum((Y[i] - Y_pred[i]) * X[i] for i in range(n))
        
        # Check for NaN values in gradients
        if any(map(lambda x: x != x, [b0_gradient, b1_gradient])):
            print("NaN gradient detected!")
            break
        
        # Gradient clipping to avoid extreme values
        b0_gradient = max(min(b0_gradient, 1e10), -1e10)
        b1_gradient = max(min(b1_gradient, 1e10), -1e10)
        
        # Update parameters with separate learning rates
        b0 -= learning_rate_b0 * b0_gradient
        b1 -= learning_rate_b1 * b1_gradient
        
        # Print gradients and parameters every 100 epochs
        if epoch % 100 == 0:
            print("Epoch {}: b0 = {:.4f}, b1 = {:.4f}".format(epoch, b0, b1))
    
    return b0, b1

# Function to display a number on the serial monitor
def display_number(number):
    print(number)

# Main loop
b0, b1 = 0.0, 0.0  # Initialize coefficients
learning_rate_b0 = 0.01  # Learning rate for b0
learning_rate_b1 = 0.0001  # Learning rate for b1
epochs = 4000  # Number of epochs

while True:
    if button_a.is_pressed():
        # Train the model using gradient descent when button A is pressed
        b0, b1 = gradient_descent(X_train, Y_train, b0, b1, learning_rate_b0, learning_rate_b1, epochs)
        print("Trained")
        sleep(500)  # Wait for 0.5 seconds

    if button_b.is_pressed():
        # Display coefficients when button B is pressed
        print("b0: {:.4f}".format(b0))
        print("b1: {:.4f}".format(b1))
        
        # Predict on test data and calculate percentage error
        Y_pred_test = predict(X_test, b0, b1)
        perc_error = percentage_error(Y_test, Y_pred_test)
        
        print("Error: {:.2f}%".format(perc_error))
        sleep(500)  # Wait for 0.5 seconds

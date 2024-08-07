from microbit import *

# Generate synthetic data representing luminosity and temperature
# Format: [luminosity, temperature, class]
# Classes: 0 = Main Sequence, 1 = Giant, 2 = Supergiant, 3 = White Dwarf
data = [
    [0.8, 5800, 0], [1.5, 6000, 0], [0.6, 5200, 0], [0.9, 5900, 0], [1.2, 6100, 0], [1.0, 5800, 0],
    [1000, 4500, 1], [2000, 4800, 1], [1500, 4600, 1], [1800, 4700, 1], [1700, 4650, 1], [1600, 4550, 1],
    [30000, 8000, 2], [50000, 8500, 2], [40000, 8200, 2], [45000, 8300, 2], [48000, 8400, 2], [35000, 8100, 2],
    [0.1, 10000, 3], [0.05, 12000, 3], [0.08, 11000, 3], [0.09, 11500, 3], [0.07, 10500, 3], [0.06, 12500, 3]
]

# Convert lists to floats for calculation purposes
for i in range(len(data)):
    data[i][0] = float(data[i][0])  # Luminosity
    data[i][1] = float(data[i][1])  # Temperature

# Normalize the data
def normalize_data(data):
    luminosities = [row[0] for row in data]
    temperatures = [row[1] for row in data]
    lum_min, lum_max = min(luminosities), max(luminosities)
    temp_min, temp_max = min(temperatures), max(temperatures)

    for row in data:
        row[0] = (row[0] - lum_min) / (lum_max - lum_min)  # Normalize luminosity
        row[1] = (row[1] - temp_min) / (temp_max - temp_min)  # Normalize temperature

    return data, lum_min, lum_max, temp_min, temp_max

data, lum_min, lum_max, temp_min, temp_max = normalize_data(data)

# Split the data into train and test sets
def train_test_split(data, test_size=0.5):
    total_size = len(data)
    test_size = int(total_size * test_size)
    indices = list(range(total_size))
    
    # Manually select indices for test set
    test_data = [data[i] for i in indices[:test_size]]
    
    # Remaining indices for train set
    train_data = [data[i] for i in indices[test_size:]]
    
    return train_data, test_data

# Split the data
train_data, test_data = train_test_split(data, test_size=0.3)

# Linear classification functions
def predict(X, weights):
    # Calculate the dot product manually
    prediction = 0.0
    for i in range(len(X)):
        prediction += X[i] * weights[i]
    return prediction

def train_linear_classifier(train_data, learning_rate, epochs):
    weights = [0.0, 0.0]  # Initialize weights for [luminosity, temperature]
    for epoch in range(epochs):
        for row in train_data:
            X = [row[0], row[1]]  # Features
            y = row[2]
            prediction = predict(X, weights)
            error = y - prediction
            # Update weights
            for i in range(len(weights)):
                weights[i] += learning_rate * error * X[i]
        if epoch % 100 == 0:
            print("Epoch {}: weights = {}".format(epoch, weights))
    return weights

def calculate_accuracy(test_data, weights):
    correct_predictions = 0
    total_predictions = len(test_data)

    for row in test_data:
        X = [row[0], row[1]]  # Features
        y = row[2]
        prediction = predict(X, weights)
        predicted_class = int(prediction)  # Round to nearest class (0, 1, 2, or 3)
        if predicted_class == y:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Classification function
def classify_planet(luminosity, temperature, weights):
    X = [luminosity, temperature]  # Features
    prediction = predict(X, weights)
    if prediction < 0.5:
        return "Main Sequence"
    elif prediction < 1.5:
        return "Giant"
    elif prediction < 2.5:
        return "Supergiant"
    else:
        return "White Dwarf"

# Function to read brightness and temperature from sliders
def read_brightness_temperature():
    # Read the brightness and temperature values from the sliders
    measured_brightness = display.read_light_level()  # Assume brightness is connected to pin0
    measured_temperature = temperature()  # Get the temperature from the micro:bit sensor

    # Normalize brightness to 0-1 range
    brightness_normalized = measured_brightness / 255.0
    # Normalize temperature to 0-1 range (assuming -5 to 50 degrees range)
    temperature_normalized = (measured_temperature + 5) / 55.0

    return brightness_normalized, temperature_normalized

# Main loop
learning_rate = 0.01  # Adjusted learning rate
epochs = 1000  # Number of epochs

weights = [0.0, 0.0]  # Initialize weights

while True:
    if button_a.is_pressed():
        # Train the model when button A is pressed
        weights = train_linear_classifier(train_data, learning_rate, epochs)
        accuracy = calculate_accuracy(test_data, weights)
        print("Model trained")
        print("Test Accuracy: {:.2f}%".format(accuracy))
        sleep(500)  # Wait for 0.5 seconds

    if button_b.is_pressed():
        # Classify a sample when button B is pressed
        sample_brightness, sample_temperature = read_brightness_temperature()
        classification = classify_planet(sample_brightness, sample_temperature, weights)
        print("Classification: {}".format(classification))
        sleep(500)  # Wait for 0.5 seconds

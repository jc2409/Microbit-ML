# Machine Learning on Microbit

Welcome to the Machine Learning on Microbit project! This project allows you to simulate basic machine learning tasks using the Microbit Python editor. You can train models, test them, and see how they perform in real-time on the Microbit. This README file will guide you through how to use the provided exercises and extend them further.

## Features

### Regression Exercise
1. **Train the Model**: 
   - Press the A button on the Microbit to start training the regression model.
   - The training progress and updates to the model parameters will be displayed in the serial monitor.
2. **Test the Model**: 
   - Press the B button to test the trained model.
   - The percentage error will be calculated and displayed, showing the model's performance.

### Classification Exercise
1. **Train the Model**:
   - Press the A button on the Microbit to start training the classification model.
   - The accuracy of the model on the test dataset will be displayed when the training is completed.
2. **Test the Model**:
   - Use the brightness and temperature sliders in the Microbit simulator to set input values.
   - The model will classify the input values into one of the star life cycle stages (Main Sequence, Giant, Supergiant, or White Dwarf).
   - The classification result will be displayed in the serial monitor.

## Extending the Exercise
To push this exercise further, you can:
- **Split the Dataset**: Divide the dataset into training, validation, and test sets. A common ratio is 8:1:1, but feel free to experiment with different ratios.
- **Hyperparameter Tuning**: Experiment with different hyperparameters such as the number of epochs, learning rate, etc., to improve model performance.
- **Expand the Dataset**: The provided dataset is small. Consider adding more data points manually to improve the model's accuracy and robustness.

## Getting Started
1. **Setup**: Open the Microbit Python editor.
2. **Load the Code**: Copy and paste the provided code into the editor.
3. **Train the Model**: Press the A button on the Microbit to start training.
4. **Test the Model**: Adjust the brightness and temperature sliders and press the B button to see the classification result.

## Dataset Information
The dataset consists of luminosity and temperature values representing different types of stars. The classes are:
- 0: Main Sequence
- 1: Giant
- 2: Supergiant
- 3: White Dwarf

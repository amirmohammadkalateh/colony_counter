Bacterial Colony Counting with an Artificial Neural Network
Overview
This project provides a Python implementation for counting bacterial colonies in images using an Artificial Neural Network (ANN).  The code utilizes TensorFlow and Keras for building and training the neural network, and OpenCV for image processing.

Features
Image Loading and Preprocessing: Loads images from specified paths, converts them to grayscale, resizes them, and normalizes pixel values.

Neural Network Model: Implements a simple feedforward neural network (ANN) using Keras to predict colony counts.

Training and Evaluation: Splits the data into training, validation, and test sets, trains the model, and evaluates its performance.

Colony Count Prediction: Predicts the colony count for a single input image.

Error Handling: Includes error handling for image loading and data processing.

Requirements
Python 3.x

TensorFlow (>=2.0)

Keras (comes with TensorFlow 2.x)

OpenCV (cv2)

NumPy

scikit-learn

Installation
Install Python 3.x.

Install the required packages.  It is highly recommended to use a virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate  # On Windows
pip install tensorflow opencv-python numpy scikit-learn

Usage
Prepare your data:

Collect a dataset of bacterial colony images.

Create a list of image paths and a corresponding list of colony counts.

Modify the main() function:

Update the image_paths list with the paths to your image files.

Update the colony_counts list with the corresponding colony counts for each image.

Run the main() function:

python your_script_name.py

(Replace your_script_name.py with the name of your Python file.)

Code Description
load_and_preprocess_data(image_paths, colony_counts, image_size=(128, 128))
Loads images from the given paths.

Converts images to grayscale.

Resizes images to the specified image_size.

Normalizes pixel values to the range [0, 1].

Returns the preprocessed image data and colony counts as NumPy arrays.

create_model(input_shape)
Creates a feedforward neural network model using Keras.

The model architecture includes:

A flatten layer to convert the 2D image data to a 1D vector.

Two dense layers with ReLU activation.

An output layer with linear activation for regression.

The model is compiled with the Adam optimizer and mean squared error loss.

train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
Trains the neural network model using the provided training data.

Supports training with a validation set.

Returns the training history.

evaluate_model(model, X_test, y_test)
Evaluates the trained model on the test data.

Prints the test loss and mean absolute error.

predict_colony_count(model, image_path, image_size=(128, 128))
Predicts the colony count for a single image.

Loads and preprocesses the image.

Uses the trained model to make the prediction.

Returns the predicted colony count.

main()
The main function that orchestrates the colony counting process.

Prepares the data, loads and preprocesses the images, splits the data, creates and trains the model, evaluates the model, and makes a prediction for a single image.

Notes
Ensure that the image paths in the image_paths list are correct and that the images exist at those locations.

The colony_counts list should have the same number of elements as the image_paths list.

For more complex images or higher accuracy, consider using a Convolutional Neural Network (CNN) instead of the feedforward ANN.

Adjust the epochs and batch_size parameters in the train_model function for optimal training.
